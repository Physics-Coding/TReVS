import argparse
import json
import os
import random
import traceback
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import torch
from tqdm import tqdm

from videollava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from videollava.conversation import SeparatorStyle, conv_templates
from videollava.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token
from videollava.model.builder import load_pretrained_model

from .datasets import chunk_sequence, iter_jsonl, resolve_video_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict, resumable Video-LLaVA VideoQA inference.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--gt-file-question", type=Path, required=True)
    parser.add_argument("--gt-file-answers", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conv-mode", default="llava_v1")
    return parser.parse_args()


def _load_json_list(path: Path) -> List[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return value


def load_annotation_chunk(args: argparse.Namespace) -> tuple[List[dict], Dict[str, object]]:
    questions = _load_json_list(args.gt_file_question)
    answers = _load_json_list(args.gt_file_answers)
    if args.max_samples > 0:
        questions = questions[: args.max_samples]

    question_ids = [str(item["question_id"]) for item in questions]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError(f"Duplicate question_id values in {args.gt_file_question}")

    answers_by_id: Dict[str, object] = {}
    for sample in answers:
        question_id = str(sample["question_id"])
        if question_id in answers_by_id:
            raise ValueError(f"Duplicate answer for question_id={question_id}")
        answers_by_id[question_id] = sample["answer"]

    missing_answers = [question_id for question_id in question_ids if question_id not in answers_by_id]
    if missing_answers:
        raise ValueError(
            f"Missing {len(missing_answers)} answers; examples={missing_answers[:5]}"
        )
    return chunk_sequence(questions, args.num_chunks, args.chunk_idx), answers_by_id


def _load_completed_ids(output_file: Path, allowed_ids: Set[str]) -> Set[str]:
    if not output_file.exists():
        return set()
    completed: Set[str] = set()
    for record in iter_jsonl(output_file):
        question_id = str(record.get("id", ""))
        if question_id not in allowed_ids:
            raise ValueError(
                f"Resume output {output_file} contains an ID outside this chunk: {question_id}"
            )
        if question_id in completed:
            raise ValueError(f"Duplicate ID in resume output {output_file}: {question_id}")
        completed.add(question_id)
    return completed


def _embedding_device(model: torch.nn.Module) -> torch.device:
    return model.get_input_embeddings().weight.device


def _video_device(model: torch.nn.Module) -> torch.device:
    tower = model.get_video_tower()
    return tower.device


def build_prompt(question: str, num_frames: int, conv_mode: str) -> tuple[str, object]:
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if conv_mode not in conv_templates:
        raise ValueError(f"Unknown conversation mode {conv_mode!r}")
    prompt_question = "".join([DEFAULT_IMAGE_TOKEN] * num_frames) + "\n" + question
    conversation = conv_templates[conv_mode].copy()
    conversation.append_message(conversation.roles[0], prompt_question)
    conversation.append_message(conversation.roles[1], None)
    return conversation.get_prompt(), conversation


def normalize_generated_answer(decoded: str, stop_string: str) -> str:
    """Remove a terminal stop sequence while retaining valid empty predictions."""
    output = decoded.strip()
    if stop_string and output.endswith(stop_string):
        output = output[: -len(stop_string)].strip()
    return output


def generate_answer(model, video_processor, tokenizer, video_path: Path, question: str, args) -> str:
    num_frames = int(video_processor.num_frames)
    prompt, conversation = build_prompt(question, num_frames=num_frames, conv_mode=args.conv_mode)
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0)
    placeholder_count = int((input_ids == IMAGE_TOKEN_INDEX).sum().item())
    if placeholder_count != num_frames:
        raise ValueError(
            f"Prompt contains {placeholder_count} visual placeholders but processor uses {num_frames} frames"
        )

    processed = video_processor.preprocess(str(video_path), return_tensors="pt")
    video_tensor = processed["pixel_values"][0]
    expected_shape = (3, num_frames, 224, 224)
    if tuple(video_tensor.shape) != expected_shape:
        raise ValueError(
            f"Unexpected processed video shape {tuple(video_tensor.shape)}; expected {expected_shape}"
        )

    embedding_device = _embedding_device(model)
    tower_device = _video_device(model)
    input_ids = input_ids.to(device=embedding_device, non_blocking=True)
    video_tensor = video_tensor.to(
        device=tower_device,
        dtype=model.get_video_tower().dtype,
        non_blocking=True,
    )

    stop_string = conversation.sep if conversation.sep_style != SeparatorStyle.TWO else conversation.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_string], tokenizer, input_ids)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[video_tensor],
            do_sample=False,
            num_beams=1,
            temperature=0.0,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
    if not getattr(model, "_videoqa_diagnostics_emitted", False):
        backbone = model.get_model()
        diagnostics = {
            "multimodal": getattr(backbone, "last_multimodal_metrics", {}),
            "phase": getattr(backbone, "last_trevs_metrics", {}),
            "layer_cache_lengths": getattr(backbone, "last_cache_lengths", []),
        }
        print("VIDEO_LLAVA_TREVS_DIAGNOSTICS=" + json.dumps(diagnostics, sort_keys=True))
        model._videoqa_diagnostics_emitted = True
    decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    # An empty decoded answer is a model prediction, not an inference failure.
    # It must reach the evaluator so it remains in the benchmark denominator.
    return normalize_generated_answer(decoded, stop_string)


def run_inference(args: argparse.Namespace) -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        torch.cuda.manual_seed_all(args.seed)

    questions, answers_by_id = load_annotation_chunk(args)
    allowed_ids = {str(item["question_id"]) for item in questions}
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        completed_ids = _load_completed_ids(args.output_file, allowed_ids)
        output_mode = "a"
    else:
        completed_ids = set()
        output_mode = "w"

    tokenizer, model, video_processor, _ = load_pretrained_model(
        model_path=str(args.model_path),
        device=args.device,
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    error_file = args.output_file.with_suffix(".errors.jsonl")

    written = 0
    with args.output_file.open(output_mode, encoding="utf-8") as output_handle:
        for sample in tqdm(questions, desc=f"chunk {args.chunk_idx}/{args.num_chunks}"):
            question_id = str(sample["question_id"])
            if question_id in completed_ids:
                continue
            answer = answers_by_id[question_id]
            try:
                video_path = resolve_video_path(args.video_dir, str(sample["video_name"]))
                prediction = generate_answer(
                    model=model,
                    video_processor=video_processor,
                    tokenizer=tokenizer,
                    video_path=video_path,
                    question=str(sample["question"]),
                    args=args,
                )
                record = {
                    "id": question_id,
                    "question": str(sample["question"]),
                    "answer": answer,
                    "pred": prediction,
                }
                output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_handle.flush()
                written += 1
            except Exception as exc:
                error_record = {
                    "id": question_id,
                    "video_name": sample.get("video_name"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                with error_file.open("a", encoding="utf-8") as error_handle:
                    error_handle.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                raise

    total_completed = len(completed_ids) + written
    if total_completed != len(questions):
        raise RuntimeError(
            f"Chunk incomplete: expected {len(questions)}, found {total_completed} completed predictions"
        )
    print(
        f"Completed chunk {args.chunk_idx}: {total_completed} predictions in {args.output_file} "
        f"({len(completed_ids)} resumed, {written} new)"
    )
    return total_completed


def main() -> None:
    run_inference(parse_args())


if __name__ == "__main__":
    main()
