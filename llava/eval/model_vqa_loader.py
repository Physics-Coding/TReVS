import argparse
import json
import datetime
import math
import os
import random
import time

import shortuuid
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.eval.efficiency_utils import (
    EfficiencyRuntimeContext,
    FirstGeneratedTokenStreamer,
    PopeEfficiencyAggregator,
    build_sample_efficiency_record,
    collect_runtime_metrics,
)
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
import logging

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


def _parse_optional_bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "")
    if raw is None:
        return default
    value = raw.strip().lower()
    if value == "":
        return default
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid {name}={raw!r}; expected a boolean-like value.")


def seed_everything(seed: int) -> None:
    if seed is None:
        return
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_worker_init_fn(base_seed: int):
    if base_seed is None:
        return None

    def _seed_worker(worker_id: int):
        worker_seed = int(base_seed) + int(worker_id)
        random.seed(worker_seed)
        if np is not None:
            np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _seed_worker


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        question_text = line["text"]
        qs = question_text
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    worker_init_fn = _make_worker_init_fn(getattr(args, "seed", None))
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
    )
    return data_loader

def eval_model(args):
    seed = args.seed
    if seed is None:
        env_seed = os.environ.get("RANDOM_SEED", None)
        seed = int(env_seed) if env_seed is not None and env_seed != "" else None
        args.seed = seed
    if seed is not None:
        print(f"[model_vqa_loader] Using seed={seed}")
    seed_everything(seed)

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    use_flash_attn = args.use_flash_attn
    if use_flash_attn is None:
        use_flash_attn = _parse_optional_bool_env("USE_FLASH_ATTN", default=False)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        use_flash_attn=use_flash_attn,
    )
    model.eval()

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    enable_efficiency_benchmark = args.enable_efficiency_benchmark
    efficiency_log_dir = os.path.expanduser(args.efficiency_log_dir) if args.efficiency_log_dir else None
    efficiency_run_name = args.efficiency_run_name or os.path.splitext(os.path.basename(answers_file))[0]
    efficiency_aggregator = PopeEfficiencyAggregator() if enable_efficiency_benchmark else None
    phase_transition_layer = int(os.environ.get("PHASE_TRANSITION_LAYER", "8"))

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)
    
    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]
        category = line.get("category", "unknown")

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        with torch.inference_mode():
            streamer = None
            if enable_efficiency_benchmark:
                torch.cuda.reset_peak_memory_stats()
                model.get_model().efficiency_ctx = EfficiencyRuntimeContext()
                streamer = FirstGeneratedTokenStreamer()
                torch.cuda.synchronize()
                t_start = time.perf_counter()
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                streamer=streamer)
            if enable_efficiency_benchmark:
                torch.cuda.synchronize()
                t_end = time.perf_counter()
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        if enable_efficiency_benchmark:
            runtime_metrics = collect_runtime_metrics(getattr(model.get_model(), "efficiency_ctx", None))
            if streamer is None or streamer.first_token_time is None:
                raise RuntimeError("TTFT measurement failed because the first generated token timestamp was not captured.")
            total_time_ms = 1000.0 * (t_end - t_start)
            ttft_ms = 1000.0 * (streamer.first_token_time - t_start)
            peak_allocated_bytes = int(torch.cuda.max_memory_allocated())
            output_tokens = int(streamer.generated_token_count)
            record = build_sample_efficiency_record(
                question_id=str(idx),
                category=str(category),
                total_time_ms=total_time_ms,
                ttft_ms=ttft_ms,
                peak_gpu_memory_allocated_bytes=peak_allocated_bytes,
                output_tokens=output_tokens,
                runtime_metrics=runtime_metrics,
                model=model,
                phase_transition_layer=phase_transition_layer,
            )
            efficiency_aggregator.add(record)
            model.get_model().efficiency_ctx = None

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
    ans_file.close()

    if enable_efficiency_benchmark and efficiency_aggregator is not None and efficiency_log_dir is not None:
        os.makedirs(efficiency_log_dir, exist_ok=True)
        samples_path = os.path.join(efficiency_log_dir, f"{efficiency_run_name}_samples.jsonl")
        summary_path = os.path.join(efficiency_log_dir, f"{efficiency_run_name}_summary.json")
        report_path = os.path.join(efficiency_log_dir, f"{efficiency_run_name}_report.md")
        efficiency_aggregator.write_samples_jsonl(samples_path)
        efficiency_aggregator.write_summary_json(
            summary_path,
            extra={
                "dataset": "POPE" if "pope" in os.path.basename(args.question_file).lower() else os.path.basename(args.question_file),
                "model_name": model_name,
                "method": os.environ.get("METHOD", ""),
                "run_name": efficiency_run_name,
                "question_file": os.path.expanduser(args.question_file),
                "answers_file": answers_file,
            },
        )
        efficiency_aggregator.write_report_markdown(
            report_path,
            extra={
                "dataset": "POPE" if "pope" in os.path.basename(args.question_file).lower() else os.path.basename(args.question_file),
                "model_name": model_name,
                "method": os.environ.get("METHOD", ""),
                "run_name": efficiency_run_name,
                "question_file": os.path.expanduser(args.question_file),
                "answers_file": answers_file,
            },
        )

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=None, help="Overrides env RANDOM_SEED when set")
    parser.add_argument(
        "--use-flash-attn",
        dest="use_flash_attn",
        action="store_true",
        help="Enable flash_attention_2 when loading the model.",
    )
    parser.add_argument(
        "--no-use-flash-attn",
        dest="use_flash_attn",
        action="store_false",
        help="Disable flash_attention_2 when loading the model.",
    )
    parser.set_defaults(use_flash_attn=None)
    parser.add_argument(
        "--enable-efficiency-benchmark",
        action="store_true",
        help="Enable POPE efficiency benchmarking and write per-sample plus aggregate metrics.",
    )
    parser.add_argument(
        "--efficiency-log-dir",
        type=str,
        default=None,
        help="Directory used to store efficiency benchmark logs when enabled.",
    )
    parser.add_argument(
        "--efficiency-run-name",
        type=str,
        default=None,
        help="Optional run name prefix used for efficiency benchmark output files.",
    )
    args = parser.parse_args()

    eval_model(args)
