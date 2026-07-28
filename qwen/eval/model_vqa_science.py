import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from qwen.eval.attention_backend import configure_qwen_attention_backend
from qwen.model.qwen2_5_vl_custom import apply_qwen2_5_vl_trevs_patches
apply_qwen2_5_vl_trevs_patches()
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from PIL import Image
import math

from evaluation.runtime import seed_everything

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

def use_device_map_from_env():
    value = os.environ.get("USE_DEVICE_MAP", "0").strip().lower()
    return value in {"1", "true", "yes", "on", "auto"}

def get_progress_interval():
    return max(1, int(os.environ.get("QWEN_PROGRESS_INTERVAL", "10")))

def log_progress(prefix, current, total, answers_file):
    print(f"[{prefix}] processed {current}/{total} samples -> {answers_file}", flush=True)

def get_fixed_image_size():
    height = int(os.environ.get("QWEN_IMAGE_RESIZED_HEIGHT", "896"))
    width = int(os.environ.get("QWEN_IMAGE_RESIZED_WIDTH", "1120"))
    return height, width

def eval_model(args):
    args.seed = seed_everything(getattr(args, "seed", None))
    print(f"[model_vqa_science] Using seed={args.seed}", flush=True)
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    apply_qwen2_5_vl_trevs_patches()
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

    model_kwargs = {"torch_dtype": "auto"}
    if use_device_map_from_env():
        model_kwargs["device_map"] = "auto"
    configure_qwen_attention_backend(model_kwargs)
    print(f"[model_vqa_science] Loading model: {model_path}", flush=True)
    print(f"[model_vqa_science] model_kwargs={model_kwargs}", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
    if "device_map" not in model_kwargs and torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    print(f"[model_vqa_science] Model ready on device={model.device}", flush=True)
    
    min_pixels = 1280 * 28 * 28
    max_pixels = 1280 * 28 * 28
    fixed_height, fixed_width = get_fixed_image_size()
    print(
        f"[model_vqa_science] Loading processor with fixed visual tokens=1280, resized={fixed_width}x{fixed_height}",
        flush=True,
    )
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    progress_interval = get_progress_interval()
    print(
        f"[model_vqa_science] Start eval: total={len(questions)}, chunk={args.chunk_idx}/{args.num_chunks}, answers={answers_file}",
        flush=True,
    )

    for i, line in enumerate(tqdm(questions, dynamic_ncols=True, mininterval=1.0)):
        idx = line["id"]
        question = line['conversations'][0]
        qs = question['value'].replace('<image>', '').strip()
        cur_prompt = qs

        if args.single_pred_prompt:
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
            cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."

        if 'image' in line:
            image_file = line["image"]
            image_path = os.path.join(args.image_folder, image_file)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {"type": "text", "text": cur_prompt},
                    ],
                }
            ]
            cur_prompt = '<image>' + '\n' + cur_prompt
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": cur_prompt},
                    ],
                }
            ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                use_cache=os.environ.get("QWEN_USE_CACHE", "1").strip().lower() in {"1", "true", "yes", "on"},
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        outputs = output_text[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
        if (i + 1) % progress_interval == 0 or (i + 1) == len(questions):
            log_progress("model_vqa_science", i + 1, len(questions), answers_file)
    ans_file.close()
    print(f"[model_vqa_science] Finished eval. answers={answers_file}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=None, help="Overrides env RANDOM_SEED")
    parser.add_argument("--layer-list", type=str, default='[]')
    parser.add_argument("--image-token-ratio-list", type=str, default='[]')
    parser.add_argument("--image-token-ratio", type=float, default=1.0)
    args = parser.parse_args()

    eval_model(args)
