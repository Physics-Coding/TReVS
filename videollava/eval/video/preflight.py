import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

from safetensors import safe_open

from .datasets import DATASETS, get_dataset_paths, resolve_video_path


CANONICAL_COUNTS = {"tgif": 25751, "msvd": 13157, "msrvtt": 72821}
EXPECTED_VIDEO_TENSORS = {
    "model.video_tower.video_tower.embeddings.class_embedding": (1024,),
    "model.video_tower.video_tower.embeddings.patch_embedding.weight": (1024, 3, 14, 14),
    "model.video_tower.video_tower.embeddings.position_embedding.weight": (257, 1024),
    "model.video_tower.video_tower.encoder.layers.0.self_attn.q_proj.weight": (1024, 1024),
    "model.video_tower.video_tower.encoder.layers.0.mlp.fc1.weight": (4096, 1024),
    "model.video_tower.video_tower.encoder.layers.0.temporal_embedding": (1, 8, 1024),
    "model.video_tower.video_tower.encoder.layers.23.temporal_attn.q_proj.weight": (1024, 1024),
}


def _load_json_list(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return value


def validate_checkpoint(model_path: Path) -> dict:
    required_files = (
        "config.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    )
    missing_files = [name for name in required_files if not (model_path / name).is_file()]
    if missing_files:
        raise FileNotFoundError(f"Checkpoint is missing required files: {missing_files}")

    with (model_path / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    expected_config = {
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "mm_hidden_size": 1024,
        "mm_vision_select_layer": -2,
    }
    mismatches = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in expected_config.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Unexpected Video-LLaVA config values: {mismatches}")

    with (model_path / "model.safetensors.index.json").open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map", {})
    if not isinstance(weight_map, dict):
        raise ValueError("model.safetensors.index.json does not contain a weight_map object")
    missing_weights = [name for name in EXPECTED_VIDEO_TENSORS if name not in weight_map]
    if missing_weights:
        raise ValueError(f"Checkpoint is missing embedded video tower weights: {missing_weights}")

    shard_handles: Dict[str, object] = {}
    try:
        for tensor_name, expected_shape in EXPECTED_VIDEO_TENSORS.items():
            shard_name = weight_map[tensor_name]
            shard_path = model_path / shard_name
            if not shard_path.is_file():
                raise FileNotFoundError(f"Missing checkpoint shard: {shard_path}")
            if shard_name not in shard_handles:
                shard_handles[shard_name] = safe_open(str(shard_path), framework="pt", device="cpu")
            actual_shape = tuple(shard_handles[shard_name].get_slice(tensor_name).get_shape())
            if actual_shape != expected_shape:
                raise ValueError(
                    f"Unexpected shape for {tensor_name}: expected {expected_shape}, got {actual_shape}"
                )
    finally:
        shard_handles.clear()

    return {
        "path": str(model_path),
        "weight_count": len(weight_map),
        "total_size_bytes": index.get("metadata", {}).get("total_size"),
        "embedded_video_tower": True,
    }


def validate_dataset(
    data_root: Path,
    dataset: str,
    *,
    require_canonical_count: bool = True,
    decode_sample: bool = False,
) -> dict:
    video_dir, question_file, answer_file = get_dataset_paths(data_root, dataset)
    for path in (video_dir, question_file, answer_file):
        if not path.exists():
            raise FileNotFoundError(path)
    questions = _load_json_list(question_file)
    answers = _load_json_list(answer_file)
    if len(questions) != len(answers):
        raise ValueError(
            f"{dataset}: question/answer length mismatch: {len(questions)} != {len(answers)}"
        )
    if require_canonical_count and len(questions) != CANONICAL_COUNTS[dataset]:
        raise ValueError(
            f"{dataset}: expected {CANONICAL_COUNTS[dataset]} packaged questions, got {len(questions)}"
        )

    question_ids = [str(item["question_id"]) for item in questions]
    answer_ids = [str(item["question_id"]) for item in answers]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError(f"{dataset}: duplicate question IDs")
    if len(answer_ids) != len(set(answer_ids)):
        raise ValueError(f"{dataset}: duplicate answer IDs")
    if set(question_ids) != set(answer_ids):
        raise ValueError(f"{dataset}: question and answer ID sets differ")

    video_paths: Dict[str, Path] = {}
    for item in questions:
        video_name = str(item["video_name"])
        if video_name not in video_paths:
            video_paths[video_name] = resolve_video_path(video_dir, video_name)

    decoded_shape = None
    deterministic = None
    if decode_sample:
        import torch

        from videollava.model.multimodal_encoder.languagebind.processing_video import (
            build_default_video_processor,
        )

        processor = build_default_video_processor(num_frames=8)
        sample_path = next(iter(video_paths.values()))
        first = processor.preprocess(str(sample_path), return_tensors="pt")["pixel_values"]
        second = processor.preprocess(str(sample_path), return_tensors="pt")["pixel_values"]
        decoded_shape = list(first.shape)
        deterministic = bool(torch.equal(first, second))
        if decoded_shape != [1, 3, 8, 224, 224]:
            raise ValueError(f"{dataset}: unexpected decoded shape {decoded_shape}")
        if not deterministic:
            raise ValueError(f"{dataset}: video preprocessing is not deterministic")

    return {
        "dataset": dataset,
        "question_count": len(questions),
        "answer_count": len(answers),
        "unique_video_count": len(video_paths),
        "decoded_shape": decoded_shape,
        "deterministic_decode": deterministic,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate local Video-LLaVA checkpoint and VideoQA data.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument("--decode-samples", action="store_true")
    parser.add_argument("--allow-noncanonical-counts", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def run_preflight(args: argparse.Namespace) -> dict:
    result = {
        "checkpoint": validate_checkpoint(args.model_path),
        "datasets": [
            validate_dataset(
                args.data_root,
                dataset,
                require_canonical_count=not args.allow_noncanonical_counts,
                decode_sample=args.decode_samples,
            )
            for dataset in args.datasets
        ],
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    result = run_preflight(parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
