#!/usr/bin/env python3
"""Validated public entry point for TReVS inference and evaluation."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PRESET_PATH = REPO_ROOT / "configs" / "presets.json"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
IMAGE_DATASETS = {
    "gqa": {
        "required": [
            "gqa/llava_gqa_testdev_balanced.jsonl",
            "gqa/images",
            "gqa/testdev_balanced_questions.json",
        ],
        "split": "testdev_balanced",
        "evaluator": "external_official",
    },
    "textvqa": {
        "required": [
            "textvqa/llava_textvqa_val_v051_ocr.jsonl",
            "textvqa/train_images",
            "textvqa/TextVQA_0.5.1_val.json",
        ],
        "split": "val",
        "evaluator": "m4c_textvqa",
    },
    "mme": {
        "required": ["mme/llava_mme.jsonl", "mme/MME_Benchmark"],
        "split": "test",
        "evaluator": "official_mme",
    },
    "mmbench": {
        "required": ["mmbench/mmbench_dev_20230712.tsv"],
        "split": "dev_20230712",
        "evaluator": "submission_only",
    },
    "mmbench_cn": {
        "required": ["mmbench/mmbench_dev_cn_20231003.tsv"],
        "split": "dev_cn_20231003",
        "evaluator": "submission_only",
    },
    "sqa": {
        "required": [
            "scienceqa/llava_test_CQM-A.json",
            "scienceqa/images/test",
            "scienceqa/pid_splits.json",
            "scienceqa/problems.json",
        ],
        "split": "test",
        "evaluator": "scienceqa",
    },
    "pope": {
        "required": [
            "pope/llava_pope_test.jsonl",
            "pope/coco/coco_pope_adversarial.json",
            "pope/coco/coco_pope_popular.json",
            "pope/coco/coco_pope_random.json",
            "coco/val2014",
        ],
        "split": "test",
        "evaluator": "pope",
    },
    "vqav2": {
        "required": [
            "vqav2/llava_vqav2_mscoco_test-dev2015.jsonl",
            "vqav2/llava_vqav2_mscoco_test2015.jsonl",
            "vqav2/test2015",
        ],
        "split": "test-dev2015",
        "evaluator": "submission_only",
    },
}
FILE_CONTRACTS: dict[str, dict[str, Any]] = {
    "gqa/llava_gqa_testdev_balanced.jsonl": {
        "container": "jsonl",
        "key_fields": ["question_id"],
        "expected_records": 12578,
    },
    "gqa/testdev_balanced_questions.json": {
        "container": "dict_keys",
        "expected_records": 12578,
    },
    "textvqa/llava_textvqa_val_v051_ocr.jsonl": {
        "container": "jsonl",
        "key_fields": ["question_id", "text"],
        "expected_records": 5000,
    },
    "textvqa/TextVQA_0.5.1_val.json": {
        "container": "data",
        "key_fields": ["image_id", "question"],
        "expected_records": 5000,
    },
    "mme/llava_mme.jsonl": {
        "container": "jsonl",
        "key_fields": ["question_id", "text"],
        "expected_records": 2374,
    },
    "mmbench/mmbench_dev_20230712.tsv": {
        "container": "tsv",
        "key_fields": ["index"],
        "expected_records": 4377,
        "embedded_image": "image",
    },
    "mmbench/mmbench_dev_cn_20231003.tsv": {
        "container": "tsv",
        "key_fields": ["index"],
        "expected_records": 4329,
        "embedded_image": "image",
    },
    "scienceqa/llava_test_CQM-A.json": {
        "container": "list",
        "key_fields": ["id"],
        "expected_records": 4241,
    },
    "scienceqa/pid_splits.json": {
        "container": "list_key",
        "list_key": "test",
        "expected_records": 4241,
    },
    "scienceqa/problems.json": {
        "container": "dict_keys",
        "minimum_records": 4241,
    },
    "pope/llava_pope_test.jsonl": {
        "container": "jsonl",
        "key_fields": ["question_id"],
        "expected_records": 8910,
    },
    "pope/coco/coco_pope_adversarial.json": {
        "container": "jsonl",
        "expected_records": 3000,
    },
    "pope/coco/coco_pope_popular.json": {
        "container": "jsonl",
        "expected_records": 3000,
    },
    "pope/coco/coco_pope_random.json": {
        "container": "jsonl",
        "expected_records": 2910,
    },
    "vqav2/llava_vqav2_mscoco_test-dev2015.jsonl": {
        "container": "jsonl",
        "key_fields": ["question_id"],
        "expected_records": 107394,
    },
    "vqav2/llava_vqav2_mscoco_test2015.jsonl": {
        "container": "jsonl",
        "key_fields": ["question_id"],
        "expected_records": 447793,
    },
}
ENVIRONMENT_CONTRACTS = {
    "llava_family": {
        "python": "3.10.19",
        "torch": "2.1.2",
        "torchvision": "0.16.2",
        "transformers": "4.37.2",
        "numpy": "1.26.4",
    },
    "qwen": {
        "python": "3.10.19",
        "torch": "2.5.1",
        "torchvision": "0.20.1",
        "transformers": "5.8.0.dev0",
        "numpy": "2.2.6",
        "transformers_commit": "a25b8efa0a3220da89493a72f57081aa5720291f",
    },
}
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_DATASETS = {
    "tgif": {
        "candidates": ["TGIF_Zero_Shot_QA", "GPT_Zero_Shot_QA/TGIF_Zero_Shot_QA"],
        "video_subdir": "mp4",
        "split": "zero_shot_test",
        "evaluator": "videoqa_gpt_judge",
    },
    "msvd": {
        "candidates": ["MSVD_Zero_Shot_QA", "GPT_Zero_Shot_QA/MSVD_Zero_Shot_QA"],
        "video_subdir": "videos",
        "split": "zero_shot_test",
        "evaluator": "videoqa_gpt_judge",
    },
    "msrvtt": {
        "candidates": ["MSRVTT_Zero_Shot_QA", "GPT_Zero_Shot_QA/MSRVTT_Zero_Shot_QA"],
        "video_subdir": "videos/all",
        "split": "zero_shot_test",
        "evaluator": "videoqa_gpt_judge",
    },
}


class ConfigurationError(ValueError):
    pass


def load_registry() -> dict[str, Any]:
    with PRESET_PATH.open("r", encoding="utf-8") as handle:
        registry = json.load(handle)
    validate_registry(registry)
    return registry


def validate_registry(registry: Mapping[str, Any]) -> None:
    families = registry.get("families")
    if not isinstance(families, dict) or not families:
        raise ConfigurationError("Preset registry has no families.")
    for family_name, family in families.items():
        layers = int(family["llm_layers"])
        crops = int(family.get("crops", 1))
        frames = int(family.get("frames", 1))
        multiplier = frames if family_name == "videollava" else crops
        for preset_name, preset in family["presets"].items():
            if preset["method"] == "dense" or preset_name == "custom":
                continue
            topk = int(preset["route_topk"])
            fps = int(preset["route_fps"])
            stage1 = multiplier * (topk + fps)
            layer = int(preset["phase_transition_layer"])
            keep = int(preset["phase_transition_keep"])
            average = (layer * stage1 + (layers - layer) * keep) / layers
            if stage1 != preset["stage1_visual_tokens"]:
                raise ConfigurationError(
                    f"{family_name}/{preset_name}: Stage-1 budget mismatch."
                )
            if abs(average - float(preset["average_visual_tokens"])) > 1e-9:
                raise ConfigurationError(
                    f"{family_name}/{preset_name}: weighted average mismatch."
                )


def parser_for(registry: Mapping[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an anonymous TReVS inference/evaluation preset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--family", required=True, choices=sorted(registry["families"]))
    parser.add_argument("--preset", required=True)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--datasets", default="all", help="Comma-separated dataset names or all")
    parser.add_argument(
        "--gpus",
        default="0",
        help="Comma-separated physical CUDA IDs; invoke from an unmasked process",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", help="Portable run identifier; defaults to a UTC timestamp")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Validate and print; write nothing")
    mode.add_argument("--preflight", action="store_true", help="Check local runtime and data; write nothing")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means the complete split")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--run-judge", action="store_true", help="Run the optional VideoQA judge")
    parser.add_argument("--stage1-topk", type=int, help="Video-LLaVA custom preset only")
    parser.add_argument("--stage1-fps", type=int, help="Video-LLaVA custom preset only")
    parser.add_argument("--stage2-keep", type=int, help="Video-LLaVA custom preset only")
    parser.add_argument("--phase-layer", type=int, default=8, help="Video-LLaVA custom preset only")
    return parser


def normalized_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_output_root(output_root: Path, data_root: Path, model_path: Path) -> None:
    protected = [data_root, model_path]
    protected.extend(
        REPO_ROOT / name
        for name in ("configs", "data", "evaluation", "llava", "qwen", "scripts", "videollava")
    )
    for path in protected:
        if output_root == path or is_relative_to(output_root, path):
            raise ConfigurationError(f"Output root must not be inside protected path: {path}")
    if output_root == REPO_ROOT:
        raise ConfigurationError("Output root must not be the repository root.")
    if is_relative_to(output_root, REPO_ROOT) and not is_relative_to(
        output_root, REPO_ROOT / "outputs"
    ):
        raise ConfigurationError("Repository-local output is allowed only below outputs/.")


def parse_datasets(requested: str, supported: Sequence[str]) -> list[str]:
    aliases = {"scienceqa": "sqa", "mmbench-cn": "mmbench_cn"}
    raw = [aliases.get(item.strip().lower(), item.strip().lower()) for item in requested.split(",")]
    if raw == ["all"]:
        return list(supported)
    if not raw or any(not item for item in raw):
        raise ConfigurationError("--datasets must not be empty.")
    if "all" in raw:
        raise ConfigurationError("Dataset 'all' cannot be combined with individual datasets.")
    unknown = sorted(set(raw).difference(supported))
    if unknown:
        raise ConfigurationError("Unsupported datasets: " + ", ".join(unknown))
    return list(dict.fromkeys(raw))


def parse_gpus(value: str) -> list[str]:
    gpus = [part.strip() for part in value.split(",") if part.strip()]
    if not gpus or any(not part.isdigit() for part in gpus):
        raise ConfigurationError("--gpus must be a comma-separated list of nonnegative integers.")
    if len(gpus) != len(set(gpus)):
        raise ConfigurationError("--gpus contains duplicate device IDs.")
    return gpus


def resolve_preset(
    registry: Mapping[str, Any], args: argparse.Namespace
) -> tuple[dict[str, Any], dict[str, Any]]:
    family = registry["families"][args.family]
    if args.preset not in family["presets"]:
        available = ", ".join(family["presets"])
        raise ConfigurationError(
            f"Unsupported preset {args.preset!r} for {args.family}; choose {available}."
        )
    preset = dict(family["presets"][args.preset])
    custom_values = (args.stage1_topk, args.stage1_fps, args.stage2_keep)
    if args.preset != "custom" and any(value is not None for value in custom_values):
        raise ConfigurationError("Custom budget flags require --family videollava --preset custom.")
    if args.preset == "custom":
        if args.family != "videollava" or any(value is None for value in custom_values):
            raise ConfigurationError(
                "Video-LLaVA custom requires --stage1-topk, --stage1-fps, and --stage2-keep."
            )
        if min(custom_values) < 0 or args.phase_layer <= 0:
            raise ConfigurationError("Custom routing budgets must be nonnegative and phase layer positive.")
        if args.stage1_topk + args.stage1_fps > family["patches_per_frame"]:
            raise ConfigurationError("Custom per-frame Stage-1 budget exceeds patches_per_frame.")
        stage1 = family["frames"] * (args.stage1_topk + args.stage1_fps)
        if args.stage2_keep >= stage1:
            raise ConfigurationError("Custom Stage-2 keep must be smaller than Stage-1 tokens.")
        if args.phase_layer >= family["llm_layers"]:
            raise ConfigurationError("Custom phase layer must be below the decoder layer count.")
        average = (
            args.phase_layer * stage1
            + (family["llm_layers"] - args.phase_layer) * args.stage2_keep
        ) / family["llm_layers"]
        preset.update(
            route_topk=args.stage1_topk,
            route_fps=args.stage1_fps,
            stage1_visual_tokens=stage1,
            phase_transition_layer=args.phase_layer,
            phase_transition_keep=args.stage2_keep,
            average_visual_tokens=average,
        )
    return family, preset


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_key(row: Mapping[str, Any], fields: Sequence[str]) -> tuple[str, ...]:
    values: list[str] = []
    for field in fields:
        value = row.get(field)
        if value is None or isinstance(value, (dict, list)) or not str(value).strip():
            raise ValueError(f"missing or invalid {field!r}")
        values.append(str(value).strip())
    return tuple(values)


def inspect_record_file(path: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    container = str(contract["container"])
    fields = tuple(contract.get("key_fields", ()))
    count = 0
    seen: set[tuple[str, ...]] = set()
    duplicate_count = 0
    errors: list[str] = []

    def accept(row: object, location: str) -> None:
        nonlocal count, duplicate_count
        count += 1
        if not fields:
            return
        if not isinstance(row, dict):
            errors.append(f"{location}: expected an object")
            return
        try:
            key = record_key(row, fields)
        except ValueError as exc:
            errors.append(f"{location}: {exc}")
            return
        if key in seen:
            duplicate_count += 1
        else:
            seen.add(key)

    try:
        if container == "jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        accept(json.loads(line), f"line {line_number}")
                    except json.JSONDecodeError as exc:
                        errors.append(f"line {line_number}: malformed JSON ({exc.msg})")
        elif container == "tsv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                if reader.fieldnames is None:
                    errors.append("TSV has no header")
                elif any(field not in reader.fieldnames for field in fields):
                    errors.append(f"TSV header lacks key fields {fields}")
                for row_number, row in enumerate(reader, start=2):
                    accept(row, f"row {row_number}")
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if container == "list":
                records = payload
            elif container == "data":
                records = payload.get("data") if isinstance(payload, dict) else None
            elif container == "list_key":
                records = payload.get(contract["list_key"]) if isinstance(payload, dict) else None
            elif container == "dict_keys":
                if not isinstance(payload, dict):
                    records = None
                else:
                    records = list(payload)
            else:
                records = None
                errors.append(f"unsupported record container {container!r}")
            if not isinstance(records, list):
                errors.append(f"expected {container} records")
            elif container in {"list_key", "dict_keys"}:
                count = len(records)
                identifiers = [str(value).strip() for value in records]
                invalid = sum(not value for value in identifiers)
                duplicate_count = len(identifiers) - len(set(identifiers))
                if invalid:
                    errors.append(f"{invalid} empty identifiers")
            else:
                for index, row in enumerate(records, start=1):
                    accept(row, f"record {index}")
    except (OSError, json.JSONDecodeError, UnicodeError, csv.Error) as exc:
        errors.append(f"cannot parse records: {exc}")

    expected = contract.get("expected_records")
    minimum = contract.get("minimum_records")
    if expected is not None and count != int(expected):
        errors.append(f"expected {expected} records, found {count}")
    if minimum is not None and count < int(minimum):
        errors.append(f"expected at least {minimum} records, found {count}")
    if duplicate_count:
        errors.append(f"found {duplicate_count} duplicate identifiers")
    return {
        "records": count,
        "unique_ids": len(seen) if fields else None,
        "duplicate_ids": duplicate_count,
        "expected_records": expected,
        "valid": not errors,
        "errors": errors[:20],
    }


def dataset_paths(data_root: Path, family_name: str, dataset: str) -> list[Path]:
    if family_name != "videollava":
        return [data_root / relative for relative in IMAGE_DATASETS[dataset]["required"]]
    definition = VIDEO_DATASETS[dataset]
    roots = [data_root / candidate for candidate in definition["candidates"]]
    dataset_root = next((path for path in roots if path.is_dir()), roots[0])
    return [
        dataset_root / definition["video_subdir"],
        dataset_root / "test_q.json",
        dataset_root / "test_a.json",
    ]


def iter_question_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, dict):
                        yield value
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        for value in payload:
            if isinstance(value, dict):
                yield value


def decode_image(path: Path) -> list[int]:
    from PIL import Image

    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return [int(image.width), int(image.height)]


def inspect_media_directory(path: Path) -> dict[str, Any]:
    files = [
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS
    ]
    result: dict[str, Any] = {"media_files": len(files), "bounded_decode": None}
    if not files:
        result["errors"] = ["no supported image files found"]
    else:
        try:
            result["bounded_decode"] = {
                "path": str(files[0]),
                "width_height": decode_image(files[0]),
            }
        except (ImportError, OSError, ValueError) as exc:
            result["errors"] = [f"bounded image decode failed: {exc}"]
    return result


def inspect_referenced_media(question_path: Path, image_root: Path) -> dict[str, Any]:
    references: set[str] = set()
    errors: list[str] = []
    try:
        for row in iter_question_rows(question_path):
            image = row.get("image")
            if image is not None and str(image).strip():
                references.add(str(image).strip().replace("\\", "/"))
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        errors.append(f"cannot inspect media references: {exc}")
    resolved = [image_root / reference for reference in sorted(references)]
    missing = [path for path in resolved if not path.is_file()]
    report: dict[str, Any] = {
        "referenced_media": len(references),
        "missing_media": len(missing),
        "missing_examples": [str(path) for path in missing[:5]],
        "bounded_decode": None,
    }
    if missing:
        errors.append(f"{len(missing)} referenced images are missing")
    existing = next((path for path in resolved if path.is_file()), None)
    if existing is not None:
        try:
            report["bounded_decode"] = {
                "path": str(existing),
                "width_height": decode_image(existing),
            }
        except (ImportError, OSError, ValueError) as exc:
            errors.append(f"bounded image decode failed: {exc}")
    elif references:
        errors.append("no referenced image can be decoded")
    report["errors"] = errors
    return report


def inspect_embedded_image(path: Path, field: str) -> dict[str, Any]:
    try:
        from PIL import Image

        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            row = next(csv.DictReader(handle, delimiter="\t"))
        encoded = str(row.get(field, "")).strip()
        with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
            image.verify()
        with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
            shape = [int(image.width), int(image.height)]
        return {"bounded_decode": {"embedded_field": field, "width_height": shape}, "errors": []}
    except (ImportError, OSError, ValueError, TypeError, StopIteration, csv.Error) as exc:
        return {"bounded_decode": None, "errors": [f"embedded image decode failed: {exc}"]}


def image_dataset_deep_checks(data_root: Path, dataset: str) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    inputs = IMAGE_DATASETS[dataset]["required"]
    question_path = data_root / inputs[0]
    media_roots = {
        "gqa": data_root / "gqa/images",
        "textvqa": data_root / "textvqa/train_images",
        "mme": data_root / "mme/MME_Benchmark",
        "sqa": data_root / "scienceqa/images/test",
        "pope": data_root / "coco/val2014",
        "vqav2": data_root / "vqav2/test2015",
    }
    if dataset in media_roots and question_path.is_file() and media_roots[dataset].is_dir():
        checks["media"] = inspect_referenced_media(question_path, media_roots[dataset])
        checks["media_inventory"] = inspect_media_directory(media_roots[dataset])
    if dataset in {"mmbench", "mmbench_cn"} and question_path.is_file():
        checks["media"] = inspect_embedded_image(question_path, "image")
    if dataset == "mme" and (data_root / "mme/MME_Benchmark").is_dir():
        try:
            from evaluation.mme.calculation import EVAL_GROUPS
            from evaluation.mme.convert_answer_to_mme import load_ground_truth

            ground_truth = load_ground_truth(data_root / "mme/MME_Benchmark")
            categories = sorted({key[0] for key in ground_truth})
            expected_categories = sorted(category for values in EVAL_GROUPS.values() for category in values)
            checks["ground_truth"] = {
                "records": len(ground_truth),
                "categories": categories,
                "valid": len(ground_truth) == 2374 and categories == expected_categories,
                "errors": [] if len(ground_truth) == 2374 and categories == expected_categories else [
                    "MME ground-truth count or category set differs from the formal release"
                ],
            }
        except (OSError, ValueError) as exc:
            checks["ground_truth"] = {"valid": False, "errors": [str(exc)]}
    if dataset == "sqa":
        try:
            questions = json.loads(question_path.read_text(encoding="utf-8"))
            splits = json.loads((data_root / "scienceqa/pid_splits.json").read_text(encoding="utf-8"))
            problems = json.loads((data_root / "scienceqa/problems.json").read_text(encoding="utf-8"))
            question_ids = {str(row["id"]) for row in questions}
            test_ids = {str(value) for value in splits["test"]}
            problem_ids = {str(value) for value in problems}
            image_count = sum(bool(problems[item].get("image")) for item in test_ids)
            valid = question_ids == test_ids and test_ids <= problem_ids and image_count == 2017
            checks["scienceqa_contract"] = {
                "question_ids": len(question_ids),
                "test_ids": len(test_ids),
                "image_questions": image_count,
                "valid": valid,
                "errors": [] if valid else ["ScienceQA test IDs or image count differ from the formal split"],
            }
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            checks["scienceqa_contract"] = {"valid": False, "errors": [str(exc)]}
    return checks


def inspect_video_dataset(data_root: Path, dataset: str, deep: bool) -> dict[str, Any]:
    try:
        from videollava.eval.video.preflight import validate_dataset

        result = validate_dataset(
            data_root,
            dataset,
            require_canonical_count=True,
            decode_sample=deep,
        )
        result.update(valid=True, errors=[])
        return result
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        return {"valid": False, "errors": [str(exc)]}


def inspect_datasets(
    data_root: Path,
    family_name: str,
    datasets: Sequence[str],
    *,
    deep: bool = False,
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    definitions = VIDEO_DATASETS if family_name == "videollava" else IMAGE_DATASETS
    for dataset in datasets:
        items: list[dict[str, Any]] = []
        for path in dataset_paths(data_root, family_name, dataset):
            item: dict[str, Any] = {
                "path": str(path),
                "exists": path.exists(),
                "kind": "directory" if path.is_dir() else ("file" if path.is_file() else "missing"),
            }
            if path.is_file():
                item["size_bytes"] = path.stat().st_size
                item["sha256"] = sha256_file(path)
                try:
                    relative = path.relative_to(data_root).as_posix()
                except ValueError:
                    relative = ""
                contract = FILE_CONTRACTS.get(relative)
                if contract is not None:
                    item["record_contract"] = inspect_record_file(path, contract)
            items.append(item)
        entry = {
            "dataset": dataset,
            "split": definitions[dataset]["split"],
            "evaluator": definitions[dataset]["evaluator"],
            "inputs": items,
        }
        if family_name == "videollava" and all(path.exists() for path in dataset_paths(data_root, family_name, dataset)):
            entry["contract"] = inspect_video_dataset(data_root, dataset, deep)
        elif deep and family_name != "videollava":
            entry["checks"] = image_dataset_deep_checks(data_root, dataset)
        manifest.append(entry)
    return manifest


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def normalized_version(value: str | None) -> str | None:
    return value.split("+", 1)[0] if value else value


def transformers_install_commit() -> str | None:
    try:
        distribution = importlib.metadata.distribution("transformers")
    except importlib.metadata.PackageNotFoundError:
        return None
    for item in distribution.files or ():
        if str(item).endswith("direct_url.json"):
            try:
                payload = json.loads(distribution.locate_file(item).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
            vcs = payload.get("vcs_info")
            if isinstance(vcs, dict):
                commit = vcs.get("commit_id")
                return str(commit) if commit else None
    return None


def environment_contract_report(environment_name: str) -> dict[str, Any]:
    expected = ENVIRONMENT_CONTRACTS[environment_name]
    installed = package_versions(("torch", "torchvision", "transformers", "numpy"))
    installed["python"] = platform.python_version()
    checks: dict[str, dict[str, Any]] = {}
    for name in ("python", "torch", "torchvision", "transformers", "numpy"):
        actual = normalized_version(installed.get(name))
        wanted = str(expected[name])
        checks[name] = {"expected": wanted, "actual": actual, "valid": actual == wanted}
    if "transformers_commit" in expected:
        commit = transformers_install_commit()
        checks["transformers_commit"] = {
            "expected": expected["transformers_commit"],
            "actual": commit,
            "valid": commit == expected["transformers_commit"],
        }
    errors = [
        f"{name}: expected {item['expected']}, found {item['actual']}"
        for name, item in checks.items()
        if not item["valid"]
    ]
    return {"environment": environment_name, "checks": checks, "valid": not errors, "errors": errors}


def checkpoint_index_report(model_path: Path) -> dict[str, Any]:
    index_names = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
    index_path = next((model_path / name for name in index_names if (model_path / name).is_file()), None)
    errors: list[str] = []
    if index_path is not None:
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = payload.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("weight_map is absent or empty")
            shard_names = sorted({str(value) for value in weight_map.values()})
            missing_shards = [name for name in shard_names if not (model_path / name).is_file()]
            symlink_shards = [name for name in shard_names if (model_path / name).is_symlink()]
            if missing_shards:
                errors.append(f"missing {len(missing_shards)} indexed weight shards")
            if symlink_shards:
                errors.append(f"found {len(symlink_shards)} symlinked weight shards")
            return {
                "layout": index_path.name,
                "index_sha256": sha256_file(index_path),
                "weight_keys": len(weight_map),
                "shards": len(shard_names),
                "missing_shards": missing_shards[:10],
                "valid": not errors,
                "errors": errors,
            }
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return {"layout": index_path.name, "valid": False, "errors": [str(exc)]}
    monolithic = next(
        (
            model_path / name
            for name in ("model.safetensors", "pytorch_model.bin")
            if (model_path / name).is_file()
        ),
        None,
    )
    if monolithic is None:
        return {"layout": None, "valid": False, "errors": ["no complete model weights or weight index found"]}
    return {
        "layout": monolithic.name,
        "size_bytes": monolithic.stat().st_size,
        "valid": not monolithic.is_symlink(),
        "errors": ["monolithic weight file is a symlink"] if monolithic.is_symlink() else [],
    }


def validate_llava_checkpoint(model_path: Path, family_name: str) -> dict[str, Any]:
    errors: list[str] = []
    config_path = model_path / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"valid": False, "errors": [f"cannot read config.json: {exc}"]}
    expected = {
        "model_type": "llava",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
    }
    mismatches = {
        name: {"expected": value, "actual": config.get(name)}
        for name, value in expected.items()
        if config.get(name) != value
    }
    architectures = config.get("architectures")
    if architectures != ["LlavaLlamaForCausalLM"]:
        mismatches["architectures"] = {
            "expected": ["LlavaLlamaForCausalLM"],
            "actual": architectures,
        }
    vision_tower = config.get("mm_vision_tower") or config.get("vision_tower")
    if not isinstance(vision_tower, str) or not vision_tower:
        errors.append("config has no mm_vision_tower")
    else:
        try:
            from transformers import CLIPVisionConfig

            CLIPVisionConfig.from_pretrained(vision_tower, local_files_only=True)
        except (ImportError, OSError, ValueError) as exc:
            errors.append(
                "vision tower is unavailable from the local Hugging Face cache; "
                "obtain the tower declared by config.json before preflight: "
                f"{exc}"
            )
    if mismatches:
        errors.append(f"checkpoint config mismatch: {mismatches}")
    tokenizer = [name for name in ("tokenizer.model", "tokenizer.json") if (model_path / name).is_file()]
    if not tokenizer or not (model_path / "tokenizer_config.json").is_file():
        errors.append("checkpoint tokenizer files are incomplete")
    weights = checkpoint_index_report(model_path)
    if not weights["valid"]:
        errors.extend(weights["errors"])
    next_contract = None
    if family_name == "llava_next":
        next_contract = {
            "image_resolution": "672x672",
            "crops": 5,
            "stage1_candidates_per_crop": 256,
            "valid": True,
        }
    return {
        "family": family_name,
        "config_sha256": sha256_file(config_path),
        "model_type": config.get("model_type"),
        "architectures": architectures,
        "vision_tower": vision_tower,
        "tokenizer_files": tokenizer,
        "weights": weights,
        "llava_next_input_contract": next_contract,
        "valid": not errors,
        "errors": errors,
    }


def validate_qwen_checkpoint(model_path: Path) -> dict[str, Any]:
    errors: list[str] = []
    config_path = model_path / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"valid": False, "errors": [f"cannot read config.json: {exc}"]}
    vision = config.get("vision_config") if isinstance(config.get("vision_config"), dict) else {}
    expected = {
        "model_type": (config.get("model_type"), "qwen2_5_vl"),
        "num_hidden_layers": (config.get("num_hidden_layers"), 28),
        "hidden_size": (config.get("hidden_size"), 3584),
        "num_attention_heads": (config.get("num_attention_heads"), 28),
        "num_key_value_heads": (config.get("num_key_value_heads"), 4),
        "vision.patch_size": (vision.get("patch_size"), 14),
        "vision.spatial_merge_size": (vision.get("spatial_merge_size"), 2),
    }
    mismatches = {
        name: {"expected": wanted, "actual": actual}
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    }
    architectures = config.get("architectures")
    if architectures != ["Qwen2_5_VLForConditionalGeneration"]:
        mismatches["architectures"] = {
            "expected": ["Qwen2_5_VLForConditionalGeneration"],
            "actual": architectures,
        }
    if mismatches:
        errors.append(f"checkpoint config mismatch: {mismatches}")
    required_processor = ("tokenizer_config.json", "preprocessor_config.json")
    missing_processor = [name for name in required_processor if not (model_path / name).is_file()]
    if missing_processor:
        errors.append("missing processor/tokenizer files: " + ", ".join(missing_processor))
    weights = checkpoint_index_report(model_path)
    if not weights["valid"]:
        errors.extend(weights["errors"])
    grid_h = 896 // 14
    grid_w = 1120 // 14
    merged = grid_h * grid_w // (2 * 2)
    grid_contract = {
        "resized_width_height": [1120, 896],
        "image_grid_thw": [1, grid_h, grid_w],
        "merged_visual_tokens": merged,
        "valid": grid_h == 64 and grid_w == 80 and merged == 1280,
    }
    if not grid_contract["valid"]:
        errors.append("Qwen image-grid contract does not resolve to 1280 merged tokens")
    return {
        "family": "qwen25vl",
        "config_sha256": sha256_file(config_path),
        "model_type": config.get("model_type"),
        "architectures": architectures,
        "weights": weights,
        "image_grid_contract": grid_contract,
        "valid": not errors,
        "errors": errors,
    }


def checkpoint_contract_report(model_path: Path, family_name: str) -> dict[str, Any]:
    if not model_path.is_dir():
        return {"valid": False, "errors": ["model path is not a checkpoint directory"]}
    if family_name in {"llava15", "llava_next"}:
        return validate_llava_checkpoint(model_path, family_name)
    if family_name == "qwen25vl":
        return validate_qwen_checkpoint(model_path)
    try:
        from videollava.eval.video.preflight import validate_checkpoint

        result = validate_checkpoint(model_path)
        result.update(family="videollava", valid=True, errors=[])
        return result
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        return {"family": "videollava", "valid": False, "errors": [str(exc)]}


def backend_contract_report(config: Mapping[str, Any]) -> dict[str, Any]:
    if config["family"] != "qwen25vl":
        errors: list[str] = []
        details: dict[str, Any] = {}
        if config["attention_backend"] == "flash_attention_2":
            available = importlib.util.find_spec("flash_attn") is not None
            try:
                version = importlib.metadata.version("flash-attn")
            except importlib.metadata.PackageNotFoundError:
                version = None
            details = {"flash_attn_available": available, "flash_attn_version": version}
            if not available or normalized_version(version) != "2.3.3":
                errors.append(
                    f"FlashAttention2 backend requires flash-attn 2.3.3; found {version!r}"
                )
        return {
            "declared": config["attention_backend"],
            "resolved": config["attention_backend"],
            **details,
            "valid": not errors,
            "errors": errors,
        }
    previous_method = os.environ.get("METHOD")
    previous_flash = os.environ.get("USE_FLASH_ATTN")
    try:
        os.environ["METHOD"] = str(config["method"])
        os.environ["USE_FLASH_ATTN"] = "0"
        from qwen.eval.attention_backend import resolve_qwen_attention_backend

        resolved = resolve_qwen_attention_backend()
    except (ImportError, ValueError) as exc:
        return {"declared": "sdpa", "resolved": None, "valid": False, "errors": [str(exc)]}
    finally:
        if previous_method is None:
            os.environ.pop("METHOD", None)
        else:
            os.environ["METHOD"] = previous_method
        if previous_flash is None:
            os.environ.pop("USE_FLASH_ATTN", None)
        else:
            os.environ["USE_FLASH_ATTN"] = previous_flash
    valid = resolved == "sdpa" if config["method"] == "trevs" else resolved is None
    return {
        "declared": "sdpa" if config["method"] == "trevs" else "native_hf",
        "resolved": resolved or "native_hf",
        "four_dimensional_mask": config["method"] == "trevs",
        "heterogeneous_layer_cache": config["method"] == "trevs",
        "valid": valid,
        "errors": [] if valid else ["Qwen attention backend did not resolve to the required path"],
    }


def gpu_info() -> dict[str, Any]:
    command = shutil.which("nvidia-smi")
    if command is None:
        return {"nvidia_smi": None, "gpus": []}
    result = subprocess.run(
        [command, "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()] if result.returncode == 0 else []
    error = result.stderr.strip() or (result.stdout.strip() if result.returncode else None)
    return {
        "nvidia_smi": result.returncode == 0,
        "gpus": lines,
        "indices": [line.split(",", 1)[0].strip() for line in lines],
        "error": error,
    }


def torch_gpu_report(requested_ids: Sequence[str]) -> dict[str, Any]:
    """Probe each requested physical GPU in its own visibility namespace.

    The launcher treats ``--gpus`` as physical NVIDIA IDs.  Running the probe
    in a child process with one requested device visible avoids conflating those
    IDs with logical indices inherited from a scheduler's CUDA_VISIBLE_DEVICES.
    """
    probe_code = """
import json
try:
    import torch
    available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count()) if available else 0
    if not available or count < 1:
        raise RuntimeError(f"torch.cuda.is_available={available}, device_count={count}")
    properties = torch.cuda.get_device_properties(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    tensor = torch.empty(1, device="cuda:0")
    del tensor
    print(json.dumps({
        "valid": True,
        "torch_cuda_version": torch.version.cuda,
        "visible_device_count": count,
        "name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": total_bytes,
        "free_memory_bytes": free_bytes,
        "allocation_test": True,
    }, sort_keys=True))
except Exception as exc:
    print(json.dumps({"valid": False, "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
"""
    errors: list[str] = []
    devices: list[dict[str, Any]] = []
    parent_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    for value in requested_ids:
        index = int(value)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = value
        try:
            result = subprocess.run(
                [sys.executable, "-c", probe_code],
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            payload = json.loads(result.stdout.strip())
            if result.returncode or not isinstance(payload, dict) or not payload.get("valid"):
                detail = payload.get("error") if isinstance(payload, dict) else result.stdout.strip()
                detail = detail or result.stderr.strip() or f"subprocess exit {result.returncode}"
                errors.append(f"CUDA device {index} allocation/properties failed: {detail}")
                continue
            devices.append({"index": index, **payload})
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"CUDA device {index} probe failed: {exc}")
    return {
        "probe_mode": "per_physical_id_subprocess",
        "parent_cuda_visible_devices": parent_visibility,
        "devices": devices,
        "valid": not errors,
        "errors": errors,
    }


def environment_snapshot(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    snapshot = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(
            ("torch", "torchvision", "transformers", "numpy", "accelerate", "decord", "pandas")
        ),
        "gpu": gpu_info(),
    }
    if config is not None:
        snapshot["attention_backend"] = backend_contract_report(config)
        snapshot["environment_contract"] = environment_contract_report(str(config["environment"]))
    return snapshot


def build_config(
    args: argparse.Namespace,
    family: Mapping[str, Any],
    preset: Mapping[str, Any],
    datasets: Sequence[str],
    gpus: Sequence[str],
) -> dict[str, Any]:
    model_path = normalized_path(args.model_path)
    data_root = normalized_path(args.data_root)
    output_root = normalized_path(args.output_root)
    validate_output_root(output_root, data_root, model_path)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not run_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in run_id):
        raise ConfigurationError("--run-id may contain only letters, digits, dot, underscore, and hyphen.")
    run_dir = output_root / args.family / args.preset / run_id
    generation = dict(family["generation"])
    if args.max_new_tokens is not None:
        if args.max_new_tokens <= 0:
            raise ConfigurationError("--max-new-tokens must be positive.")
        generation["max_new_tokens"] = args.max_new_tokens
    attention_backend = family["attention_backend"]
    if args.family == "qwen25vl" and preset["method"] == "dense":
        attention_backend = "native_hf"
    return {
        "schema_version": 1,
        "family": args.family,
        "family_display_name": family["display_name"],
        "preset": args.preset,
        "method": preset["method"],
        "run_id": run_id,
        "seed": args.seed,
        "model_path": str(model_path),
        "data_root": str(data_root),
        "output_root": str(output_root),
        "run_dir": str(run_dir),
        "datasets": list(datasets),
        "gpus": list(gpus),
        "environment": family["environment"],
        "attention_backend": attention_backend,
        "image_resolution": family["image_resolution"],
        "frames": family["frames"],
        "crops": family["crops"],
        "llm_layers": family["llm_layers"],
        "budget": dict(preset),
        "trevs": dict(family["trevs"]),
        "generation": generation,
        "max_samples": args.max_samples,
        "run_judge": bool(args.run_judge),
    }


def command_plan(config: Mapping[str, Any]) -> list[list[str]]:
    if config["family"] == "videollava":
        base = REPO_ROOT / "scripts" / "videollava"
        commands = [["bash", str(base / f"run_qa_{dataset}.sh")] for dataset in config["datasets"]]
    else:
        runner = REPO_ROOT / "scripts" / "common" / "run_image_benchmark.sh"
        commands = [
            ["bash", str(runner), str(config["family"]), dataset]
            for dataset in config["datasets"]
        ]
    if config["family"] == "videollava":
        if config["run_judge"]:
            commands.extend(
                ["bash", str(base / f"eval_qa_{dataset}.sh")] for dataset in config["datasets"]
            )
    return commands


def portable_command_plan(config: Mapping[str, Any]) -> list[list[str]]:
    portable: list[list[str]] = []
    for command in command_plan(config):
        row: list[str] = []
        for value in command:
            path = Path(value)
            try:
                row.append(path.relative_to(REPO_ROOT).as_posix() if path.is_absolute() else value)
            except ValueError:
                row.append(value)
        portable.append(row)
    return portable


def source_manifest_sha256() -> str | None:
    manifest = REPO_ROOT / "MANIFEST.sha256"
    return sha256_file(manifest) if manifest.is_file() and not manifest.is_symlink() else None


def runtime_environment(config: Mapping[str, Any]) -> dict[str, str]:
    budget = config["budget"]
    trevs = config["trevs"]
    environment = os.environ.copy()
    environment.update(
        {
            "REPO_ROOT": str(REPO_ROOT),
            "PYTHONPATH": str(REPO_ROOT) + (":" + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""),
            "PYTHONDONTWRITEBYTECODE": "1",
            "MODEL_PATH": str(config["model_path"]),
            "DATA_ROOT": str(config["data_root"]),
            "OUTPUT_ROOT": str(config["output_root"]),
            "RESULT_ROOT": str(config["run_dir"]),
            "RUN_DIR": str(config["run_dir"]),
            "RUN_ID": str(config["run_id"]),
            "EXPERIMENT": str(config["run_id"]),
            "TReVS_FAMILY": str(config["family"]),
            "TREVS_PRESET": str(config["preset"]),
            "METHOD": str(config["method"]),
            "TARGET_DATASET": ",".join(config["datasets"]),
            "CUDA_VISIBLE_DEVICES": ",".join(config["gpus"]),
            "RANDOM_SEED": str(config["seed"]),
            "MAX_SAMPLES": str(config["max_samples"]),
            "MAX_NEW_TOKENS": str(config["generation"]["max_new_tokens"]),
            "RUN_JUDGE": "1" if config["run_judge"] else "0",
            "USE_FLASH_ATTN": "1" if config["attention_backend"] == "flash_attention_2" else "0",
            "ATTN_IMPLEMENTATION": str(config["attention_backend"]),
            "TREVS_TEXT_SCORE_MODE": str(trevs["text_score_mode"]),
            "TREVS_VISUAL_TEMPERATURE": str(trevs["visual_temperature"]),
            "TREVS_TEXT_TEMPERATURE": str(trevs["text_temperature"]),
            "TREVS_SEMANTIC_LAYER": str(trevs["semantic_layer"]),
            "TREVS_STAGE1_SCORING": str(trevs["stage1_scoring"]),
            "TREVS_PHASE_SCORING": str(trevs["phase_scoring"]),
            "TREVS_USE_SINK_TOKEN": "1" if trevs["use_sink_token"] else "0",
            "DOUBLE_TRACK_USE_CONSISTENCY_REWARD": "1" if trevs["use_consistency_reward"] else "0",
            "TREVS_USE_CONSISTENCY_REWARD": "1" if trevs["use_consistency_reward"] else "0",
            "LLM_NUM_LAYERS": str(config["llm_layers"]),
        }
    )
    if budget["route_topk"] is not None:
        environment["TREVS_ROUTE_TOPK"] = str(budget["route_topk"])
        environment["TREVS_ROUTE_FPS"] = str(budget["route_fps"])
    if budget["phase_transition_layer"] is not None:
        environment["PHASE_TRANSITION_LAYER"] = str(budget["phase_transition_layer"])
    environment["PHASE_TRANSITION_N_KEEP"] = str(budget["phase_transition_keep"])
    if config["family"] == "llava_next":
        environment["LLAVA_IMAGE_GRID_PINPOINTS"] = "[[672, 672]]"
    if config["family"] == "qwen25vl":
        environment.update(
            {
                "QWEN_ATTN_IMPLEMENTATION": "sdpa",
                "QWEN_IMAGE_RESIZED_WIDTH": "1120",
                "QWEN_IMAGE_RESIZED_HEIGHT": "896",
                "QWEN_USE_CACHE": "1",
            }
        )
    if config["family"] == "videollava":
        environment["VIDEOLLAVA_TREVS_PRESET"] = str(config["preset"])
    return environment


def preflight_report(config: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    model_path = Path(config["model_path"])
    data_root = Path(config["data_root"])
    manifest = inspect_datasets(data_root, config["family"], config["datasets"], deep=True)
    missing_inputs = [
        item["path"]
        for dataset in manifest
        for item in dataset["inputs"]
        if not item["exists"]
    ]
    expected_environment = config["environment"]
    required_modules = ["torch", "transformers", "numpy"]
    required_modules.append("PIL")
    if expected_environment == "qwen":
        required_modules.extend(["qwen_vl_utils", "pandas"])
    elif config["family"] == "videollava":
        required_modules.extend(["decord", "safetensors"])
    elif config["attention_backend"] == "flash_attention_2":
        required_modules.append("flash_attn")
    modules = {name: importlib.util.find_spec(name) is not None for name in required_modules}
    environment_contract = environment_contract_report(str(expected_environment))
    checkpoint = checkpoint_contract_report(model_path, str(config["family"]))
    backend = backend_contract_report(config)
    gpu = gpu_info()
    torch_gpu = torch_gpu_report(config["gpus"])
    report = {
        "status": "ok",
        "model_path": {"path": str(model_path), "exists": model_path.is_dir()},
        "checkpoint": checkpoint,
        "data_root": {"path": str(data_root), "exists": data_root.is_dir()},
        "datasets": manifest,
        "dependencies": {"modules": modules, "locked_environment": environment_contract},
        "attention_backend": backend,
        "gpu": {"nvidia_smi": gpu, "torch": torch_gpu},
        "requested_gpu_ids": config["gpus"],
        "commands": command_plan(config),
    }
    failures: list[str] = []
    if not model_path.is_dir():
        failures.append("model path is not a checkpoint directory")
    elif not checkpoint.get("valid"):
        failures.append("checkpoint contract failed: " + "; ".join(checkpoint.get("errors", [])))
    if not data_root.is_dir():
        failures.append("data root does not exist")
    if missing_inputs:
        failures.append(f"{len(missing_inputs)} required dataset paths are missing")
    missing_modules = [name for name, available in modules.items() if not available]
    if missing_modules:
        failures.append("missing Python modules: " + ", ".join(missing_modules))
    if not environment_contract["valid"]:
        failures.append("locked environment mismatch: " + "; ".join(environment_contract["errors"]))
    if not gpu.get("nvidia_smi"):
        failures.append("nvidia-smi did not report an available GPU")
    else:
        missing_gpu_ids = sorted(set(config["gpus"]).difference(gpu.get("indices", [])))
        if missing_gpu_ids:
            failures.append("requested physical GPU IDs are unavailable: " + ", ".join(missing_gpu_ids))
    if not torch_gpu["valid"]:
        failures.append("torch CUDA validation failed: " + "; ".join(torch_gpu["errors"]))
    if not backend["valid"]:
        failures.append("attention backend contract failed: " + "; ".join(backend["errors"]))
    for dataset in manifest:
        for item in dataset["inputs"]:
            contract = item.get("record_contract")
            if contract is not None and not contract["valid"]:
                failures.append(
                    f"{dataset['dataset']} record contract failed for {item['path']}: "
                    + "; ".join(contract["errors"])
                )
        dataset_contract = dataset.get("contract")
        if dataset_contract is not None and not dataset_contract.get("valid", False):
            failures.append(
                f"{dataset['dataset']} dataset contract failed: "
                + "; ".join(dataset_contract.get("errors", []))
            )
        for check_name, check in dataset.get("checks", {}).items():
            if check.get("errors"):
                failures.append(
                    f"{dataset['dataset']} {check_name} failed: " + "; ".join(check["errors"])
                )
    if failures:
        report["status"] = "failed"
        report["failures"] = failures
    return report, not failures


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def inspect_video_predictions(path: Path) -> int:
    seen: set[str] = set()
    count = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or not {"id", "question", "answer", "pred"} <= set(row):
                    raise RuntimeError(f"Malformed VideoQA prediction at {path}:{line_number}")
                identifier = str(row["id"]).strip()
                if not identifier or identifier in seen:
                    raise RuntimeError(f"Empty or duplicate VideoQA prediction ID: {identifier!r}")
                seen.add(identifier)
                count += 1
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Malformed VideoQA prediction JSON in {path}: {exc}") from exc
    if count == 0:
        raise RuntimeError(f"VideoQA prediction file is empty: {path}")
    return count


def write_video_prediction_status(config: Mapping[str, Any], dataset: str) -> None:
    run_dir = Path(config["run_dir"])
    predictions = run_dir / "predictions" / dataset / "predictions.jsonl"
    count = inspect_video_predictions(predictions)
    write_json(
        run_dir / "metrics" / dataset / "evaluator.json",
        {
            "schema_version": 1,
            "dataset": dataset,
            "split": VIDEO_DATASETS[dataset]["split"],
            "status": "prediction_only",
            "evaluator": "videoqa_gpt_judge_not_run",
            "run_type": "smoke" if config["max_samples"] else "full",
            "samples": count,
            "comparable_to_paper": False,
            "diagnostics": [
                "Predictions were generated and ID-validated, but no text judge score is available."
            ],
        },
    )


def write_video_judge_metric(config: Mapping[str, Any], dataset: str) -> None:
    run_dir = Path(config["run_dir"])
    judge_root = run_dir / "metrics" / dataset / "judge"
    summaries = sorted(judge_root.glob("*/*/summary.json"))
    if len(summaries) != 1:
        raise RuntimeError(
            f"Expected one VideoQA judge summary for {dataset}; found {len(summaries)}."
        )
    try:
        summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read VideoQA judge summary: {summaries[0]}") from exc
    required = {
        "accuracy",
        "average_score",
        "successful_count",
        "failed_count",
        "total_predictions",
        "complete",
        "judge_model",
        "prompt_version",
    }
    if not isinstance(summary, dict) or not required <= set(summary) or not summary["complete"]:
        raise RuntimeError(f"VideoQA judge summary is incomplete or malformed: {summaries[0]}")
    if summary["accuracy"] is None or summary["average_score"] is None:
        raise RuntimeError(f"VideoQA judge summary contains no score: {summaries[0]}")
    run_type = "smoke" if config["max_samples"] else "full"
    write_json(
        run_dir / "metrics" / dataset / "evaluator.json",
        {
            "schema_version": 1,
            "dataset": dataset,
            "split": VIDEO_DATASETS[dataset]["split"],
            "status": "scored",
            "evaluator": "videoqa_gpt_judge",
            "run_type": run_type,
            "samples": int(summary["total_predictions"]),
            "accuracy_percent": 100.0 * float(summary["accuracy"]),
            "average_score": float(summary["average_score"]),
            "judge_model": str(summary["judge_model"]),
            "prompt_version": str(summary["prompt_version"]),
            "comparable_to_paper": run_type == "full",
            "source_summary": summaries[0].relative_to(run_dir).as_posix(),
            "diagnostics": []
            if run_type == "full"
            else ["A subset judge run is not comparable with a paper metric."],
        },
    )


def execute(config: Mapping[str, Any]) -> int:
    run_dir = Path(config["run_dir"])
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ConfigurationError(f"Run directory is not empty: {run_dir}")
    for subdir in ("logs", "predictions", "metrics", "submissions"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "run_config.json", config)
    write_json(run_dir / "environment.json", environment_snapshot(config))
    write_json(
        run_dir / "data_manifest.json",
        inspect_datasets(Path(config["data_root"]), config["family"], config["datasets"]),
    )
    environment = runtime_environment(config)
    for command in command_plan(config):
        dataset = Path(command[-1]).stem.replace("run_qa_", "").replace("eval_qa_", "")
        log_path = run_dir / "logs" / f"{dataset}.log"
        with log_path.open("a", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(f"Dataset command failed ({result.returncode}); see {log_path}")
        if config["family"] == "videollava":
            script_name = Path(command[-1]).name
            if script_name.startswith("run_qa_"):
                write_video_prediction_status(config, dataset)
            elif script_name.startswith("eval_qa_"):
                write_video_judge_metric(config, dataset)
    if config["family"] != "videollava":
        if config["max_samples"]:
            smoke_records = {}
            for dataset in config["datasets"]:
                source = run_dir / "metrics" / dataset / "smoke.json"
                smoke_records[dataset] = json.loads(source.read_text(encoding="utf-8"))
            write_json(
                run_dir / "metrics" / "summary.json",
                {
                    "schema_version": 1,
                    "run_type": "smoke",
                    "datasets": smoke_records,
                    "diagnostics": ["Subset smoke results are not paper-score reproductions."],
                },
            )
        else:
            sources = {
                "gqa": run_dir / "submissions" / "gqa" / "predictions.json",
                "mmbench": run_dir / "submissions" / "mmbench" / "predictions.xlsx",
                "mmbench_cn": run_dir / "submissions" / "mmbench_cn" / "predictions.xlsx",
                "sqa": run_dir / "metrics" / "sqa" / "evaluator.json",
                "textvqa": run_dir / "metrics" / "textvqa" / "evaluator.json",
                "vqav2": run_dir / "submissions" / "vqav2" / "predictions.json",
                "mme": run_dir / "metrics" / "mme" / "evaluator.json",
                "pope": run_dir / "metrics" / "pope" / "evaluator.json",
            }
            aggregate_command = [sys.executable, "-m", "evaluation.aggregate_metrics"]
            for dataset in config["datasets"]:
                aggregate_command.extend([f"--{dataset}", str(sources[dataset])])
            aggregate_command.extend(
                [
                    "--only-provided",
                    "--fail-on-error",
                    "--output",
                    str(run_dir / "metrics" / "summary.json"),
                ]
            )
            result = subprocess.run(
                aggregate_command,
                cwd=REPO_ROOT,
                env=environment,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError("Metric aggregation failed.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    registry = load_registry()
    parser = parser_for(registry)
    raw_arguments = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(raw_arguments)
    try:
        if args.max_samples < 0:
            raise ConfigurationError("--max-samples must be nonnegative.")
        family, preset = resolve_preset(registry, args)
        datasets = parse_datasets(args.datasets, family["datasets"])
        gpus = parse_gpus(args.gpus)
        config = build_config(args, family, preset, datasets, gpus)
        config["invocation"] = ["python", "scripts/reproduce.py", *raw_arguments]
        config["planned_commands"] = portable_command_plan(config)
        config["source_manifest_sha256"] = source_manifest_sha256()
        if args.dry_run:
            print(json.dumps({"mode": "dry_run", "config": config, "commands": command_plan(config)}, indent=2))
            return 0
        if args.preflight:
            report, passed = preflight_report(config)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if passed else 1
        report, passed = preflight_report(config)
        if not passed:
            print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
            return 1
        return execute(config)
    except (ConfigurationError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
