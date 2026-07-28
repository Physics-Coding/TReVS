#!/usr/bin/env python3
"""Shared image-benchmark runner used by all three image model families."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_FAMILIES = {"llava15", "llava_next", "qwen25vl"}
SUPPORTED_DATASETS = {
    "gqa",
    "textvqa",
    "mme",
    "mmbench",
    "mmbench_cn",
    "sqa",
    "pope",
    "vqav2",
}


class RunnerError(RuntimeError):
    pass


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RunnerError(f"Required environment variable is unset: {name}")
    return value


def run(command: Sequence[str], *, env: Mapping[str, str] | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
    if result.returncode:
        raise RunnerError(f"Command exited with status {result.returncode}: {' '.join(command)}")


def run_to_file(command: Sequence[str], output: Path) -> None:
    print("+ " + " ".join(command), flush=True)
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.stdout + result.stderr, encoding="utf-8")
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode:
        raise RunnerError(f"Command exited with status {result.returncode}: {' '.join(command)}")


def load_questions(path: Path) -> tuple[str, Any, int]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RunnerError(f"Malformed JSONL in {path} at line {line_number}") from exc
                if not isinstance(row, dict):
                    raise RunnerError(f"Question line {line_number} in {path} is not an object.")
                rows.append(row)
        return "jsonl", rows, len(rows)
    if suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RunnerError(f"Malformed JSON question file: {path}") from exc
        if not isinstance(payload, list):
            raise RunnerError(f"Question JSON must contain a list: {path}")
        return "json", payload, len(payload)
    if suffix == ".tsv":
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                if reader.fieldnames is None:
                    raise RunnerError(f"Question TSV has no header: {path}")
                rows = list(reader)
        except (OSError, UnicodeError, csv.Error) as exc:
            raise RunnerError(f"Malformed TSV question file: {path}") from exc
        if not rows:
            raise RunnerError(f"Question TSV is empty: {path}")
        return "tsv", {"fieldnames": reader.fieldnames, "rows": rows}, len(rows)
    raise RunnerError(f"Unsupported question-file format: {path}")


def subset_questions(source: Path, destination: Path, max_samples: int) -> tuple[Path, int]:
    kind, payload, count = load_questions(source)
    if max_samples <= 0 or max_samples >= count:
        return source, count
    destination.parent.mkdir(parents=True, exist_ok=True)
    if kind == "jsonl":
        text = "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in payload[:max_samples])
    elif kind == "json":
        text = json.dumps(payload[:max_samples], indent=2, ensure_ascii=True) + "\n"
    else:
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=payload["fieldnames"],
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(payload["rows"][:max_samples])
        return destination, max_samples
    destination.write_text(text, encoding="utf-8")
    return destination, max_samples


def question_rows(kind: str, payload: Any) -> list[Mapping[str, Any]]:
    if kind == "tsv":
        rows = payload["rows"]
    else:
        rows = payload
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise RunnerError("Question input does not contain object records.")
    return rows


def record_identity(row: Mapping[str, Any], dataset: str, *, prediction: bool) -> tuple[str, ...]:
    if prediction:
        identifier = row.get("question_id")
    else:
        identifier = row.get("question_id", row.get("id", row.get("index")))
    if identifier is None or isinstance(identifier, (dict, list)) or not str(identifier).strip():
        raise RunnerError(f"{dataset} record has an empty or invalid question identifier.")
    identity = [str(identifier).strip()]
    if dataset in {"textvqa", "mme"}:
        prompt = row.get("prompt" if prediction else "text")
        if prompt is None or isinstance(prompt, (dict, list)) or not str(prompt).strip():
            raise RunnerError(f"{dataset} record has an empty or invalid prompt.")
        identity.append(str(prompt).strip())
    return tuple(identity)


def expected_identities(source: Path, dataset: str, max_samples: int) -> list[tuple[str, ...]]:
    kind, payload, count = load_questions(source)
    rows = question_rows(kind, payload)
    if len(rows) != count:
        raise RunnerError(f"Question parser count mismatch for {source}: {len(rows)} != {count}.")
    if 0 < max_samples < count:
        rows = rows[:max_samples]
    identities = [record_identity(row, dataset, prediction=False) for row in rows]
    if len(identities) != len(set(identities)):
        raise RunnerError(f"{dataset} question input contains duplicate evaluation identities.")
    return identities


def dataset_inputs(data_root: Path, dataset: str) -> dict[str, Path]:
    mappings: dict[str, dict[str, Path]] = {
        "gqa": {
            "questions": data_root / "gqa" / "llava_gqa_testdev_balanced.jsonl",
            "images": data_root / "gqa" / "images",
            "official_questions": data_root / "gqa" / "testdev_balanced_questions.json",
        },
        "textvqa": {
            "questions": data_root / "textvqa" / "llava_textvqa_val_v051_ocr.jsonl",
            "images": data_root / "textvqa" / "train_images",
            "annotations": data_root / "textvqa" / "TextVQA_0.5.1_val.json",
        },
        "mme": {
            "questions": data_root / "mme" / "llava_mme.jsonl",
            "images": data_root / "mme" / "MME_Benchmark",
        },
        "mmbench": {
            "questions": data_root / "mmbench" / "mmbench_dev_20230712.tsv",
        },
        "mmbench_cn": {
            "questions": data_root / "mmbench" / "mmbench_dev_cn_20231003.tsv",
        },
        "sqa": {
            "questions": data_root / "scienceqa" / "llava_test_CQM-A.json",
            "images": data_root / "scienceqa" / "images" / "test",
            "base": data_root / "scienceqa",
        },
        "pope": {
            "questions": data_root / "pope" / "llava_pope_test.jsonl",
            "images": data_root / "coco" / "val2014",
            "annotations": data_root / "pope" / "coco",
        },
        "vqav2": {
            "questions": data_root / "vqav2" / "llava_vqav2_mscoco_test-dev2015.jsonl",
            "submission_questions": data_root / "vqav2" / "llava_vqav2_mscoco_test2015.jsonl",
            "images": data_root / "vqav2" / "test2015",
        },
    }
    return mappings[dataset]


def validate_inputs(inputs: Mapping[str, Path]) -> None:
    missing = [f"{name}={path}" for name, path in inputs.items() if not path.exists()]
    if missing:
        raise RunnerError("Missing dataset inputs: " + ", ".join(missing))


def inference_module(family: str, dataset: str) -> str:
    prefix = "qwen.eval" if family == "qwen25vl" else "llava.eval"
    if dataset in {"mmbench", "mmbench_cn"}:
        return f"{prefix}.model_vqa_mmbench"
    if dataset == "sqa":
        return f"{prefix}.model_vqa_science"
    return f"{prefix}.model_vqa_loader"


def command_base(
    family: str,
    dataset: str,
    model_path: Path,
    inputs: Mapping[str, Path],
    question_file: Path,
    max_new_tokens: int,
    seed: int,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        inference_module(family, dataset),
        "--model-path",
        str(model_path),
        "--question-file",
        str(question_file),
        "--temperature",
        "0",
    ]
    if "images" in inputs:
        command.extend(["--image-folder", str(inputs["images"])])
    if dataset in {"mmbench", "mmbench_cn", "sqa"}:
        command.append("--single-pred-prompt")
    if dataset == "mmbench_cn":
        command.extend(["--lang", "cn"])
    if family != "qwen25vl":
        command.extend(["--conv-mode", "vicuna_v1"])
    command.extend(["--seed", str(seed)])
    command.extend(["--max_new_tokens", str(max_new_tokens)])
    return command


def merge_chunks(
    chunks: Sequence[Path],
    destination: Path,
    dataset: str,
    expected: Sequence[tuple[str, ...]],
) -> None:
    seen: set[tuple[str, ...]] = set()
    rows: list[dict[str, Any]] = []
    for chunk in chunks:
        if not chunk.is_file():
            raise RunnerError(f"Inference did not create chunk output: {chunk}")
        with chunk.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RunnerError(f"Malformed prediction JSON in {chunk}:{line_number}") from exc
                if not isinstance(row, dict) or "question_id" not in row or "text" not in row:
                    raise RunnerError(f"Prediction {chunk}:{line_number} lacks question_id/text.")
                identity = record_identity(row, dataset, prediction=True)
                if identity in seen:
                    raise RunnerError(f"Duplicate prediction identity: {identity!r}")
                seen.add(identity)
                rows.append(row)
    expected_set = set(expected)
    missing = expected_set.difference(seen)
    extra = seen.difference(expected_set)
    if missing or extra:
        raise RunnerError(
            f"Prediction identity mismatch: missing={len(missing)}, extra={len(extra)}."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def run_inference(
    command: Sequence[str],
    answer_file: Path,
    gpus: Sequence[str],
    dataset: str,
    expected: Sequence[tuple[str, ...]],
) -> None:
    chunk_dir = answer_file.parent / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[subprocess.Popen[bytes], Path, Path]] = []
    for index, gpu in enumerate(gpus):
        chunk_file = chunk_dir / f"{len(gpus)}_{index}.jsonl"
        chunk_log = chunk_dir / f"{len(gpus)}_{index}.log"
        child_env = os.environ.copy()
        child_env["CUDA_VISIBLE_DEVICES"] = gpu
        full_command = [
            *command,
            "--answers-file",
            str(chunk_file),
            "--num-chunks",
            str(len(gpus)),
            "--chunk-idx",
            str(index),
        ]
        print(f"+ CUDA_VISIBLE_DEVICES={gpu} " + " ".join(full_command), flush=True)
        log_handle = chunk_log.open("wb")
        process = subprocess.Popen(
            full_command,
            cwd=REPO_ROOT,
            env=child_env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        log_handle.close()
        processes.append((process, chunk_file, chunk_log))
    failures = []
    for process, _, chunk_log in processes:
        returncode = process.wait()
        if returncode:
            failures.append(f"pid={process.pid}, status={returncode}, log={chunk_log}")
    if failures:
        raise RunnerError("Inference worker failure: " + "; ".join(failures))
    merge_chunks([item[1] for item in processes], answer_file, dataset, expected)


def write_smoke_metric(dataset: str, metric_path: Path, predictions: Path, count: int) -> None:
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": dataset,
                "status": "smoke_only",
                "evaluator": "schema_validation",
                "sample_count": count,
                "prediction_file": str(predictions),
                "diagnostics": ["A subset run is not comparable with a paper metric."],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def evaluate(
    family: str,
    dataset: str,
    inputs: Mapping[str, Path],
    predictions: Path,
    run_dir: Path,
    smoke_count: int,
) -> None:
    metrics_dir = run_dir / "metrics" / dataset
    submissions_dir = run_dir / "submissions" / dataset
    metrics_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir.mkdir(parents=True, exist_ok=True)
    if smoke_count > 0:
        write_smoke_metric(dataset, metrics_dir / "smoke.json", predictions, smoke_count)
        return
    if dataset == "gqa":
        run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "convert_gqa_for_eval.py"),
                "--src",
                str(predictions),
                "--dst",
                str(submissions_dir / "predictions.json"),
            ]
        )
    elif dataset in {"mmbench", "mmbench_cn"}:
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "convert_mmbench_for_submission.py"),
            "--annotation-file",
            str(inputs["questions"]),
            "--result-dir",
            str(predictions.parent),
            "--upload-dir",
            str(submissions_dir),
            "--experiment",
            predictions.stem,
        ]
        if family == "qwen25vl":
            command.append("--normalize-choice")
        run(command)
    elif dataset == "vqav2":
        run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "convert_vqav2_for_submission.py"),
                "--src",
                str(predictions),
                "--questions",
                str(inputs["submission_questions"]),
                "--dst",
                str(submissions_dir / "predictions.json"),
            ]
        )
    elif dataset == "sqa":
        evaluator = "qwen/eval/eval_science_qa.py" if family == "qwen25vl" else "llava/eval/eval_science_qa.py"
        run(
            [
                sys.executable,
                str(REPO_ROOT / evaluator),
                "--base-dir",
                str(inputs["base"]),
                "--result-file",
                str(predictions),
                "--output-file",
                str(metrics_dir / "details.json"),
                "--output-result",
                str(metrics_dir / "evaluator.json"),
            ]
        )
    elif dataset == "textvqa":
        module = "qwen.eval.eval_textvqa" if family == "qwen25vl" else "llava.eval.eval_textvqa"
        run_to_file(
            [
                sys.executable,
                "-m",
                module,
                "--annotation-file",
                str(inputs["annotations"]),
                "--result-file",
                str(predictions),
                "--output-json",
                str(metrics_dir / "evaluator.json"),
            ],
            metrics_dir / "evaluator.log",
        )
    elif dataset == "pope":
        run_to_file(
            [
                sys.executable,
                str(REPO_ROOT / "llava" / "eval" / "eval_pope.py"),
                "--annotation-dir",
                str(inputs["annotations"]),
                "--question-file",
                str(inputs["questions"]),
                "--result-file",
                str(predictions),
                "--output-json",
                str(metrics_dir / "evaluator.json"),
            ],
            metrics_dir / "evaluator.log",
        )
    elif dataset == "mme":
        converted = metrics_dir / "converted"
        run(
            [
                sys.executable,
                "-m",
                "evaluation.mme.convert_answer_to_mme",
                "--answers-file",
                str(predictions),
                "--data-path",
                str(inputs["images"]),
                "--output-dir",
                str(converted),
                "--summary-json",
                str(metrics_dir / "conversion.json"),
            ]
        )
        run(
            [
                sys.executable,
                "-m",
                "evaluation.mme.calculation",
                "--results-dir",
                str(converted),
                "--output-json",
                str(metrics_dir / "evaluator.json"),
                "--json-only",
            ]
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=sorted(SUPPORTED_FAMILIES))
    parser.add_argument("dataset", choices=sorted(SUPPORTED_DATASETS))
    args = parser.parse_args(argv)
    try:
        configured_family = required_env("TReVS_FAMILY")
        if configured_family != args.family:
            raise RunnerError(
                f"Launcher family {args.family} does not match TReVS_FAMILY={configured_family}."
            )
        model_path = Path(required_env("MODEL_PATH")).resolve()
        data_root = Path(required_env("DATA_ROOT")).resolve()
        run_dir = Path(required_env("RUN_DIR")).resolve()
        if not model_path.exists():
            raise RunnerError(f"Model path does not exist: {model_path}")
        inputs = dataset_inputs(data_root, args.dataset)
        validate_inputs(inputs)
        gpus = [part.strip() for part in required_env("CUDA_VISIBLE_DEVICES").split(",") if part.strip()]
        if not gpus:
            raise RunnerError("CUDA_VISIBLE_DEVICES contains no device IDs.")
        max_samples = int(os.environ.get("MAX_SAMPLES", "0"))
        max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "128"))
        seed = int(os.environ.get("RANDOM_SEED", "42"))
        question_file, expected_count = subset_questions(
            inputs["questions"],
            run_dir / "input_subsets" / f"{args.dataset}{inputs['questions'].suffix}",
            max_samples,
        )
        identities = expected_identities(inputs["questions"], args.dataset, max_samples)
        if len(identities) != expected_count:
            raise RunnerError(
                f"Selected question identity count mismatch: {len(identities)} != {expected_count}."
            )
        predictions = run_dir / "predictions" / args.dataset / "predictions.jsonl"
        command = command_base(
            args.family,
            args.dataset,
            model_path,
            inputs,
            question_file,
            max_new_tokens,
            seed,
        )
        run_inference(command, predictions, gpus, args.dataset, identities)
        evaluate(
            args.family,
            args.dataset,
            inputs,
            predictions,
            run_dir,
            expected_count if max_samples > 0 else 0,
        )
        print(f"Completed {args.family}/{args.dataset}: {predictions}")
        return 0
    except (OSError, ValueError, RunnerError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
