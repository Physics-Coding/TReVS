#!/usr/bin/env python3
"""Build a strict VQAv2 submission file from TReVS JSONL predictions."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_answer_processor():
    evaluator_path = REPO_ROOT / "llava" / "eval" / "m4c_evaluator.py"
    spec = importlib.util.spec_from_file_location("trevs_vqav2_m4c_evaluator", evaluator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load answer processor: {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EvalAIAnswerProcessor


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at line {line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL line {line_number} is not an object.")
            rows.append(row)
    return rows


def convert(source: Path, questions: Path, destination: Path) -> dict[str, int]:
    predictions: dict[object, str] = {}
    empty_predictions = 0
    for row_number, row in enumerate(read_jsonl(source), start=1):
        missing = {"question_id", "text"}.difference(row)
        if missing:
            raise ValueError(f"Prediction {row_number} is missing: {', '.join(sorted(missing))}")
        question_id = row["question_id"]
        if question_id in predictions:
            raise ValueError(f"Duplicate prediction question_id: {question_id!r}")
        text = str(row["text"])
        empty_predictions += not text.strip()
        predictions[question_id] = text

    question_rows = read_jsonl(questions)
    question_ids: list[object] = []
    seen_questions: set[object] = set()
    for row_number, row in enumerate(question_rows, start=1):
        if "question_id" not in row:
            raise ValueError(f"Question {row_number} has no question_id.")
        question_id = row["question_id"]
        if question_id in seen_questions:
            raise ValueError(f"Duplicate official question_id: {question_id!r}")
        seen_questions.add(question_id)
        question_ids.append(question_id)

    missing_ids = [question_id for question_id in question_ids if question_id not in predictions]
    extra_ids = sorted(set(predictions).difference(seen_questions), key=str)
    if missing_ids or extra_ids:
        raise ValueError(
            f"Prediction/question ID mismatch: missing={len(missing_ids)}, extra={len(extra_ids)}"
        )

    answer_processor = load_answer_processor()()
    output = [
        {"question_id": question_id, "answer": answer_processor(predictions[question_id])}
        for question_id in question_ids
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=True) + "\n", encoding="utf-8")
    return {
        "predictions": len(output),
        "empty_predictions": empty_predictions,
        "missing_predictions": 0,
        "extra_predictions": 0,
    }


def legacy_paths(directory: Path, split: str, checkpoint: str) -> tuple[Path, Path, Path]:
    source = directory / "answers" / split / checkpoint / "merge.jsonl"
    questions = directory / "llava_vqav2_mscoco_test2015.jsonl"
    destination = directory / "answers_upload" / split / f"{checkpoint}.json"
    return source, questions, destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--dst", type=Path)
    parser.add_argument("--dir", type=Path, help="Legacy directory layout")
    parser.add_argument("--ckpt", help="Legacy experiment name")
    parser.add_argument("--split", help="Legacy split name")
    args = parser.parse_args(argv)
    if args.src or args.questions or args.dst:
        if not all((args.src, args.questions, args.dst)):
            parser.error("--src, --questions, and --dst must be supplied together.")
        source, questions, destination = args.src, args.questions, args.dst
    elif args.dir and args.ckpt and args.split:
        source, questions, destination = legacy_paths(args.dir, args.split, args.ckpt)
    else:
        parser.error("Use modern --src/--questions/--dst or all legacy arguments.")
    try:
        summary = convert(source, questions, destination)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
