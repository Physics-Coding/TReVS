"""Strict ScienceQA evaluator with full-test and image-only metrics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence


class ScienceQAEvaluationError(ValueError):
    pass


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ScienceQAEvaluationError(f"Malformed JSON in {path}: {exc.msg}") from exc


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ScienceQAEvaluationError(
                    f"Malformed prediction JSONL at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(row, dict) or "question_id" not in row or "text" not in row:
                raise ScienceQAEvaluationError(
                    f"Prediction line {line_number} lacks question_id/text."
                )
            question_id = str(row["question_id"])
            if question_id in predictions:
                raise ScienceQAEvaluationError(f"Duplicate question_id: {question_id!r}")
            predictions[question_id] = row
    if not predictions:
        raise ScienceQAEvaluationError("Prediction file is empty.")
    return predictions


def parse_choice(text: object, choice_count: int, options: Sequence[str]) -> str:
    value = str(text).strip()
    valid = list(options[:choice_count])
    if value in valid:
        return value
    valid_pattern = "".join(re.escape(option) for option in valid)
    patterns = (
        rf"^\s*\(?\s*([{valid_pattern}])\s*\)?\s*[\.:,)]?\s*$",
        rf"^\s*\(?\s*([{valid_pattern}])\s*\)?\s*[\.:,)]\s+",
        rf"(?:the\s+answer\s+is|answer\s*:)\s*\(?\s*([{valid_pattern}])\s*\)?",
    )
    for pattern in patterns:
        match = re.search(pattern, value, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return "FAILED"


def evaluate(
    base_dir: Path,
    result_file: Path,
    *,
    split: str = "test",
    options: Sequence[str] = ("A", "B", "C", "D", "E"),
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    split_payload = load_json(base_dir / "pid_splits.json")
    problems_payload = load_json(base_dir / "problems.json")
    if not isinstance(split_payload, dict) or split not in split_payload:
        raise ScienceQAEvaluationError(f"ScienceQA split is unavailable: {split!r}")
    if not isinstance(problems_payload, dict):
        raise ScienceQAEvaluationError("ScienceQA problems.json must be an object.")
    split_ids = [str(value) for value in split_payload[split]]
    if len(split_ids) != len(set(split_ids)):
        raise ScienceQAEvaluationError(f"ScienceQA split {split!r} contains duplicate IDs.")
    missing_problems = [question_id for question_id in split_ids if question_id not in problems_payload]
    if missing_problems:
        raise ScienceQAEvaluationError(
            f"ScienceQA problems.json is missing {len(missing_problems)} split IDs."
        )
    predictions = load_predictions(result_file)
    missing = sorted(set(split_ids).difference(predictions))
    extra = sorted(set(predictions).difference(split_ids))
    if missing or extra:
        raise ScienceQAEvaluationError(
            f"Prediction/split ID mismatch: missing={len(missing)}, extra={len(extra)}"
        )

    details: dict[str, list[dict[str, Any]]] = {"correct": [], "incorrect": []}
    result_indices: dict[str, int] = {}
    outputs: dict[str, str] = {}
    correct = 0
    image_correct = 0
    image_count = 0
    for question_id in split_ids:
        problem = problems_payload[question_id]
        if not isinstance(problem, dict):
            raise ScienceQAEvaluationError(f"Problem {question_id!r} is not an object.")
        choices = problem.get("choices")
        answer_index = problem.get("answer")
        if not isinstance(choices, list) or not isinstance(answer_index, int):
            raise ScienceQAEvaluationError(f"Problem {question_id!r} lacks choices/answer.")
        if answer_index < 0 or answer_index >= len(choices) or len(choices) > len(options):
            raise ScienceQAEvaluationError(f"Problem {question_id!r} has an invalid answer index.")
        prediction = predictions[question_id]
        prediction_text = str(prediction["text"])
        parsed = parse_choice(prediction_text, len(choices), options)
        parsed_index = options.index(parsed) if parsed in options[: len(choices)] else -1
        is_correct = parsed_index == answer_index
        is_image = bool(problem.get("image"))
        correct += int(is_correct)
        image_count += int(is_image)
        image_correct += int(is_image and is_correct)
        record = {
            "question_id": question_id,
            "parsed_answer": parsed,
            "ground_truth": options[answer_index],
            "prompt": prediction.get("prompt", ""),
            "prediction": prediction_text,
            "is_image_question": is_image,
        }
        details["correct" if is_correct else "incorrect"].append(record)
        result_indices[question_id] = parsed_index
        outputs[question_id] = prediction_text
    count = len(split_ids)
    if count == 0 or image_count == 0:
        raise ScienceQAEvaluationError("ScienceQA split or image-question subset is empty.")
    summary = {
        "schema_version": 1,
        "dataset": "sqa",
        "split": split,
        "status": "scored",
        "evaluator": "local_scienceqa",
        "acc": 100.0 * correct / count,
        "correct": correct,
        "count": count,
        "image_accuracy_percent": 100.0 * image_correct / image_count,
        "image_correct": image_correct,
        "image_count": image_count,
        "results": result_indices,
        "outputs": outputs,
    }
    return summary, details


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--output-result", type=Path, required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args(argv)
    try:
        summary, details = evaluate(args.base_dir, args.result_file, split=args.split)
    except (OSError, ScienceQAEvaluationError) as exc:
        parser.error(str(exc))
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_result.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(details, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_result.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Total: {summary['count']}, Correct: {summary['correct']}, "
        f"Accuracy: {summary['acc']:.2f}%, "
        f"IMG-Accuracy: {summary['image_accuracy_percent']:.2f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
