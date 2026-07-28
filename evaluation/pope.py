"""Strict POPE evaluator with structured per-category metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


EXPECTED_CATEGORIES = ("adversarial", "popular", "random")


class POPEEvaluationError(ValueError):
    pass


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise POPEEvaluationError(f"Malformed JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise POPEEvaluationError(f"JSONL row {path}:{line_number} is not an object.")
            rows.append(row)
    return rows


def binary_prediction(text: object) -> int:
    first_sentence = str(text).split(".", 1)[0].replace(",", "")
    words = first_sentence.split()
    return 0 if any(word in {"No", "no", "not"} for word in words) else 1


def category_metrics(predictions: Sequence[int], labels: Sequence[int]) -> dict[str, float | int]:
    if len(predictions) != len(labels) or not labels:
        raise POPEEvaluationError("POPE prediction and label counts differ or are empty.")
    tp = tn = fp = fn = 0
    for prediction, label in zip(predictions, labels):
        if prediction == 1 and label == 1:
            tp += 1
        elif prediction == 1 and label == 0:
            fp += 1
        elif prediction == 0 and label == 0:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(labels)
    return {
        "samples": len(labels),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": sum(predictions) / len(predictions),
    }


def evaluate(annotation_dir: Path, question_file: Path, result_file: Path) -> dict[str, Any]:
    questions = read_jsonl(question_file)
    question_by_id: dict[object, dict[str, Any]] = {}
    ordered_by_category: dict[str, list[object]] = {category: [] for category in EXPECTED_CATEGORIES}
    for index, question in enumerate(questions, start=1):
        if "question_id" not in question or "category" not in question:
            raise POPEEvaluationError(f"Question {index} lacks question_id/category.")
        question_id = question["question_id"]
        category = str(question["category"]).lower()
        if question_id in question_by_id:
            raise POPEEvaluationError(f"Duplicate question_id: {question_id!r}")
        if category not in ordered_by_category:
            raise POPEEvaluationError(f"Unsupported POPE category: {category!r}")
        question_by_id[question_id] = question
        ordered_by_category[category].append(question_id)

    answers: dict[object, dict[str, Any]] = {}
    for index, answer in enumerate(read_jsonl(result_file), start=1):
        if "question_id" not in answer or "text" not in answer:
            raise POPEEvaluationError(f"Answer {index} lacks question_id/text.")
        question_id = answer["question_id"]
        if question_id in answers:
            raise POPEEvaluationError(f"Duplicate answer question_id: {question_id!r}")
        answers[question_id] = answer
    missing = set(question_by_id).difference(answers)
    extra = set(answers).difference(question_by_id)
    if missing or extra:
        raise POPEEvaluationError(
            f"Answer/question ID mismatch: missing={len(missing)}, extra={len(extra)}"
        )

    categories: dict[str, dict[str, float | int]] = {}
    for category in EXPECTED_CATEGORIES:
        label_path = annotation_dir / f"coco_pope_{category}.json"
        label_rows = read_jsonl(label_path)
        labels = []
        for index, row in enumerate(label_rows, start=1):
            label = str(row.get("label", "")).lower()
            if label not in {"yes", "no"}:
                raise POPEEvaluationError(f"Invalid label in {label_path}:{index}")
            labels.append(1 if label == "yes" else 0)
        predictions = [
            binary_prediction(answers[question_id]["text"])
            for question_id in ordered_by_category[category]
        ]
        categories[category] = category_metrics(predictions, labels)
    mean_f1 = sum(float(value["f1"]) for value in categories.values()) / len(categories)
    mean_accuracy = sum(float(value["accuracy"]) for value in categories.values()) / len(categories)
    return {
        "schema_version": 1,
        "dataset": "pope",
        "split": "test_three_subsets",
        "status": "scored",
        "evaluator": "official_pope",
        "sample_count": len(questions),
        "metrics": {
            "mean_f1": mean_f1,
            "mean_accuracy": mean_accuracy,
            "categories": categories,
        },
    }


def print_legacy(summary: dict[str, Any]) -> None:
    categories = summary["metrics"]["categories"]
    for category in EXPECTED_CATEGORIES:
        metrics = categories[category]
        print(f"Category: {category}, # samples: {metrics['samples']}")
        print("TP\tFP\tTN\tFN\t")
        print(f"{metrics['tp']}\t{metrics['fp']}\t{metrics['tn']}\t{metrics['fn']}")
        print(f"Accuracy: {metrics['accuracy']}")
        print(f"Precision: {metrics['precision']}")
        print(f"Recall: {metrics['recall']}")
        print(f"F1 score: {metrics['f1']}")
        print(f"Yes ratio: {metrics['yes_ratio']}")
        print("====================================")
    print(summary["metrics"]["mean_f1"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--question-file", type=Path, required=True)
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    try:
        summary = evaluate(args.annotation_dir, args.question_file, args.result_file)
    except (OSError, POPEEvaluationError) as exc:
        parser.error(str(exc))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print_legacy(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
