"""Strict TextVQA v0.5.1 evaluator using the retained M4C normalizer."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any, Sequence


def _load_textvqa_evaluator():
    """Load the retained M4C evaluator without importing the LLaVA model stack."""

    evaluator_path = Path(__file__).resolve().parents[1] / "llava" / "eval" / "m4c_evaluator.py"
    spec = importlib.util.spec_from_file_location("trevs_m4c_evaluator", evaluator_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the retained M4C evaluator: {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TextVQAAccuracyEvaluator


TextVQAAccuracyEvaluator = _load_textvqa_evaluator()


class TextVQAEvaluationError(ValueError):
    pass


def prompt_question(prompt: object) -> str:
    text = str(prompt)
    if text.startswith("OCR tokens: "):
        match = re.search(r"Question: (.*?) Short answer:", text, re.DOTALL)
        if match is None:
            raise TextVQAEvaluationError("Malformed OCR-token TextVQA prompt.")
        question = match.group(1)
    elif "Reference OCR token: " in text and len(text.split("\n")) == 3:
        question = text.split("\n")[1] if text.startswith("Reference OCR token:") else text.split("\n")[0]
    elif len(text.split("\n")) == 2:
        question = text.split("\n")[0]
    else:
        raise TextVQAEvaluationError("Unsupported TextVQA prompt format.")
    return question.lower().strip()


def evaluate(annotation_file: Path, result_file: Path) -> dict[str, Any]:
    try:
        payload = json.loads(annotation_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TextVQAEvaluationError(f"Malformed TextVQA annotation JSON: {exc.msg}") from exc
    annotations = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(annotations, list) or not annotations:
        raise TextVQAEvaluationError("TextVQA annotation JSON has no nonempty data list.")
    keyed_annotations: dict[tuple[str, str], dict[str, Any]] = {}
    for index, annotation in enumerate(annotations, start=1):
        if not isinstance(annotation, dict) or not {"image_id", "question", "answers"}.issubset(annotation):
            raise TextVQAEvaluationError(f"Annotation {index} lacks image_id/question/answers.")
        answers = annotation["answers"]
        if (
            not isinstance(answers, list)
            or len(answers) != 10
            or any(not isinstance(answer, str) for answer in answers)
        ):
            raise TextVQAEvaluationError(
                f"Annotation {index} must contain exactly 10 string answers."
            )
        key = (str(annotation["image_id"]), str(annotation["question"]).lower().strip())
        if key in keyed_annotations:
            raise TextVQAEvaluationError(f"Duplicate TextVQA annotation key: {key!r}")
        keyed_annotations[key] = annotation

    predictions: dict[tuple[str, str], dict[str, Any]] = {}
    with result_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TextVQAEvaluationError(
                    f"Malformed prediction JSONL at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(row, dict) or not {"question_id", "prompt", "text"}.issubset(row):
                raise TextVQAEvaluationError(
                    f"Prediction line {line_number} lacks question_id/prompt/text."
                )
            key = (str(row["question_id"]), prompt_question(row["prompt"]))
            if key in predictions:
                raise TextVQAEvaluationError(f"Duplicate TextVQA prediction key: {key!r}")
            predictions[key] = row
    missing = set(keyed_annotations).difference(predictions)
    extra = set(predictions).difference(keyed_annotations)
    if missing or extra:
        raise TextVQAEvaluationError(
            f"Prediction/annotation key mismatch: missing={len(missing)}, extra={len(extra)}"
        )
    prediction_list = [
        {
            "pred_answer": predictions[key]["text"],
            "gt_answers": keyed_annotations[key]["answers"],
        }
        for key in keyed_annotations
    ]
    accuracy = 100.0 * TextVQAAccuracyEvaluator().eval_pred_list(prediction_list)
    return {
        "schema_version": 1,
        "dataset": "textvqa",
        "split": "val_v0.5.1",
        "status": "scored",
        "evaluator": "m4c_textvqa",
        "samples": len(prediction_list),
        "accuracy_percent": accuracy,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-file", type=Path, required=True)
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    try:
        summary = evaluate(args.annotation_file, args.result_file)
    except (OSError, TextVQAEvaluationError) as exc:
        parser.error(str(exc))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(result_file_name(args.result_file))
    print(f"Samples: {summary['samples']}\nAccuracy: {summary['accuracy_percent']:.2f}%\n")
    return 0


def result_file_name(path: Path) -> str:
    return path.stem


if __name__ == "__main__":
    raise SystemExit(main())
