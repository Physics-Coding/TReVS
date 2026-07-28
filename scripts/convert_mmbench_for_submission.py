#!/usr/bin/env python3
"""Create a complete MMBench submission workbook with strict ID checks."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence


def normalize_choice_prediction(prediction: object) -> object:
    if prediction is None:
        return prediction
    text = str(prediction).strip()
    patterns = (
        r"^\s*\(?\s*([A-D])\s*\)?\s*[\.:,)]?\s*$",
        r"^\s*\(?\s*([A-D])\s*\)?\s*[\.:,)]\s+",
        r"(?:answer|the\s+answer\s+is)\s*:?\s*\(?\s*([A-D])\s*\)?",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return text


def load_predictions(path: Path, normalize: bool) -> dict[object, object]:
    predictions: dict[object, object] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at line {line_number}: {exc.msg}") from exc
            if not isinstance(row, dict) or "question_id" not in row or "text" not in row:
                raise ValueError(f"Prediction line {line_number} lacks question_id/text.")
            question_id = row["question_id"]
            if question_id in predictions:
                raise ValueError(f"Duplicate question_id: {question_id!r}")
            value = normalize_choice_prediction(row["text"]) if normalize else row["text"]
            predictions[question_id] = value
    if not predictions:
        raise ValueError("Prediction file is empty.")
    return predictions


def convert(
    annotation_file: Path,
    result_file: Path,
    output_file: Path,
    *,
    normalize: bool,
) -> dict[str, int]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("MMBench conversion requires pandas and openpyxl.") from exc
    frame = pd.read_table(annotation_file)
    if "index" not in frame.columns:
        raise ValueError("MMBench annotation TSV has no index column.")
    if frame["index"].duplicated().any():
        raise ValueError("MMBench annotation TSV contains duplicate indices.")
    predictions = load_predictions(result_file, normalize)
    official_ids = set(frame["index"].tolist())
    normalized_predictions: dict[object, object] = {}
    for question_id, value in predictions.items():
        candidate = question_id
        if candidate not in official_ids:
            try:
                candidate = int(question_id)
            except (TypeError, ValueError):
                pass
        if candidate in normalized_predictions:
            raise ValueError(f"Duplicate normalized question_id: {candidate!r}")
        normalized_predictions[candidate] = value
    missing = official_ids.difference(normalized_predictions)
    extra = set(normalized_predictions).difference(official_ids)
    if missing or extra:
        raise ValueError(f"Prediction/index mismatch: missing={len(missing)}, extra={len(extra)}")
    removable = [
        column
        for column in ("hint", "category", "source", "image", "comment", "l2-category")
        if column in frame.columns
    ]
    output = frame.drop(columns=removable).copy()
    output["prediction"] = output["index"].map(normalized_predictions)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output.to_excel(output_file, index=False, engine="openpyxl")
    empty = int(output["prediction"].isna().sum()) + int(
        output["prediction"].fillna("").astype(str).str.strip().eq("").sum()
    )
    return {"predictions": len(output), "empty_predictions": empty}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-file", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--upload-dir", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--normalize-choice", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = convert(
            args.annotation_file,
            args.result_dir / f"{args.experiment}.jsonl",
            args.upload_dir / f"{args.experiment}.xlsx",
            normalize=args.normalize_choice,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
