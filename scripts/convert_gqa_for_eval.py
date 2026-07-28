#!/usr/bin/env python3
"""Convert unique TReVS JSONL predictions to the official GQA JSON schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def convert(source: Path, destination: Path) -> dict[str, int]:
    rows = []
    seen: set[object] = set()
    empty_predictions = 0
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                prediction = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at line {line_number}: {exc.msg}") from exc
            if not isinstance(prediction, dict):
                raise ValueError(f"Prediction line {line_number} is not an object.")
            missing = {"question_id", "text"}.difference(prediction)
            if missing:
                raise ValueError(
                    f"Prediction line {line_number} is missing: {', '.join(sorted(missing))}"
                )
            question_id = prediction["question_id"]
            if question_id in seen:
                raise ValueError(f"Duplicate question_id: {question_id!r}")
            seen.add(question_id)
            text = str(prediction["text"]).rstrip(".").lower()
            empty_predictions += not text.strip()
            rows.append({"questionId": question_id, "prediction": text})
    if not rows:
        raise ValueError("Prediction file is empty.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(rows, ensure_ascii=True) + "\n", encoding="utf-8")
    return {"predictions": len(rows), "empty_predictions": empty_predictions}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = convert(args.src, args.dst)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
