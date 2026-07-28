"""Convert TReVS JSONL predictions to the tab-separated MME format."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple


GroundTruthKey = Tuple[str, str, str]
GroundTruth = Dict[GroundTruthKey, str]

SHORT_ANSWER_INSTRUCTION = "Answer the question using a single word or phrase."
YES_NO_INSTRUCTION = "Please answer yes or no."


class MMEConversionError(ValueError):
    """Raised when MME inputs are malformed, ambiguous, or incomplete."""


def sanitize_tsv_field(value: object) -> str:
    """Return a single-line, single-field representation accepted by MME."""

    if value is None:
        return ""
    text = str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")
    return " ".join(text.split())


def _read_jsonl(path: Path) -> List[Mapping[str, object]]:
    rows: List[Mapping[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MMEConversionError(
                    f"Malformed JSON in {path} at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise MMEConversionError(
                    f"Expected an object in {path} at line {line_number}."
                )
            rows.append(row)
    return rows


def load_ground_truth(data_path: str | Path) -> GroundTruth:
    """Load the official MME question/answer tree into a keyed mapping."""

    root = Path(data_path)
    if not root.is_dir():
        raise MMEConversionError(f"MME data directory does not exist: {root}")

    ground_truth: GroundTruth = {}
    for category_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if (category_dir / "images").is_dir():
            image_dir = category_dir / "images"
            qa_dir = category_dir / "questions_answers_YN"
        else:
            image_dir = category_dir
            qa_dir = category_dir
        if not image_dir.is_dir() or not qa_dir.is_dir():
            raise MMEConversionError(
                f"Category {category_dir.name!r} is missing images or questions_answers_YN."
            )

        for qa_file in sorted(qa_dir.glob("*.txt")):
            with qa_file.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    fields = line.rstrip("\r\n").split("\t")
                    if len(fields) != 2:
                        raise MMEConversionError(
                            f"Expected two TSV fields in {qa_file} at line {line_number}; "
                            f"found {len(fields)}."
                        )
                    question, answer = (field.strip() for field in fields)
                    if answer.lower() not in {"yes", "no"}:
                        raise MMEConversionError(
                            f"Invalid ground-truth answer {answer!r} in {qa_file} "
                            f"at line {line_number}."
                        )
                    key = (category_dir.name, qa_file.name, question)
                    if key in ground_truth:
                        raise MMEConversionError(
                            f"Duplicate ground-truth question in {qa_file} at line {line_number}."
                        )
                    ground_truth[key] = answer

    if not ground_truth:
        raise MMEConversionError(f"No MME ground-truth questions found below {root}")
    return ground_truth


def _prediction_identity(question_id: object) -> Tuple[str, str]:
    value = str(question_id).strip().replace("\\", "/")
    parts = [part for part in value.split("/") if part]
    if len(parts) < 2:
        raise MMEConversionError(
            f"MME question_id must contain a category and image name: {question_id!r}"
        )
    category = parts[0]
    filename = Path(parts[-1]).stem + ".txt"
    return category, filename


def _prompt_candidates(prompt: object) -> List[str]:
    base = str(prompt).replace(SHORT_ANSWER_INSTRUCTION, "").strip()
    candidates = [base]
    if YES_NO_INSTRUCTION not in base:
        candidates.extend(
            [
                f"{base} {YES_NO_INSTRUCTION}",
                f"{base}  {YES_NO_INSTRUCTION}",
            ]
        )
    # Preserve order while removing accidental duplicates.
    return list(dict.fromkeys(candidates))


def _resolve_ground_truth_key(
    category: str,
    filename: str,
    prompt: object,
    ground_truth: Mapping[GroundTruthKey, str],
) -> GroundTruthKey:
    matches = [
        (category, filename, candidate)
        for candidate in _prompt_candidates(prompt)
        if (category, filename, candidate) in ground_truth
    ]
    if not matches:
        raise MMEConversionError(
            f"Prediction does not match MME ground truth: category={category!r}, "
            f"file={filename!r}, prompt={str(prompt)!r}"
        )
    if len(matches) > 1:
        raise MMEConversionError(
            f"Prediction prompt is ambiguous in MME ground truth: {str(prompt)!r}"
        )
    return matches[0]


def convert_answers(
    answers_file: str | Path,
    data_path: str | Path,
    output_dir: str | Path,
    *,
    require_complete: bool = True,
) -> Dict[str, object]:
    """Convert predictions and return a structured conversion summary.

    The default is deliberately strict: every official question must have exactly
    one prediction. Set ``require_complete=False`` only for smoke-test subsets.
    """

    answer_path = Path(answers_file)
    if not answer_path.is_file():
        raise MMEConversionError(f"MME answer file does not exist: {answer_path}")
    ground_truth = load_ground_truth(data_path)
    predictions = _read_jsonl(answer_path)
    resolved: MutableMapping[GroundTruthKey, str] = {}

    required_fields = {"question_id", "prompt", "text"}
    for row_number, prediction in enumerate(predictions, start=1):
        missing_fields = sorted(required_fields.difference(prediction))
        if missing_fields:
            raise MMEConversionError(
                f"Prediction {row_number} is missing fields: {', '.join(missing_fields)}"
            )
        category, filename = _prediction_identity(prediction["question_id"])
        key = _resolve_ground_truth_key(
            category, filename, prediction["prompt"], ground_truth
        )
        if key in resolved:
            raise MMEConversionError(
                f"Duplicate prediction for category={key[0]!r}, file={key[1]!r}, "
                f"prompt={key[2]!r}"
            )
        resolved[key] = sanitize_tsv_field(prediction["text"])

    missing_keys = sorted(set(ground_truth).difference(resolved))
    if require_complete and missing_keys:
        example = missing_keys[0]
        raise MMEConversionError(
            f"Missing {len(missing_keys)} of {len(ground_truth)} MME predictions; "
            f"first missing key is {example!r}."
        )

    grouped: MutableMapping[str, List[Tuple[GroundTruthKey, str]]] = defaultdict(list)
    for key, prediction_text in resolved.items():
        grouped[key[0]].append((key, prediction_text))

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_files: List[str] = []
    for category in sorted(grouped):
        output_path = destination / f"{category}.txt"
        output_files.append(str(output_path))
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            for (key, prediction_text) in sorted(
                grouped[category], key=lambda item: (item[0][1], item[0][2])
            ):
                _, filename, prompt = key
                row = (
                    sanitize_tsv_field(filename),
                    sanitize_tsv_field(prompt),
                    sanitize_tsv_field(ground_truth[key]),
                    prediction_text,
                )
                handle.write("\t".join(row) + "\n")

    return {
        "schema_version": 1,
        "status": "converted" if not missing_keys else "incomplete",
        "predictions": len(resolved),
        "ground_truth_questions": len(ground_truth),
        "missing_predictions": len(missing_keys),
        "categories": sorted(grouped),
        "output_files": output_files,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers-file", required=True, help="TReVS JSONL predictions")
    parser.add_argument("--data-path", required=True, help="Official MME data root")
    parser.add_argument("--output-dir", required=True, help="Destination for category TSV files")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow a prediction subset (intended only for smoke tests)",
    )
    parser.add_argument("--summary-json", help="Optional conversion summary path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        summary = convert_answers(
            args.answers_file,
            args.data_path,
            args.output_dir,
            require_complete=not args.allow_incomplete,
        )
    except (OSError, MMEConversionError) as exc:
        raise SystemExit(f"MME conversion failed: {exc}") from exc

    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
