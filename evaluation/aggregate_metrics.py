"""Normalize benchmark evaluator outputs into one auditable JSON summary.

This module does not estimate scores from model logs. It accepts the structured
output of an official/local evaluator when one exists, and marks upload-only
artifacts explicitly as ``submission_only``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Callable, Dict, List, Mapping, MutableMapping, Sequence


SUPPORTED_DATASETS = (
    "gqa",
    "mmbench",
    "mmbench_cn",
    "sqa",
    "textvqa",
    "vqav2",
    "mme",
    "pope",
)


class MetricParseError(ValueError):
    """Raised when an evaluator artifact is present but cannot be trusted."""


class DuplicateJSONKeyError(MetricParseError):
    """Raised when a JSON object contains duplicate keys."""


def _reject_duplicate_keys(pairs: Sequence[tuple[str, object]]) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJSONKeyError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> object:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise MetricParseError(
            f"Malformed JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc


def _load_jsonl(path: Path) -> List[Mapping[str, object]]:
    rows: List[Mapping[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
            except json.JSONDecodeError as exc:
                raise MetricParseError(
                    f"Malformed JSONL at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise MetricParseError(f"JSONL line {line_number} is not an object.")
            rows.append(row)
    if not rows:
        raise MetricParseError("The JSONL artifact is empty.")
    return rows


def _number(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise MetricParseError(f"{name} must be numeric, not boolean.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MetricParseError(f"{name} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise MetricParseError(f"{name} must be finite: {value!r}")
    return result


def _percent(value: object, name: str, *, ratio: bool | None = None) -> float:
    result = _number(value, name)
    if ratio is True or (ratio is None and 0.0 <= result <= 1.0):
        result *= 100.0
    if not 0.0 <= result <= 100.0:
        raise MetricParseError(f"{name} is outside [0, 100]: {result}")
    return result


def _nonnegative_int(value: object, name: str) -> int:
    numeric = _number(value, name)
    integer = int(numeric)
    if numeric != integer or integer < 0:
        raise MetricParseError(f"{name} must be a non-negative integer: {value!r}")
    return integer


def _positive_int(value: object, name: str) -> int:
    integer = _nonnegative_int(value, name)
    if integer == 0:
        raise MetricParseError(f"{name} must be greater than zero.")
    return integer


def _record(
    dataset: str,
    status: str,
    source: Path,
    evaluator: str,
    *,
    metrics: Mapping[str, object] | None = None,
    sample_count: int | None = None,
    diagnostics: Sequence[str] = (),
) -> Dict[str, object]:
    result: Dict[str, object] = {
        "dataset": dataset,
        "status": status,
        "source": str(source),
        "evaluator": evaluator,
        "metrics": dict(metrics or {}),
        "diagnostics": list(diagnostics),
    }
    if sample_count is not None:
        result["sample_count"] = sample_count
    return result


def _validate_unique_rows(
    rows: Sequence[Mapping[str, object]], id_keys: Sequence[str], required: Sequence[str]
) -> tuple[str, int]:
    if not rows:
        raise MetricParseError("Submission artifact contains no rows.")
    available_id_key = next(
        (key for key in id_keys if all(key in row for row in rows)), None
    )
    if available_id_key is None:
        raise MetricParseError(
            "Rows do not share a supported ID field: " + ", ".join(id_keys)
        )
    seen: set[object] = set()
    empty_values = 0
    for index, row in enumerate(rows, start=1):
        missing = [key for key in required if key not in row]
        if missing:
            raise MetricParseError(
                f"Row {index} is missing fields: {', '.join(sorted(missing))}"
            )
        row_id = row[available_id_key]
        if row_id is None or str(row_id).strip() == "":
            raise MetricParseError(f"Row {index} has an empty {available_id_key}.")
        try:
            duplicate = row_id in seen
        except TypeError as exc:
            raise MetricParseError(
                f"Row {index} has an invalid non-scalar {available_id_key}: {row_id!r}"
            ) from exc
        if duplicate:
            raise MetricParseError(
                f"Duplicate {available_id_key} in submission artifact: {row_id!r}"
            )
        seen.add(row_id)
        if any(row[key] is None or str(row[key]).strip() == "" for key in required):
            empty_values += 1
    return available_id_key, empty_values


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise MetricParseError("Expected a UTF-8 text evaluator artifact.") from exc


def parse_gqa(path: Path) -> Dict[str, object]:
    if path.suffix.lower() in {".json", ".jsonl"}:
        data = _load_jsonl(path) if path.suffix.lower() == ".jsonl" else _load_json(path)
        if isinstance(data, list):
            rows = [row for row in data if isinstance(row, dict)]
            if len(rows) != len(data):
                raise MetricParseError("GQA prediction artifact contains a non-object row.")
            _, empty = _validate_unique_rows(
                rows, ("questionId", "question_id"), ("prediction",)
            )
            return _record(
                "gqa",
                "submission_only",
                path,
                "external_official",
                metrics={"predictions": len(rows), "empty_predictions": empty},
                sample_count=len(rows),
                diagnostics=("Run the official GQA evaluator to obtain accuracy.",),
            )
        if not isinstance(data, dict):
            raise MetricParseError("GQA JSON must be an object or prediction list.")
        key = next(
            (name for name in ("accuracy_percent", "overall_accuracy", "accuracy") if name in data),
            None,
        )
        if key is None:
            raise MetricParseError("GQA score JSON has no recognized accuracy field.")
        sample_count = data.get("count", data.get("samples"))
        return _record(
            "gqa",
            "scored",
            path,
            "official_gqa",
            metrics={"accuracy_percent": _percent(data[key], key)},
            sample_count=_positive_int(sample_count, "sample count")
            if sample_count is not None
            else None,
        )

    text = _read_text(path)
    overall_matches = re.findall(
        r"(?im)^\s*overall\s+accuracy\s*(?:=|:)\s*([0-9]+(?:\.[0-9]+)?)\s*%?\s*$",
        text,
    )
    matches = overall_matches or re.findall(
        r"(?im)^\s*accuracy\s*(?:=|:)\s*([0-9]+(?:\.[0-9]+)?)\s*%?\s*$",
        text,
    )
    if len(matches) != 1:
        raise MetricParseError(
            f"Expected one GQA accuracy line, found {len(matches)}."
        )
    return _record(
        "gqa",
        "scored",
        path,
        "official_gqa",
        metrics={"accuracy_percent": _percent(matches[0], "accuracy")},
    )


def _mmbench_summary(data: Mapping[str, object], path: Path) -> Dict[str, object]:
    duplicate_indices = data.get("input_duplicate_indices", 0)
    missing_rows = data.get("official_rows_missing", 0)
    if _nonnegative_int(duplicate_indices, "input_duplicate_indices"):
        raise MetricParseError("MMBench summary reports duplicate input indices.")
    if _nonnegative_int(missing_rows, "official_rows_missing"):
        raise MetricParseError("MMBench summary reports missing official rows.")

    runs = data.get("runs")
    if not isinstance(runs, list) or not runs:
        raise MetricParseError("MMBench clean summary has no scoring runs.")
    selected = next(
        (
            run
            for run in runs
            if isinstance(run, dict)
            and run.get("effective_judge", run.get("requested_judge")) == "exact_matching"
        ),
        runs[0],
    )
    if not isinstance(selected, dict):
        raise MetricParseError("MMBench scoring run is not an object.")
    result_rows = selected.get("result")
    if not isinstance(result_rows, list) or len(result_rows) != 1:
        raise MetricParseError("Expected exactly one MMBench result row.")
    result = result_rows[0]
    if not isinstance(result, dict) or "Overall" not in result:
        raise MetricParseError("MMBench result row has no Overall score.")
    sample_count = data.get("clean_rows", data.get("input_rows"))
    if sample_count is None:
        raise MetricParseError("MMBench clean summary has no sample count.")
    clean_rows = _positive_int(sample_count, "clean_rows")
    if "input_rows" in data:
        input_rows = _positive_int(data["input_rows"], "input_rows")
        if clean_rows > input_rows:
            raise MetricParseError("MMBench clean_rows exceeds input_rows.")
    if "input_unique_indices" in data:
        unique_indices = _positive_int(
            data["input_unique_indices"], "input_unique_indices"
        )
        if "input_rows" in data and unique_indices > _nonnegative_int(
            data["input_rows"], "input_rows"
        ):
            raise MetricParseError("MMBench input_unique_indices exceeds input_rows.")
    metrics: Dict[str, object] = {
        "overall_percent": _percent(result["Overall"], "Overall", ratio=True),
        "judge": selected.get("effective_judge", selected.get("requested_judge")),
    }
    if "split" in result:
        metrics["split"] = result["split"]
    return _record(
        "mmbench",
        "scored",
        path,
        "official_mmbench",
        metrics=metrics,
        sample_count=clean_rows,
    )


def parse_mmbench(path: Path) -> Dict[str, object]:
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        return _record(
            "mmbench",
            "submission_only",
            path,
            "external_official",
            metrics={"artifact_bytes": path.stat().st_size},
            diagnostics=("Score this workbook with the official MMBench evaluator.",),
        )
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows_with_overall = [row for row in rows if row.get("Overall") not in {None, ""}]
        if len(rows_with_overall) != 1:
            raise MetricParseError(
                f"Expected one MMBench CSV row with Overall, found {len(rows_with_overall)}."
            )
        row = rows_with_overall[0]
        return _record(
            "mmbench",
            "scored",
            path,
            "official_mmbench",
            metrics={
                "overall_percent": _percent(row["Overall"], "Overall", ratio=True),
                "split": row.get("split"),
            },
        )
    data = _load_json(path)
    if isinstance(data, list):
        rows = [row for row in data if isinstance(row, dict)]
        if len(rows) != len(data):
            raise MetricParseError("MMBench submission contains a non-object row.")
        id_key, empty = _validate_unique_rows(rows, ("index", "question_id"), ("prediction",))
        return _record(
            "mmbench",
            "submission_only",
            path,
            "external_official",
            metrics={"predictions": len(rows), "empty_predictions": empty, "id_field": id_key},
            sample_count=len(rows),
            diagnostics=("Score this artifact with the official MMBench evaluator.",),
        )
    if not isinstance(data, dict):
        raise MetricParseError("MMBench JSON must be an object or submission list.")
    if "runs" in data:
        return _mmbench_summary(data, path)
    key = next((key for key in ("overall_percent", "Overall", "overall") if key in data), None)
    if key is None:
        raise MetricParseError("MMBench JSON has no clean-summary or Overall score.")
    return _record(
        "mmbench",
        "scored",
        path,
        "official_mmbench",
        metrics={"overall_percent": _percent(data[key], key)},
    )


def parse_mmbench_cn(path: Path) -> Dict[str, object]:
    record = parse_mmbench(path)
    record["dataset"] = "mmbench_cn"
    if record["evaluator"] == "official_mmbench":
        record["evaluator"] = "official_mmbench_cn"
    return record


def parse_sqa(path: Path) -> Dict[str, object]:
    data = _load_json(path)
    if not isinstance(data, dict):
        raise MetricParseError("ScienceQA result must be a JSON object.")
    missing = [
        key
        for key in (
            "acc",
            "correct",
            "count",
            "image_accuracy_percent",
            "image_correct",
            "image_count",
        )
        if key not in data
    ]
    if missing:
        raise MetricParseError("ScienceQA result is missing: " + ", ".join(missing))
    count = _nonnegative_int(data["count"], "count")
    correct = _nonnegative_int(data["correct"], "correct")
    if count == 0 or correct > count:
        raise MetricParseError("ScienceQA correct/count values are inconsistent.")
    accuracy = _percent(data["acc"], "acc", ratio=False)
    expected = 100.0 * correct / count
    if not math.isclose(accuracy, expected, abs_tol=0.02):
        raise MetricParseError(
            f"ScienceQA accuracy {accuracy} does not match correct/count ({expected})."
        )
    results = data.get("results")
    if results is not None and (not isinstance(results, dict) or len(results) != count):
        raise MetricParseError("ScienceQA results size does not match count.")
    image_count = _nonnegative_int(data["image_count"], "image_count")
    image_correct = _nonnegative_int(data["image_correct"], "image_correct")
    if image_count == 0 or image_correct > image_count:
        raise MetricParseError("ScienceQA image correct/count values are inconsistent.")
    image_accuracy = _percent(
        data["image_accuracy_percent"], "image_accuracy_percent", ratio=False
    )
    expected_image = 100.0 * image_correct / image_count
    if not math.isclose(image_accuracy, expected_image, abs_tol=0.02):
        raise MetricParseError(
            "ScienceQA image accuracy does not match image_correct/image_count "
            f"({expected_image})."
        )
    return _record(
        "sqa",
        "scored",
        path,
        "local_scienceqa",
        metrics={
            "accuracy_percent": accuracy,
            "correct": correct,
            "image_accuracy_percent": image_accuracy,
            "image_correct": image_correct,
            "image_count": image_count,
        },
        sample_count=count,
    )


def parse_textvqa(path: Path) -> Dict[str, object]:
    if path.suffix.lower() == ".json":
        data = _load_json(path)
        if not isinstance(data, dict):
            raise MetricParseError("TextVQA score JSON must be an object.")
        key = next((key for key in ("accuracy_percent", "accuracy", "acc") if key in data), None)
        if key is None:
            raise MetricParseError("TextVQA score JSON has no accuracy field.")
        count_value = data.get("samples", data.get("count"))
        return _record(
            "textvqa",
            "scored",
            path,
            "m4c_textvqa",
            metrics={"accuracy_percent": _percent(data[key], key)},
            sample_count=_positive_int(count_value, "sample count")
            if count_value is not None
            else None,
        )
    text = _read_text(path)
    sample_matches = re.findall(r"(?im)^\s*Samples:\s*([0-9]+)\s*$", text)
    accuracy_matches = re.findall(
        r"(?im)^\s*Accuracy:\s*([0-9]+(?:\.[0-9]+)?)%\s*$", text
    )
    if len(sample_matches) != 1 or len(accuracy_matches) != 1:
        raise MetricParseError(
            "Expected one TextVQA Samples line and one percentage Accuracy line."
        )
    return _record(
        "textvqa",
        "scored",
        path,
        "m4c_textvqa",
        metrics={"accuracy_percent": _percent(accuracy_matches[0], "Accuracy", ratio=False)},
        sample_count=_positive_int(sample_matches[0], "Samples"),
    )


def parse_vqav2(path: Path) -> Dict[str, object]:
    data = _load_json(path)
    if isinstance(data, list):
        rows = [row for row in data if isinstance(row, dict)]
        if len(rows) != len(data):
            raise MetricParseError("VQAv2 submission contains a non-object row.")
        _, empty = _validate_unique_rows(rows, ("question_id",), ("answer",))
        diagnostics = ["Upload this JSON to the official VQAv2 evaluation server."]
        if empty:
            diagnostics.append(f"The artifact contains {empty} empty answers.")
        return _record(
            "vqav2",
            "submission_only",
            path,
            "external_official",
            metrics={"predictions": len(rows), "empty_predictions": empty},
            sample_count=len(rows),
            diagnostics=diagnostics,
        )
    if not isinstance(data, dict):
        raise MetricParseError("VQAv2 result must be a JSON object or submission list.")
    score_container = data.get("accuracy", data)
    if not isinstance(score_container, dict):
        raise MetricParseError("VQAv2 accuracy field must be an object.")
    key = next((key for key in ("overall_percent", "overall", "Overall") if key in score_container), None)
    if key is None:
        raise MetricParseError("VQAv2 official result has no overall score.")
    count_value = data.get("count", data.get("samples"))
    return _record(
        "vqav2",
        "scored",
        path,
        "official_vqav2",
        metrics={"accuracy_percent": _percent(score_container[key], key)},
        sample_count=_positive_int(count_value, "sample count")
        if count_value is not None
        else None,
    )


def parse_mme(path: Path) -> Dict[str, object]:
    if path.suffix.lower() == ".json":
        data = _load_json(path)
        if not isinstance(data, dict):
            raise MetricParseError("MME score JSON must be an object.")
        metrics = data.get("metrics", data)
        if not isinstance(metrics, dict):
            raise MetricParseError("MME metrics field must be an object.")
        aliases = {
            "perception_score": ("perception_score", "perception_total"),
            "cognition_score": ("cognition_score", "cognition_total"),
            "overall_score": ("overall_score", "overall_total"),
        }
        parsed: Dict[str, float] = {}
        for output_key, candidates in aliases.items():
            source_key = next((key for key in candidates if key in metrics), None)
            if source_key is None:
                raise MetricParseError(f"MME JSON has no {output_key} field.")
            parsed[output_key] = _number(metrics[source_key], source_key)
        limits = {
            "perception_score": 2000.0,
            "cognition_score": 800.0,
            "overall_score": 2800.0,
        }
        for key, upper_bound in limits.items():
            if not 0.0 <= parsed[key] <= upper_bound:
                raise MetricParseError(
                    f"MME {key} is outside [0, {upper_bound}]: {parsed[key]}"
                )
        if not math.isclose(
            parsed["overall_score"],
            parsed["perception_score"] + parsed["cognition_score"],
            abs_tol=1e-6,
        ):
            raise MetricParseError("MME overall score does not equal its two group scores.")
        if data.get("status") == "partial" or data.get("missing_categories"):
            raise MetricParseError("MME score JSON is incomplete.")
        return _record(
            "mme", "scored", path, "official_mme", metrics=parsed
        )

    text = _read_text(path)
    values: Dict[str, float] = {}
    patterns = {
        "perception_score": r"=+\s*Perception\s*=+.*?^\s*total score:\s*([0-9]+(?:\.[0-9]+)?)",
        "cognition_score": r"=+\s*Cognition\s*=+.*?^\s*total score:\s*([0-9]+(?:\.[0-9]+)?)",
        "overall_score": r"=+\s*Overall MME Total\s*=+.*?^\s*total score:\s*([0-9]+(?:\.[0-9]+)?)",
    }
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if len(matches) != 1:
            raise MetricParseError(f"Expected one legacy MME {key} value, found {len(matches)}.")
        values[key] = _number(matches[0], key)
    if not math.isclose(
        values["overall_score"],
        values["perception_score"] + values["cognition_score"],
        abs_tol=1e-3,
    ):
        raise MetricParseError("Legacy MME total does not equal the two group totals.")
    return _record("mme", "scored", path, "official_mme", metrics=values)


def parse_pope(path: Path) -> Dict[str, object]:
    if path.suffix.lower() == ".json":
        data = _load_json(path)
        if not isinstance(data, dict):
            raise MetricParseError("POPE score JSON must be an object.")
        metrics = data.get("metrics", data)
        if not isinstance(metrics, dict) or "mean_f1" not in metrics:
            raise MetricParseError("POPE score JSON has no mean_f1 field.")
        mean_f1 = _number(metrics["mean_f1"], "mean_f1")
        if not 0.0 <= mean_f1 <= 1.0:
            raise MetricParseError("POPE mean_f1 must be a ratio in [0, 1].")
        output_metrics: Dict[str, object] = {"mean_f1": mean_f1}
        if "mean_accuracy" in metrics:
            mean_accuracy = _number(
                metrics["mean_accuracy"], "mean_accuracy"
            )
            if not 0.0 <= mean_accuracy <= 1.0:
                raise MetricParseError("POPE mean_accuracy must be a ratio in [0, 1].")
            output_metrics["mean_accuracy"] = mean_accuracy
        categories = metrics.get("categories")
        if categories is not None:
            if not isinstance(categories, dict):
                raise MetricParseError("POPE categories must be an object.")
            expected = {"random", "popular", "adversarial"}
            if set(categories) != expected:
                raise MetricParseError(
                    "POPE categories must be exactly: " + ", ".join(sorted(expected))
                )
            category_samples = 0
            category_f1_values: List[float] = []
            category_accuracy_values: List[float] = []
            for category_name, category in categories.items():
                if not isinstance(category, dict):
                    raise MetricParseError(
                        f"POPE category {category_name!r} must be an object."
                    )
                for field in ("samples", "f1", "accuracy"):
                    if field not in category:
                        raise MetricParseError(
                            f"POPE category {category_name!r} lacks {field}."
                        )
                category_samples += _positive_int(
                    category["samples"], f"{category_name} samples"
                )
                f1 = _number(category["f1"], f"{category_name} f1")
                accuracy = _number(category["accuracy"], f"{category_name} accuracy")
                if not 0.0 <= f1 <= 1.0 or not 0.0 <= accuracy <= 1.0:
                    raise MetricParseError("POPE category metrics must be ratios in [0, 1].")
                category_f1_values.append(f1)
                category_accuracy_values.append(accuracy)
            computed_f1 = sum(category_f1_values) / len(category_f1_values)
            if not math.isclose(mean_f1, computed_f1, abs_tol=1e-9):
                raise MetricParseError("POPE mean_f1 does not match category values.")
            if "mean_accuracy" in output_metrics:
                computed_accuracy = sum(category_accuracy_values) / len(
                    category_accuracy_values
                )
                if not math.isclose(
                    float(output_metrics["mean_accuracy"]), computed_accuracy, abs_tol=1e-9
                ):
                    raise MetricParseError(
                        "POPE mean_accuracy does not match category values."
                    )
            if "sample_count" in data and _positive_int(
                data["sample_count"], "sample_count"
            ) != category_samples:
                raise MetricParseError(
                    "POPE sample_count does not match category sample totals."
                )
            output_metrics["categories"] = categories
        return _record(
            "pope",
            "scored",
            path,
            "official_pope",
            metrics=output_metrics,
            sample_count=_positive_int(data["sample_count"], "sample_count")
            if "sample_count" in data
            else None,
        )

    text = _read_text(path)
    block_pattern = re.compile(
        r"(?ims)^Category:\s*([^,\n]+),\s*# samples:\s*([0-9]+)\s*$"
        r"(.*?)(?=^Category:|\Z)"
    )
    categories: MutableMapping[str, Dict[str, float | int]] = {}
    for match in block_pattern.finditer(text):
        category = match.group(1).strip().lower()
        if category in categories:
            raise MetricParseError(f"Duplicate POPE category: {category}")
        block = match.group(3)
        accuracy = re.findall(
            r"(?im)^\s*Accuracy:\s*([0-9]+(?:\.[0-9]+)?)\s*$", block
        )
        f1 = re.findall(
            r"(?im)^\s*F1 score:\s*([0-9]+(?:\.[0-9]+)?)\s*$", block
        )
        if len(accuracy) != 1 or len(f1) != 1:
            raise MetricParseError(
                f"POPE category {category!r} lacks one Accuracy/F1 pair."
            )
        accuracy_value = _number(accuracy[0], f"{category} accuracy")
        f1_value = _number(f1[0], f"{category} f1")
        if not 0.0 <= accuracy_value <= 1.0 or not 0.0 <= f1_value <= 1.0:
            raise MetricParseError("POPE category metrics must be ratios in [0, 1].")
        categories[category] = {
            "accuracy": accuracy_value,
            "f1": f1_value,
            "samples": _positive_int(match.group(2), "samples"),
        }

    expected = {"random", "popular", "adversarial"}
    missing = sorted(expected.difference(categories))
    if missing:
        raise MetricParseError("Missing POPE categories: " + ", ".join(missing))
    mean_f1 = sum(float(category["f1"]) for category in categories.values()) / len(categories)
    mean_accuracy = (
        sum(float(category["accuracy"]) for category in categories.values()) / len(categories)
    )
    return _record(
        "pope",
        "scored",
        path,
        "official_pope",
        metrics={
            "mean_f1": mean_f1,
            "mean_accuracy": mean_accuracy,
            "categories": categories,
        },
        sample_count=sum(int(category["samples"]) for category in categories.values()),
    )


PARSERS: Mapping[str, Callable[[Path], Dict[str, object]]] = {
    "gqa": parse_gqa,
    "mmbench": parse_mmbench,
    "mmbench_cn": parse_mmbench_cn,
    "sqa": parse_sqa,
    "textvqa": parse_textvqa,
    "vqav2": parse_vqav2,
    "mme": parse_mme,
    "pope": parse_pope,
}


def parse_dataset(dataset: str, source: str | Path | None) -> Dict[str, object]:
    """Parse one dataset artifact without allowing a bad source to abort a run."""

    normalized = dataset.lower()
    if normalized not in PARSERS:
        raise ValueError(
            f"Unsupported dataset {dataset!r}; expected one of {', '.join(SUPPORTED_DATASETS)}."
        )
    if source is None:
        return {
            "dataset": normalized,
            "status": "missing",
            "source": None,
            "evaluator": None,
            "metrics": {},
            "diagnostics": ["No evaluator artifact was provided."],
        }
    path = Path(source)
    if not path.is_file():
        return {
            "dataset": normalized,
            "status": "missing",
            "source": str(path),
            "evaluator": None,
            "metrics": {},
            "diagnostics": ["Evaluator artifact does not exist or is not a regular file."],
        }
    try:
        return PARSERS[normalized](path)
    except (OSError, MetricParseError) as exc:
        return {
            "dataset": normalized,
            "status": "invalid",
            "source": str(path),
            "evaluator": None,
            "metrics": {},
            "diagnostics": [str(exc)],
        }


def aggregate_metrics(
    sources: Mapping[str, str | Path | None],
    *,
    include_missing: bool = True,
) -> Dict[str, object]:
    """Parse evaluator artifacts for all supported benchmarks."""

    datasets = SUPPORTED_DATASETS if include_missing else tuple(sources)
    unknown = sorted(set(sources).difference(SUPPORTED_DATASETS))
    if unknown:
        raise ValueError("Unsupported datasets: " + ", ".join(unknown))
    records = {dataset: parse_dataset(dataset, sources.get(dataset)) for dataset in datasets}
    counts = {
        status: sum(record["status"] == status for record in records.values())
        for status in ("scored", "submission_only", "missing", "invalid")
    }
    return {
        "schema_version": 1,
        "datasets": records,
        "status_counts": counts,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for dataset in SUPPORTED_DATASETS:
        option_strings = [f"--{dataset}"]
        dashed_option = f"--{dataset.replace('_', '-')}"
        if dashed_option not in option_strings:
            option_strings.append(dashed_option)
        parser.add_argument(
            *option_strings,
            dest=dataset,
            metavar="PATH",
            help=f"{dataset.upper()} evaluator artifact",
        )
    parser.add_argument("--output", help="Write the aggregate JSON to this path")
    parser.add_argument(
        "--only-provided",
        action="store_true",
        help="Omit missing records for dataset arguments that were not provided",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit with status 2 if any requested artifact is missing or invalid",
    )
    parser.add_argument(
        "--require-scored",
        action="store_true",
        help="Also treat submission_only as a failing status",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    sources = {dataset: getattr(args, dataset) for dataset in SUPPORTED_DATASETS}
    if args.only_provided:
        sources = {dataset: path for dataset, path in sources.items() if path is not None}
    summary = aggregate_metrics(sources, include_missing=not args.only_provided)
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")

    requested = summary["datasets"]
    assert isinstance(requested, dict)
    failing = {"missing", "invalid"}
    if args.require_scored:
        failing.add("submission_only")
    if args.fail_on_error and any(record["status"] in failing for record in requested.values()):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
