"""Compute the official MME accuracy, accuracy-plus, and aggregate scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple


EVAL_GROUPS: Mapping[str, Tuple[str, ...]] = {
    "Perception": (
        "existence",
        "count",
        "position",
        "color",
        "posters",
        "celebrity",
        "scene",
        "landmark",
        "artwork",
        "OCR",
    ),
    "Cognition": (
        "commonsense_reasoning",
        "numerical_calculation",
        "text_translation",
        "code_reasoning",
    ),
}


class MMEScoringError(ValueError):
    """Raised when converted MME result files violate the official schema."""


def parse_prediction(answer: str) -> str:
    """Apply the official MME yes/no prefix rule."""

    normalized = answer.strip().lower()
    if normalized in {"yes", "no"}:
        return normalized
    prefix = normalized[:4]
    if "yes" in prefix:
        return "yes"
    if "no" in prefix:
        return "no"
    return "other"


def compute_binary_metrics(gts: Sequence[str], preds: Sequence[str]) -> Dict[str, float | int]:
    if len(gts) != len(preds):
        raise MMEScoringError("Ground-truth and prediction counts differ.")
    if not gts:
        raise MMEScoringError("Cannot score an empty MME category.")

    tp = sum(gt == "yes" and pred == "yes" for gt, pred in zip(gts, preds))
    fn = sum(gt == "yes" and pred == "no" for gt, pred in zip(gts, preds))
    fp = sum(gt == "no" and pred == "yes" for gt, pred in zip(gts, preds))
    tn = sum(gt == "no" and pred == "no" for gt, pred in zip(gts, preds))
    other_count = sum(pred == "other" for pred in preds)
    correct = sum(gt == pred for gt, pred in zip(gts, preds))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "TP": tp,
        "FN": fn,
        "TN": tn,
        "FP": fp,
        "precision": precision,
        "recall": recall,
        "other_num": other_count,
        "acc": correct / len(gts),
    }


def _read_category(path: Path) -> List[Tuple[str, str, str, str]]:
    rows: List[Tuple[str, str, str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 4:
                raise MMEScoringError(
                    f"Expected four TSV fields in {path} at line {line_number}; "
                    f"found {len(fields)}."
                )
            image_name, question, gt_answer, prediction = fields
            gt = gt_answer.strip().lower()
            if gt not in {"yes", "no"}:
                raise MMEScoringError(
                    f"Invalid ground-truth answer {gt_answer!r} in {path} "
                    f"at line {line_number}."
                )
            rows.append((image_name, question, gt, parse_prediction(prediction)))
    if not rows:
        raise MMEScoringError(f"MME category file is empty: {path}")
    if len(rows) % 2:
        raise MMEScoringError(
            f"MME category {path} has {len(rows)} rows; expected pairs of two questions."
        )
    return rows


def _score_category(path: Path) -> Dict[str, float | int]:
    rows = _read_category(path)
    gts = [row[2] for row in rows]
    predictions = [row[3] for row in rows]
    metrics = compute_binary_metrics(gts, predictions)

    correct_pairs = 0
    for offset in range(0, len(rows), 2):
        pair = rows[offset : offset + 2]
        if pair[0][0] != pair[1][0]:
            raise MMEScoringError(
                f"MME pair in {path} at lines {offset + 1}-{offset + 2} "
                "contains different image names."
            )
        if all(row[2] == row[3] for row in pair):
            correct_pairs += 1

    image_count = len(rows) // 2
    acc_plus = correct_pairs / image_count
    metrics.update(
        {
            "acc_plus": acc_plus,
            "score": 100.0 * (float(metrics["acc"]) + acc_plus),
            "questions": len(rows),
            "images": image_count,
        }
    )
    return metrics


class MMEMetricsCalculator:
    """Reusable implementation of the official MME calculation script."""

    def score(self, results_dir: str | Path, *, require_all_categories: bool = True) -> Dict[str, object]:
        root = Path(results_dir)
        if not root.is_dir():
            raise MMEScoringError(f"MME results directory does not exist: {root}")

        categories: MutableMapping[str, Dict[str, float | int]] = {}
        groups: MutableMapping[str, Dict[str, object]] = {}
        expected_categories = {
            category for group_categories in EVAL_GROUPS.values() for category in group_categories
        }
        available_categories = {path.stem for path in root.glob("*.txt")}
        missing_categories = sorted(expected_categories.difference(available_categories))
        if require_all_categories and missing_categories:
            raise MMEScoringError(
                "Missing MME category files: " + ", ".join(missing_categories)
            )

        for group_name, group_categories in EVAL_GROUPS.items():
            group_scores: MutableMapping[str, float] = {}
            for category in group_categories:
                path = root / f"{category}.txt"
                if not path.is_file():
                    continue
                category_metrics = _score_category(path)
                categories[category] = category_metrics
                group_scores[category] = float(category_metrics["score"])
            groups[group_name] = {
                "score": sum(group_scores.values()),
                "categories": group_scores,
            }

        if not categories:
            raise MMEScoringError(f"No recognized MME category files found in {root}")
        return {
            "schema_version": 1,
            "dataset": "mme",
            "status": "scored" if not missing_categories else "partial",
            "evaluator": "official_mme",
            "groups": groups,
            "categories": categories,
            "metrics": {
                "perception_score": float(groups["Perception"]["score"]),
                "cognition_score": float(groups["Cognition"]["score"]),
                "overall_score": sum(float(group["score"]) for group in groups.values()),
            },
            "missing_categories": missing_categories,
        }


def score_results(
    results_dir: str | Path, *, require_all_categories: bool = True
) -> Dict[str, object]:
    return MMEMetricsCalculator().score(
        results_dir, require_all_categories=require_all_categories
    )


def format_human_readable(summary: Mapping[str, object]) -> str:
    lines: List[str] = []
    groups = summary["groups"]
    assert isinstance(groups, dict)
    for group_name in EVAL_GROUPS:
        group = groups[group_name]
        assert isinstance(group, dict)
        lines.append(f"=========== {group_name} ===========")
        lines.append(f"total score: {float(group['score']):.6f}")
        categories = group["categories"]
        assert isinstance(categories, dict)
        for category, score in categories.items():
            lines.append(f"\t{category} score: {float(score):.6f}")
        lines.append("")
    metrics = summary["metrics"]
    assert isinstance(metrics, dict)
    lines.append("=========== Overall MME Total ===========")
    lines.append(f"total score: {float(metrics['overall_score']):.6f}")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, help="Directory of MME category TSV files")
    parser.add_argument("--output-json", help="Optional structured result path")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Score available categories only (intended only for smoke tests)",
    )
    parser.add_argument(
        "--json-only", action="store_true", help="Print JSON instead of the legacy human summary"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        summary = score_results(
            args.results_dir, require_all_categories=not args.allow_incomplete
        )
    except (OSError, MMEScoringError) as exc:
        raise SystemExit(f"MME scoring failed: {exc}") from exc

    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered if args.json_only else format_human_readable(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
