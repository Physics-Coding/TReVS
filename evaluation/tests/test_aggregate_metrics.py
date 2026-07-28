from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.aggregate_metrics import (
    SUPPORTED_DATASETS,
    aggregate_metrics,
    main,
    parse_dataset,
)


FIXTURES = Path(__file__).parent / "fixtures"


class AggregateMetricsTests(unittest.TestCase):
    def test_all_supported_datasets_have_explicit_statuses(self) -> None:
        sources = {
            "gqa": FIXTURES / "gqa.txt",
            "mmbench": FIXTURES / "mmbench_clean_summary.json",
            "mmbench_cn": FIXTURES / "mmbench_clean_summary.json",
            "sqa": FIXTURES / "sqa_result.json",
            "textvqa": FIXTURES / "textvqa.txt",
            "vqav2": FIXTURES / "vqav2_submission.json",
            "mme": FIXTURES / "mme_score.json",
            "pope": FIXTURES / "pope.txt",
        }
        summary = aggregate_metrics(sources)

        self.assertEqual(set(summary["datasets"]), set(SUPPORTED_DATASETS))
        self.assertEqual(summary["status_counts"]["scored"], 7)
        self.assertEqual(summary["status_counts"]["submission_only"], 1)
        self.assertEqual(
            summary["datasets"]["gqa"]["metrics"]["accuracy_percent"], 62.5
        )
        self.assertEqual(
            summary["datasets"]["mmbench"]["metrics"]["overall_percent"], 75.0
        )
        self.assertEqual(summary["datasets"]["mmbench_cn"]["dataset"], "mmbench_cn")
        self.assertEqual(
            summary["datasets"]["mmbench_cn"]["evaluator"], "official_mmbench_cn"
        )
        self.assertAlmostEqual(
            summary["datasets"]["pope"]["metrics"]["mean_f1"],
            (1.0 + 0.6666666667) / 3,
        )
        self.assertEqual(
            summary["datasets"]["vqav2"]["metrics"]["empty_predictions"], 1
        )

    def test_missing_source_is_not_silently_scored(self) -> None:
        record = parse_dataset("gqa", FIXTURES / "does-not-exist.txt")
        self.assertEqual(record["status"], "missing")
        self.assertEqual(record["metrics"], {})

    def test_malformed_json_is_invalid(self) -> None:
        record = parse_dataset("sqa", FIXTURES / "malformed.json")
        self.assertEqual(record["status"], "invalid")
        self.assertIn("Malformed JSON", record["diagnostics"][0])

    def test_duplicate_json_key_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sqa.json"
            path.write_text(
                '{"acc": 50, "acc": 75, "correct": 1, "count": 2}',
                encoding="utf-8",
            )
            record = parse_dataset("sqa", path)
        self.assertEqual(record["status"], "invalid")
        self.assertIn("Duplicate JSON key", record["diagnostics"][0])

    def test_duplicate_submission_id_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "vqav2.json"
            path.write_text(
                json.dumps(
                    [
                        {"question_id": 1, "answer": "yes"},
                        {"question_id": 1, "answer": "no"},
                    ]
                ),
                encoding="utf-8",
            )
            record = parse_dataset("vqav2", path)
        self.assertEqual(record["status"], "invalid")
        self.assertIn("Duplicate question_id", record["diagnostics"][0])

    def test_mmbench_empty_predictions_are_counted_once_per_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mmbench.json"
            path.write_text(
                json.dumps(
                    [
                        {"index": 1, "prediction": None},
                        {"index": 2, "prediction": "   "},
                        {"index": 3, "prediction": "A"},
                    ]
                ),
                encoding="utf-8",
            )
            record = parse_dataset("mmbench", path)
        self.assertEqual(record["status"], "submission_only")
        self.assertEqual(record["sample_count"], 3)
        self.assertEqual(record["metrics"]["empty_predictions"], 2)

    def test_mmbench_duplicate_or_empty_id_is_invalid(self) -> None:
        invalid_rows = (
            [
                {"index": 1, "prediction": "A"},
                {"index": 1, "prediction": "B"},
            ],
            [{"index": "", "prediction": "A"}],
        )
        for rows in invalid_rows:
            with self.subTest(rows=rows), tempfile.TemporaryDirectory() as temp_dir:
                path = Path(temp_dir) / "mmbench.json"
                path.write_text(json.dumps(rows), encoding="utf-8")
                record = parse_dataset("mmbench", path)
                self.assertEqual(record["status"], "invalid")

    def test_mmbench_clean_summary_rejects_duplicate_missing_or_zero_rows(self) -> None:
        base = json.loads(
            (FIXTURES / "mmbench_clean_summary.json").read_text(encoding="utf-8")
        )
        mutations = (
            ("input_duplicate_indices", 1, "duplicate input indices"),
            ("official_rows_missing", 1, "missing official rows"),
            ("clean_rows", 0, "greater than zero"),
        )
        for field, value, message in mutations:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp_dir:
                payload = dict(base)
                payload[field] = value
                path = Path(temp_dir) / "summary.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                record = parse_dataset("mmbench", path)
                self.assertEqual(record["status"], "invalid")
                self.assertIn(message, record["diagnostics"][0])

    def test_scienceqa_results_size_must_match_sample_count(self) -> None:
        payload = json.loads((FIXTURES / "sqa_result.json").read_text(encoding="utf-8"))
        payload["results"].pop("synthetic-2")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sqa.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            record = parse_dataset("sqa", path)
        self.assertEqual(record["status"], "invalid")
        self.assertIn("results size", record["diagnostics"][0])

    def test_structured_pope_sample_count_must_match_categories(self) -> None:
        categories = {
            name: {"samples": 1, "f1": 1.0, "accuracy": 1.0}
            for name in ("adversarial", "popular", "random")
        }
        payload = {
            "sample_count": 4,
            "metrics": {
                "mean_f1": 1.0,
                "mean_accuracy": 1.0,
                "categories": categories,
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pope.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            record = parse_dataset("pope", path)
        self.assertEqual(record["status"], "invalid")
        self.assertIn("sample_count", record["diagnostics"][0])

    def test_inconsistent_scienceqa_totals_are_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sqa.json"
            path.write_text(
                json.dumps(
                    {
                        "acc": 90.0,
                        "correct": 1,
                        "count": 2,
                        "image_accuracy_percent": 100.0,
                        "image_correct": 1,
                        "image_count": 1,
                    }
                ),
                encoding="utf-8",
            )
            record = parse_dataset("sqa", path)
        self.assertEqual(record["status"], "invalid")
        self.assertIn("does not match", record["diagnostics"][0])

    def test_submission_only_fails_when_scored_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "summary.json"
            exit_code = main(
                [
                    "--vqav2",
                    str(FIXTURES / "vqav2_submission.json"),
                    "--only-provided",
                    "--output",
                    str(output),
                    "--fail-on-error",
                    "--require-scored",
                ]
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(exit_code, 2)
        self.assertEqual(payload["datasets"]["vqav2"]["status"], "submission_only")


if __name__ == "__main__":
    unittest.main()
