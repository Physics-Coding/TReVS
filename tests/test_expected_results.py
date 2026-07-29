from __future__ import annotations

import csv
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class ExpectedResultsTests(unittest.TestCase):
    def setUp(self) -> None:
        with (REPO_ROOT / "expected_results" / "paper_metrics.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            self.rows = list(csv.DictReader(handle))

    def test_schema_and_unique_metric_keys(self) -> None:
        self.assertEqual(
            list(self.rows[0]),
            [
                "family",
                "preset",
                "dataset",
                "split",
                "metric",
                "value",
                "tolerance",
                "seed",
                "evaluator",
                "source",
            ],
        )
        keys = [
            (row["family"], row["preset"], row["dataset"], row["split"], row["metric"])
            for row in self.rows
        ]
        self.assertEqual(len(keys), len(set(keys)))

    def test_final_table_anchor_values(self) -> None:
        values = {
            (row["family"], row["preset"], row["dataset"], row["metric"]): float(row["value"])
            for row in self.rows
        }
        self.assertEqual(values[("llava15", "32", "mme", "overall_score")], 1651.0)
        self.assertEqual(
            values[("llava15", "64", "aggregate", "relative_accuracy_percent")],
            96.5,
        )
        self.assertEqual(values[("llava15", "128", "gqa", "accuracy_percent")], 60.3)
        self.assertEqual(values[("llava_next", "320", "mme", "overall_score")], 1826.0)
        self.assertEqual(
            values[("videollava", "136", "aggregate", "relative_accuracy_percent")],
            99.0,
        )

    def test_only_table_backed_family_results_are_distributed(self) -> None:
        families = {row["family"] for row in self.rows}
        self.assertEqual(families, {"llava15", "llava_next", "videollava"})
        self.assertNotIn("qwen25vl", families)


if __name__ == "__main__":
    unittest.main()
