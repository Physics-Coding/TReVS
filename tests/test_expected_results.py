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
        self.assertEqual(values[("llava15", "32", "mme", "overall_score")], 1749.1)
        self.assertEqual(
            values[("llava15", "64", "aggregate", "relative_accuracy_percent")],
            97.1,
        )
        self.assertEqual(values[("llava15", "128", "gqa", "accuracy_percent")], 60.4)
        self.assertEqual(values[("qwen25vl", "142", "textvqa", "accuracy_percent")], 72.0)
        self.assertEqual(values[("qwen25vl", "284", "mme", "overall_score")], 2372.0)
        self.assertEqual(
            values[("qwen25vl", "426", "aggregate", "relative_accuracy_percent")],
            100.1,
        )

    def test_unreported_families_have_no_invented_scores(self) -> None:
        self.assertEqual({row["family"] for row in self.rows}, {"llava15", "qwen25vl"})


if __name__ == "__main__":
    unittest.main()
