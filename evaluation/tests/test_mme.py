from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.mme.calculation import (
    EVAL_GROUPS,
    MMEScoringError,
    format_human_readable,
    parse_prediction,
    score_results,
)
from evaluation.mme.convert_answer_to_mme import (
    MMEConversionError,
    convert_answers,
)


FIXTURES = Path(__file__).parent / "fixtures"


class MMEConversionTests(unittest.TestCase):
    def test_complete_conversion_is_single_line_tsv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "converted"
            summary = convert_answers(
                FIXTURES / "mme_answers.jsonl",
                FIXTURES / "mme_data",
                output_dir,
            )
            lines = (output_dir / "existence.txt").read_text(encoding="utf-8").splitlines()

        self.assertEqual(summary["status"], "converted")
        self.assertEqual(summary["predictions"], 2)
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(len(line.split("\t")) == 4 for line in lines))
        self.assertTrue(all("\n" not in field for line in lines for field in line.split("\t")))

    def test_missing_prediction_fails_in_strict_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            answer_path = Path(temp_dir) / "answers.jsonl"
            answer_path.write_text(
                json.dumps(
                    {
                        "question_id": "existence/sample.jpg",
                        "prompt": "Is the synthetic object visible? Please answer yes or no.",
                        "text": "yes",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(MMEConversionError, "Missing 1 of 2"):
                convert_answers(
                    answer_path,
                    FIXTURES / "mme_data",
                    Path(temp_dir) / "converted",
                )

    def test_duplicate_prediction_fails(self) -> None:
        duplicate = (FIXTURES / "mme_answers.jsonl").read_text(encoding="utf-8").splitlines()[0]
        with tempfile.TemporaryDirectory() as temp_dir:
            answer_path = Path(temp_dir) / "answers.jsonl"
            answer_path.write_text(duplicate + "\n" + duplicate + "\n", encoding="utf-8")
            with self.assertRaisesRegex(MMEConversionError, "Duplicate prediction"):
                convert_answers(
                    answer_path,
                    FIXTURES / "mme_data",
                    Path(temp_dir) / "converted",
                    require_complete=False,
                )

    def test_malformed_jsonl_fails_with_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            answer_path = Path(temp_dir) / "answers.jsonl"
            answer_path.write_text('{"question_id":\n', encoding="utf-8")
            with self.assertRaisesRegex(MMEConversionError, "line 1"):
                convert_answers(
                    answer_path,
                    FIXTURES / "mme_data",
                    Path(temp_dir) / "converted",
                    require_complete=False,
                )


class MMEScoringTests(unittest.TestCase):
    @staticmethod
    def _write_all_categories(root: Path) -> None:
        for categories in EVAL_GROUPS.values():
            for category in categories:
                (root / f"{category}.txt").write_text(
                    "synthetic.jpg\tIs it visible?\tYes\tYes. certainly\n"
                    "synthetic.jpg\tIs it absent?\tNo\tNo\n",
                    encoding="utf-8",
                )

    def test_all_official_categories_score_to_2800(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_all_categories(root)
            summary = score_results(root)
        self.assertEqual(summary["status"], "scored")
        self.assertEqual(summary["metrics"]["perception_score"], 2000.0)
        self.assertEqual(summary["metrics"]["cognition_score"], 800.0)
        self.assertEqual(summary["metrics"]["overall_score"], 2800.0)
        self.assertIn("Overall MME Total", format_human_readable(summary))

    def test_missing_category_fails_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "existence.txt").write_text(
                "synthetic.jpg\tQ1\tYes\tYes\nsynthetic.jpg\tQ2\tNo\tNo\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(MMEScoringError, "Missing MME category files"):
                score_results(root)
            partial = score_results(root, require_all_categories=False)
        self.assertEqual(partial["status"], "partial")
        self.assertEqual(partial["metrics"]["overall_score"], 200.0)

    def test_odd_row_count_and_mismatched_pair_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            category = root / "existence.txt"
            category.write_text("one.jpg\tQ1\tYes\tYes\n", encoding="utf-8")
            with self.assertRaisesRegex(MMEScoringError, "expected pairs"):
                score_results(root, require_all_categories=False)

            category.write_text(
                "one.jpg\tQ1\tYes\tYes\ntwo.jpg\tQ2\tNo\tNo\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(MMEScoringError, "different image names"):
                score_results(root, require_all_categories=False)

    def test_prediction_parser_preserves_official_prefix_rule(self) -> None:
        self.assertEqual(parse_prediction("Yes, it is."), "yes")
        self.assertEqual(parse_prediction("Nope"), "no")
        self.assertEqual(parse_prediction("unclear"), "other")


if __name__ == "__main__":
    unittest.main()
