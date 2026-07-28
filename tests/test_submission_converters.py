from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.convert_gqa_for_eval import convert as convert_gqa
from scripts.convert_vqav2_for_submission import convert as convert_vqav2


class SubmissionConverterTests(unittest.TestCase):
    def test_gqa_conversion_rejects_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "predictions.jsonl"
            row = {"question_id": 1, "text": "Blue."}
            source.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate question_id"):
                convert_gqa(source, root / "submission.json")

    def test_gqa_conversion_keeps_empty_prediction_in_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "predictions.jsonl"
            source.write_text(
                json.dumps({"question_id": 1, "text": ""}) + "\n",
                encoding="utf-8",
            )
            summary = convert_gqa(source, root / "submission.json")
            self.assertEqual(summary, {"predictions": 1, "empty_predictions": 1})

    def test_vqav2_rejects_missing_and_extra_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions = root / "questions.jsonl"
            predictions = root / "predictions.jsonl"
            questions.write_text(
                json.dumps({"question_id": 1}) + "\n" + json.dumps({"question_id": 2}) + "\n",
                encoding="utf-8",
            )
            predictions.write_text(
                json.dumps({"question_id": 1, "text": "yes"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing=1"):
                convert_vqav2(predictions, questions, root / "submission.json")

    def test_vqav2_complete_conversion_is_ordered_by_official_questions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions = root / "questions.jsonl"
            predictions = root / "predictions.jsonl"
            questions.write_text(
                json.dumps({"question_id": 2}) + "\n" + json.dumps({"question_id": 1}) + "\n",
                encoding="utf-8",
            )
            predictions.write_text(
                json.dumps({"question_id": 1, "text": "two"}) + "\n"
                + json.dumps({"question_id": 2, "text": "one"})
                + "\n",
                encoding="utf-8",
            )
            destination = root / "submission.json"
            convert_vqav2(predictions, questions, destination)
            rows = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual([row["question_id"] for row in rows], [2, 1])


if __name__ == "__main__":
    unittest.main()
