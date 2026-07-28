from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.pope import POPEEvaluationError, evaluate as evaluate_pope
from evaluation.scienceqa import ScienceQAEvaluationError, evaluate as evaluate_scienceqa
from evaluation.textvqa import TextVQAEvaluationError, evaluate as evaluate_textvqa


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class StrictScienceQATests(unittest.TestCase):
    def make_fixture(self, root: Path) -> tuple[Path, Path]:
        base = root / "scienceqa"
        base.mkdir()
        (base / "pid_splits.json").write_text(json.dumps({"test": ["1", "2"]}), encoding="utf-8")
        (base / "problems.json").write_text(
            json.dumps(
                {
                    "1": {"choices": ["red", "blue"], "answer": 0, "image": "one.png"},
                    "2": {"choices": ["up", "down"], "answer": 1, "image": None},
                }
            ),
            encoding="utf-8",
        )
        predictions = root / "sqa.jsonl"
        write_jsonl(
            predictions,
            [
                {"question_id": "1", "prompt": "<image> question", "text": "A"},
                {"question_id": "2", "prompt": "question", "text": "A"},
            ],
        )
        return base, predictions

    def test_reports_image_only_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary, _ = evaluate_scienceqa(*self.make_fixture(Path(directory)))
        self.assertEqual(summary["acc"], 50.0)
        self.assertEqual(summary["image_accuracy_percent"], 100.0)
        self.assertEqual(summary["image_count"], 1)

    def test_missing_prediction_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base, predictions = self.make_fixture(Path(directory))
            predictions.write_text(predictions.read_text().splitlines()[0] + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ScienceQAEvaluationError, "missing=1"):
                evaluate_scienceqa(base, predictions)

    def test_duplicate_id_and_malformed_json_are_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base, predictions = self.make_fixture(Path(directory))
            first_row = predictions.read_text(encoding="utf-8").splitlines()[0]
            predictions.write_text(first_row + "\n" + first_row + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ScienceQAEvaluationError, "Duplicate question_id"):
                evaluate_scienceqa(base, predictions)

            predictions.write_text('{"question_id":\n', encoding="utf-8")
            with self.assertRaisesRegex(ScienceQAEvaluationError, "Malformed"):
                evaluate_scienceqa(base, predictions)


class StrictTextVQATests(unittest.TestCase):
    def test_empty_prediction_is_scored_not_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = root / "annotations.json"
            predictions = root / "predictions.jsonl"
            annotations.write_text(
                json.dumps(
                    {
                        "data": [
                            {
                                "image_id": "image-1",
                                "question": "what color?",
                                "answers": ["red"] * 10,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            write_jsonl(
                predictions,
                [
                    {
                        "question_id": "image-1",
                        "prompt": "What color?\nShort answer:",
                        "text": "",
                    }
                ],
            )
            summary = evaluate_textvqa(annotations, predictions)
        self.assertEqual(summary["samples"], 1)
        self.assertEqual(summary["accuracy_percent"], 0.0)

    def test_duplicate_prediction_key_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = root / "annotations.json"
            predictions = root / "predictions.jsonl"
            annotations.write_text(
                json.dumps(
                    {
                        "data": [
                            {"image_id": "1", "question": "q?", "answers": ["a"] * 10}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            row = {"question_id": "1", "prompt": "Q?\nShort answer:", "text": "a"}
            write_jsonl(predictions, [row, row])
            with self.assertRaisesRegex(TextVQAEvaluationError, "Duplicate"):
                evaluate_textvqa(annotations, predictions)

    def test_missing_id_malformed_json_and_wrong_answer_count_are_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = root / "annotations.json"
            predictions = root / "predictions.jsonl"
            annotations.write_text(
                json.dumps(
                    {
                        "data": [
                            {
                                "image_id": "1",
                                "question": "q?",
                                "answers": ["a"] * 10,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            write_jsonl(
                predictions,
                [{"question_id": "2", "prompt": "Q?\nShort answer:", "text": "a"}],
            )
            with self.assertRaisesRegex(TextVQAEvaluationError, "missing=1, extra=1"):
                evaluate_textvqa(annotations, predictions)

            predictions.write_text('{"question_id":\n', encoding="utf-8")
            with self.assertRaisesRegex(TextVQAEvaluationError, "Malformed"):
                evaluate_textvqa(annotations, predictions)

            annotations.write_text(
                json.dumps(
                    {
                        "data": [
                            {"image_id": "1", "question": "q?", "answers": ["a"] * 9}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(TextVQAEvaluationError, "exactly 10"):
                evaluate_textvqa(annotations, predictions)


class StrictPOPETests(unittest.TestCase):
    def make_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        annotations = root / "annotations"
        annotations.mkdir()
        questions = root / "questions.jsonl"
        predictions = root / "predictions.jsonl"
        question_rows = []
        prediction_rows = []
        for index, category in enumerate(("adversarial", "popular", "random"), start=1):
            write_jsonl(annotations / f"coco_pope_{category}.json", [{"label": "yes"}])
            question_rows.append({"question_id": index, "category": category})
            prediction_rows.append({"question_id": index, "text": "yes"})
        write_jsonl(questions, question_rows)
        write_jsonl(predictions, prediction_rows)
        return annotations, questions, predictions

    def test_scores_all_three_categories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = evaluate_pope(*self.make_fixture(Path(directory)))
        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(summary["metrics"]["mean_f1"], 1.0)

    def test_missing_id_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            annotations, questions, predictions = self.make_fixture(Path(directory))
            predictions.write_text("\n".join(predictions.read_text().splitlines()[:2]) + "\n")
            with self.assertRaisesRegex(POPEEvaluationError, "missing=1"):
                evaluate_pope(annotations, questions, predictions)

    def test_malformed_json_and_category_count_mismatch_are_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            annotations, questions, predictions = self.make_fixture(Path(directory))
            questions.write_text('{"question_id":\n', encoding="utf-8")
            with self.assertRaisesRegex(POPEEvaluationError, "Malformed"):
                evaluate_pope(annotations, questions, predictions)

        with tempfile.TemporaryDirectory() as directory:
            annotations, questions, predictions = self.make_fixture(Path(directory))
            write_jsonl(
                annotations / "coco_pope_random.json",
                [{"label": "yes"}, {"label": "no"}],
            )
            with self.assertRaisesRegex(POPEEvaluationError, "counts differ"):
                evaluate_pope(annotations, questions, predictions)


if __name__ == "__main__":
    unittest.main()
