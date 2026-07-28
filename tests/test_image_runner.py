from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.common.run_image_benchmark import (
    RunnerError,
    command_base,
    expected_identities,
    load_questions,
    merge_chunks,
    subset_questions,
)


class ImageRunnerIdentityTests(unittest.TestCase):
    def test_multiline_tsv_is_counted_and_subset_by_logical_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "questions.tsv"
            source.write_text(
                'index\tquestion\timage\n1\t"first line\nsecond line"\tencoded-1\n'
                '2\t"another question"\tencoded-2\n',
                encoding="utf-8",
            )
            kind, _, count = load_questions(source)
            self.assertEqual((kind, count), ("tsv", 2))
            self.assertEqual(expected_identities(source, "mmbench", 0), [("1",), ("2",)])

            subset, subset_count = subset_questions(source, root / "subset.tsv", 1)
            self.assertEqual(subset_count, 1)
            self.assertEqual(load_questions(subset)[2], 1)
            self.assertEqual(expected_identities(subset, "mmbench", 0), [("1",)])

    def test_every_image_evaluator_receives_resolved_generation_limit(self) -> None:
        inputs = {"images": Path("/images")}
        for family in ("llava15", "llava_next", "qwen25vl"):
            for dataset in ("gqa", "mmbench", "mmbench_cn", "sqa"):
                with self.subTest(family=family, dataset=dataset):
                    command = command_base(
                        family,
                        dataset,
                        Path("/model"),
                        inputs,
                        Path("/questions"),
                        73,
                        42,
                    )
                    index = command.index("--max_new_tokens")
                    self.assertEqual(command[index + 1], "73")

    def test_textvqa_allows_repeated_image_id_with_distinct_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions = root / "questions.jsonl"
            questions.write_text(
                json.dumps({"question_id": "image-1", "text": "First question?"}) + "\n"
                + json.dumps({"question_id": "image-1", "text": "Second question?"})
                + "\n",
                encoding="utf-8",
            )
            expected = expected_identities(questions, "textvqa", 0)
            chunk = root / "chunk.jsonl"
            chunk.write_text(
                json.dumps(
                    {"question_id": "image-1", "prompt": "First question?", "text": "one"}
                )
                + "\n"
                + json.dumps(
                    {"question_id": "image-1", "prompt": "Second question?", "text": "two"}
                )
                + "\n",
                encoding="utf-8",
            )
            destination = root / "predictions.jsonl"
            merge_chunks([chunk], destination, "textvqa", expected)
            self.assertEqual(len(destination.read_text(encoding="utf-8").splitlines()), 2)

    def test_merge_rejects_equal_count_with_missing_and_extra_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunk = root / "chunk.jsonl"
            chunk.write_text(
                json.dumps({"question_id": 1, "text": "answer"}) + "\n"
                + json.dumps({"question_id": 3, "text": "answer"})
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RunnerError, "missing=1, extra=1"):
                merge_chunks([chunk], root / "predictions.jsonl", "gqa", [("1",), ("2",)])

    def test_merge_rejects_duplicate_composite_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {"question_id": "image-1", "prompt": "Question?", "text": "answer"}
            chunk = root / "chunk.jsonl"
            chunk.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RunnerError, "Duplicate prediction identity"):
                merge_chunks(
                    [chunk],
                    root / "predictions.jsonl",
                    "mme",
                    [("image-1", "Question?")],
                )


if __name__ == "__main__":
    unittest.main()
