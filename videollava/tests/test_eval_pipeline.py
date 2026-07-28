import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from videollava.eval.video.datasets import get_dataset_paths
from videollava.eval.video.eval_video_qa import (
    PROMPT_VERSION,
    RateLimiter,
    _load_existing_judgments,
    aggregate_judgments,
    evaluate_predictions,
    parse_judge_content,
    request_judgment,
)
from videollava.eval.video.merge_chunks import merge_chunk_files
from videollava.eval.video.run_inference_video_qa import (
    load_annotation_chunk,
    normalize_generated_answer,
    run_inference,
)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, records) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


class DatasetAndChunkTests(unittest.TestCase):
    def test_eval_root_resolves_direct_directory_before_packaged_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            direct_root = root / "TGIF_Zero_Shot_QA"
            packaged_root = root / "GPT_Zero_Shot_QA" / "TGIF_Zero_Shot_QA"
            (direct_root / "mp4").mkdir(parents=True)
            (packaged_root / "mp4").mkdir(parents=True)
            video_dir, questions, answers = get_dataset_paths(root, "tgif")
            self.assertEqual(video_dir, direct_root / "mp4")
            self.assertEqual(questions, direct_root / "test_q.json")
            self.assertEqual(answers, direct_root / "test_a.json")

    def test_eval_root_resolves_packaged_intermediate_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / "GPT_Zero_Shot_QA" / "TGIF_Zero_Shot_QA"
            (dataset_root / "mp4").mkdir(parents=True)
            video_dir, questions, answers = get_dataset_paths(root, "tgif")
            self.assertEqual(video_dir, dataset_root / "mp4")
            self.assertEqual(questions, dataset_root / "test_q.json")
            self.assertEqual(answers, dataset_root / "test_a.json")

    def test_answer_lookup_survives_global_truncation_and_chunking(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions = [
                {"question_id": str(index), "question": f"q{index}", "video_name": "v"}
                for index in range(7)
            ]
            answers = [
                {"question_id": str(index), "answer": f"a{index}"}
                for index in reversed(range(7))
            ]
            question_file = root / "q.json"
            answer_file = root / "a.json"
            write_json(question_file, questions)
            write_json(answer_file, answers)
            args = argparse.Namespace(
                gt_file_question=question_file,
                gt_file_answers=answer_file,
                max_samples=5,
                num_chunks=2,
                chunk_idx=1,
            )
            chunk, answers_by_id = load_annotation_chunk(args)
            self.assertEqual([item["question_id"] for item in chunk], ["3", "4"])
            self.assertEqual(answers_by_id["3"], "a3")
            self.assertEqual(answers_by_id["4"], "a4")

    def test_strict_merge_restores_question_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            question_file = root / "questions.json"
            write_json(
                question_file,
                [{"question_id": value} for value in ("a", "b", "c")],
            )
            chunk0 = root / "2_0.jsonl"
            chunk1 = root / "2_1.jsonl"
            make_record = lambda value: {
                "id": value,
                "question": value,
                "answer": value,
                "pred": value,
            }
            write_jsonl(chunk0, [make_record("b"), make_record("a")])
            write_jsonl(chunk1, [make_record("c")])
            output = root / "merged.jsonl"
            count = merge_chunk_files(question_file, [chunk0, chunk1], output)
            self.assertEqual(count, 3)
            observed = [json.loads(line)["id"] for line in output.read_text().splitlines()]
            self.assertEqual(observed, ["a", "b", "c"])

    def test_merge_rejects_duplicate_and_missing_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            question_file = root / "questions.json"
            write_json(question_file, [{"question_id": "a"}, {"question_id": "b"}])
            record = {"id": "a", "question": "q", "answer": "a", "pred": "p"}
            chunk0 = root / "0.jsonl"
            chunk1 = root / "1.jsonl"
            write_jsonl(chunk0, [record])
            write_jsonl(chunk1, [record])
            with self.assertRaisesRegex(ValueError, "Duplicate prediction"):
                merge_chunk_files(question_file, [chunk0, chunk1], root / "out.jsonl")
            write_jsonl(chunk1, [])
            with self.assertRaisesRegex(ValueError, "incomplete"):
                merge_chunk_files(question_file, [chunk0, chunk1], root / "out.jsonl")

    def test_stop_only_answer_is_preserved_as_empty_prediction(self):
        self.assertEqual(normalize_generated_answer("  ###  ", "###"), "")
        self.assertEqual(normalize_generated_answer("  answer###  ", "###"), "answer")

        class FakeModel:
            def eval(self):
                return self

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            question_file = root / "questions.json"
            answer_file = root / "answers.json"
            output_file = root / "predictions.jsonl"
            write_json(
                question_file,
                [{"question_id": "empty", "question": "What happens?", "video_name": "clip.mp4"}],
            )
            write_json(answer_file, [{"question_id": "empty", "answer": "A reference answer."}])
            args = argparse.Namespace(
                gt_file_question=question_file,
                gt_file_answers=answer_file,
                max_samples=0,
                num_chunks=1,
                chunk_idx=0,
                output_file=output_file,
                video_dir=root,
                resume=False,
                device="cpu",
                seed=42,
                model_path=root / "model",
                attn_implementation="sdpa",
            )
            with mock.patch(
                "videollava.eval.video.run_inference_video_qa.load_pretrained_model",
                return_value=(object(), FakeModel(), object(), 0),
            ), mock.patch(
                "videollava.eval.video.run_inference_video_qa.resolve_video_path",
                return_value=root / "clip.mp4",
            ), mock.patch(
                "videollava.eval.video.run_inference_video_qa.generate_answer",
                return_value="",
            ):
                self.assertEqual(run_inference(args), 1)

            records = [json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(records, [{"id": "empty", "question": "What happens?", "answer": "A reference answer.", "pred": ""}])
            merged = root / "merged.jsonl"
            self.assertEqual(merge_chunk_files(question_file, [output_file], merged), 1)
            self.assertEqual(json.loads(merged.read_text(encoding="utf-8")) ["pred"], "")


class FakeResponse:
    def __init__(self, content, status_error=None):
        self.content = content
        self.status_error = status_error

    def raise_for_status(self):
        if self.status_error is not None:
            raise self.status_error

    def json(self):
        return {"choices": [{"message": {"content": self.content}}]}


class JudgeTests(unittest.TestCase):
    def setUp(self):
        self.record = {
            "id": "7",
            "question": "What happens?",
            "answer": "A person sits.",
            "pred": "The person sits down.",
        }

    def test_safe_json_parser_accepts_fence_and_rejects_invalid_schema(self):
        self.assertEqual(
            parse_judge_content('```json\n{"pred":"yes","score":5}\n```'),
            {"pred": "yes", "score": 5},
        )
        for invalid in (
            "{'pred': 'yes', 'score': 5}",
            '{"pred":"yes","score":4.5}',
            '{"pred":"maybe","score":3}',
            '{"pred":"yes","score":5,"extra":1}',
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_judge_content(invalid)

    def test_request_retries_malformed_response_then_succeeds(self):
        responses = iter(
            [FakeResponse("not json"), FakeResponse('{"pred":"yes","score":4}')]
        )
        calls = []

        def fake_post(*args, **kwargs):
            calls.append((args, kwargs))
            return next(responses)

        result = request_judgment(
            self.record,
            api_key="not-a-real-key",
            base_url="http://mock.example.invalid/v1",
            model="gpt-4.1-mock",
            timeout=1,
            max_retries=1,
            retry_base_seconds=0,
            rate_limiter=RateLimiter(0),
            post=fake_post,
        )
        self.assertEqual(len(calls), 2)
        request_args, request_kwargs = calls[-1]
        self.assertEqual(request_args[0], "http://mock.example.invalid/v1/chat/completions")
        self.assertEqual(request_kwargs["timeout"], 1)
        self.assertEqual(request_kwargs["json"]["model"], "gpt-4.1-mock")
        self.assertEqual(
            request_kwargs["headers"]["Authorization"], "Bearer not-a-real-key"
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["pred"], "yes")
        self.assertEqual(result["score"], 4)
        self.assertEqual(result["prompt_version"], PROMPT_VERSION)

    def test_permanent_format_error_is_reported_and_aggregated(self):
        result = request_judgment(
            self.record,
            api_key="not-a-real-key",
            base_url="http://mock.example.invalid/v1",
            model="mock",
            timeout=1,
            max_retries=0,
            retry_base_seconds=0,
            rate_limiter=RateLimiter(0),
            post=lambda *args, **kwargs: FakeResponse("bad"),
        )
        self.assertEqual(result["status"], "failed")
        summary = aggregate_judgments([result], total_predictions=1, model="mock")
        self.assertEqual(summary["successful_count"], 0)
        self.assertEqual(summary["failed_count"], 1)
        self.assertFalse(summary["complete"])

    def test_resume_uses_last_duplicate_record_and_compacts_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction_file = root / "predictions.jsonl"
            output_dir = root / "judge"
            output_dir.mkdir()
            write_jsonl(prediction_file, [self.record])
            write_jsonl(
                output_dir / "judgments.jsonl",
                [
                    {
                        "id": "7",
                        "status": "failed",
                        "model": "mock",
                        "prompt_version": PROMPT_VERSION,
                    },
                    {
                        "id": "7",
                        "status": "ok",
                        "pred": "yes",
                        "score": 5,
                        "model": "mock",
                        "prompt_version": PROMPT_VERSION,
                    },
                ],
            )
            existing = _load_existing_judgments(
                output_dir / "judgments.jsonl", {"7"}
            )
            self.assertEqual(existing["7"]["status"], "ok")
            args = argparse.Namespace(
                pred_path=prediction_file,
                output_dir=output_dir,
                summary_path=output_dir / "summary.json",
                model=None,
                workers=1,
                timeout=1.0,
                max_retries=0,
                retry_base_seconds=0.0,
                requests_per_minute=0,
                resume=True,
            )
            environment = {
                "OPENAI_API_KEY": "not-a-real-key",
                "OPENAI_BASE_URL": "http://mock.example.invalid/v1",
                "VIDEO_QA_JUDGE_MODEL": "mock",
            }
            with mock.patch.dict(os.environ, environment, clear=False), mock.patch(
                "videollava.eval.video.eval_video_qa.requests.post",
                side_effect=AssertionError("No network request expected for a completed ID"),
            ):
                summary = evaluate_predictions(args)
            self.assertTrue(summary["complete"])
            compacted = (output_dir / "judgments.jsonl").read_text().splitlines()
            self.assertEqual(len(compacted), 1)
            self.assertEqual(json.loads(compacted[0])["status"], "ok")

    def test_resume_rejects_mixed_judge_models_and_prompt_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "judgments.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "id": "7",
                        "status": "ok",
                        "pred": "yes",
                        "score": 5,
                        "model": "gpt-3.5-mock",
                        "prompt_version": PROMPT_VERSION,
                    }
                ],
            )
            with self.assertRaisesRegex(ValueError, "mixes model"):
                _load_existing_judgments(
                    path,
                    {"7"},
                    expected_model="gpt-4.1-mock",
                    expected_prompt_version=PROMPT_VERSION,
                )


if __name__ == "__main__":
    unittest.main()
