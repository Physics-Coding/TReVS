from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("trevs_reproduce_metadata", REPO_ROOT / "scripts/reproduce.py")
assert SPEC is not None and SPEC.loader is not None
reproduce = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reproduce)


class VideoRunMetadataTests(unittest.TestCase):
    def config(self, run_dir: Path, max_samples: int = 0) -> dict:
        return {"run_dir": str(run_dir), "max_samples": max_samples}

    def write_predictions(self, run_dir: Path, dataset: str = "tgif") -> None:
        path = run_dir / "predictions" / dataset / "predictions.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {"id": "q1", "question": "Question?", "answer": "yes", "pred": "yes"}
            )
            + "\n",
            encoding="utf-8",
        )

    def test_prediction_only_status_is_explicitly_unscored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_predictions(root)
            reproduce.write_video_prediction_status(self.config(root), "tgif")
            metric = json.loads(
                (root / "metrics" / "tgif" / "evaluator.json").read_text(encoding="utf-8")
            )
        self.assertEqual(metric["status"], "prediction_only")
        self.assertFalse(metric["comparable_to_paper"])
        self.assertEqual(metric["samples"], 1)

    def test_complete_judge_summary_becomes_structured_metric(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path = root / "metrics" / "tgif" / "judge" / "family" / "model" / "summary.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(
                json.dumps(
                    {
                        "accuracy": 0.75,
                        "average_score": 4.25,
                        "successful_count": 4,
                        "failed_count": 0,
                        "total_predictions": 4,
                        "complete": True,
                        "judge_model": "external-model",
                        "prompt_version": "judge-v1",
                    }
                ),
                encoding="utf-8",
            )
            reproduce.write_video_judge_metric(self.config(root, max_samples=4), "tgif")
            metric = json.loads(
                (root / "metrics" / "tgif" / "evaluator.json").read_text(encoding="utf-8")
            )
        self.assertEqual(metric["status"], "scored")
        self.assertEqual(metric["run_type"], "smoke")
        self.assertEqual(metric["accuracy_percent"], 75.0)
        self.assertFalse(metric["comparable_to_paper"])

    def test_incomplete_judge_summary_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path = root / "metrics" / "tgif" / "judge" / "family" / "model" / "summary.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(json.dumps({"complete": False}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "incomplete or malformed"):
                reproduce.write_video_judge_metric(self.config(root), "tgif")


if __name__ == "__main__":
    unittest.main()
