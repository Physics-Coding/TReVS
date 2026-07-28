from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPRODUCE = REPO_ROOT / "scripts" / "reproduce.py"


class PresetContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = json.loads(
            (REPO_ROOT / "configs" / "presets.json").read_text(encoding="utf-8")
        )["families"]

    def test_named_stage_budgets_are_exact(self) -> None:
        expected = {
            ("llava15", "32"): (48, 16, 64, 21, 31.75),
            ("llava15", "64"): (96, 32, 128, 42, 63.5),
            ("llava15", "128"): (192, 64, 256, 85, 127.75),
            ("llava_next", "160"): (48, 16, 320, 106, 159.5),
            ("llava_next", "320"): (96, 32, 640, 213, 319.75),
            ("llava_next", "640"): (192, 64, 1280, 426, 639.5),
            ("qwen25vl", "142"): (204, 68, 272, 90, 142.0),
            ("qwen25vl", "284"): (408, 136, 544, 180, 284.0),
            ("qwen25vl", "426"): (612, 204, 816, 270, 426.0),
            ("videollava", "136"): (26, 8, 272, 90, 135.5),
            ("videollava", "960"): (180, 60, 1920, 640, 960.0),
        }
        for (family, preset), values in expected.items():
            with self.subTest(family=family, preset=preset):
                config = self.registry[family]["presets"][preset]
                self.assertEqual(
                    (
                        config["route_topk"],
                        config["route_fps"],
                        config["stage1_visual_tokens"],
                        config["phase_transition_keep"],
                        config["average_visual_tokens"],
                    ),
                    values,
                )

    def test_weighted_averages_recompute_from_layers(self) -> None:
        for family_name, family in self.registry.items():
            layers = family["llm_layers"]
            for preset_name, preset in family["presets"].items():
                if preset_name == "custom" or preset["method"] == "dense":
                    continue
                phase = preset["phase_transition_layer"]
                average = (
                    phase * preset["stage1_visual_tokens"]
                    + (layers - phase) * preset["phase_transition_keep"]
                ) / layers
                with self.subTest(family=family_name, preset=preset_name):
                    self.assertAlmostEqual(average, preset["average_visual_tokens"])

    def test_qwen_trevs_presets_are_sdpa(self) -> None:
        qwen = self.registry["qwen25vl"]
        self.assertEqual(qwen["attention_backend"], "sdpa")
        self.assertEqual(qwen["trevs"]["visual_temperature"], 1.4)


class LauncherContractTests(unittest.TestCase):
    def dry_run(self, family: str, preset: str, *extra: str) -> dict:
        command = [
            "python3",
            str(REPRODUCE),
            "--family",
            family,
            "--preset",
            preset,
            "--model-path",
            "/models/model",
            "--data-root",
            "/datasets",
            "--dry-run",
            *extra,
        ]
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        return json.loads(result.stdout)

    def test_every_public_named_preset_dry_runs(self) -> None:
        presets = {
            "llava15": ("32", "64", "128"),
            "llava_next": ("160", "320", "640"),
            "qwen25vl": ("142", "284", "426", "dense"),
            "videollava": ("136", "960", "dense"),
        }
        for family, family_presets in presets.items():
            for preset in family_presets:
                with self.subTest(family=family, preset=preset):
                    payload = self.dry_run(family, preset)
                    self.assertEqual(payload["mode"], "dry_run")
                    self.assertEqual(payload["config"]["family"], family)
                    self.assertTrue(payload["commands"])
                    for command in payload["commands"]:
                        self.assertTrue(Path(command[1]).is_file())

    def test_qwen_dense_records_native_hf_backend(self) -> None:
        dense = self.dry_run("qwen25vl", "dense")["config"]
        trevs = self.dry_run("qwen25vl", "142")["config"]
        self.assertEqual(dense["attention_backend"], "native_hf")
        self.assertEqual(trevs["attention_backend"], "sdpa")

    def test_custom_video_preset_records_exact_average(self) -> None:
        payload = self.dry_run(
            "videollava",
            "custom",
            "--stage1-topk",
            "30",
            "--stage1-fps",
            "10",
            "--stage2-keep",
            "80",
            "--phase-layer",
            "4",
        )
        budget = payload["config"]["budget"]
        self.assertEqual(budget["stage1_visual_tokens"], 320)
        self.assertEqual(budget["average_visual_tokens"], 110.0)

    def test_dry_run_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not-created"
            self.dry_run("llava15", "64", "--output-root", str(output))
            self.assertFalse(output.exists())

    def test_output_below_data_root_is_rejected(self) -> None:
        result = subprocess.run(
            [
                "python3",
                str(REPRODUCE),
                "--family",
                "llava15",
                "--preset",
                "64",
                "--model-path",
                "/models/model",
                "--data-root",
                "/datasets",
                "--output-root",
                "/datasets/results",
                "--dry-run",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("protected path", result.stderr)


if __name__ == "__main__":
    unittest.main()
