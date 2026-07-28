import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from videollava.eval.video.datasets import get_dataset_paths, resolve_video_path


REPO_ROOT = Path(__file__).parents[2]
CHECKPOINT_VALUE = os.environ.get("TREVS_TEST_VIDEOLLAVA_CHECKPOINT", "").strip()
DATA_ROOT_VALUE = os.environ.get("TREVS_TEST_VIDEO_DATA_ROOT", "").strip()
CHECKPOINT = Path(CHECKPOINT_VALUE).expanduser() if CHECKPOINT_VALUE else None
DATA_ROOT = Path(DATA_ROOT_VALUE).expanduser() if DATA_ROOT_VALUE else None


class ImportIsolationTests(unittest.TestCase):
    def test_package_sources_have_no_llava_or_qwen_imports(self):
        package_root = REPO_ROOT / "videollava"
        for path in package_root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    imported = [node.module or ""]
                else:
                    continue
                self.assertFalse(
                    any(
                        name in {"llava", "qwen"}
                        or name.startswith("llava.")
                        or name.startswith("qwen.")
                        for name in imported
                    ),
                    f"cross-package import in {path.relative_to(REPO_ROOT)}: {imported}",
                )

    def test_top_level_import_has_no_registry_or_cross_package_side_effects(self):
        program = r'''
import json
import sys
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
before = dict(CONFIG_MAPPING._extra_content)
import videollava
after_top = dict(CONFIG_MAPPING._extra_content)
from videollava.model import builder
after_builder = dict(CONFIG_MAPPING._extra_content)
forbidden = sorted(
    name for name in sys.modules
    if name == "llava" or name.startswith("llava.")
    or name == "qwen" or name.startswith("qwen.")
    or name == "visionzip" or name.startswith("visionzip.")
)
print(json.dumps({
    "top_unchanged": before == after_top,
    "builder_unchanged": before == after_builder,
    "forbidden": forbidden,
}))
'''
        result = subprocess.run(
            [sys.executable, "-c", program],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertTrue(payload["top_unchanged"])
        self.assertTrue(payload["builder_unchanged"])
        self.assertEqual(payload["forbidden"], [])

    def test_no_optional_training_or_competing_method_imports(self):
        program = r'''
import json
import sys
from videollava.model.builder import load_pretrained_model
loaded = sorted(
    name for name in sys.modules
    if name == "peft" or name.startswith("peft.")
    or name == "pytorchvideo" or name.startswith("pytorchvideo.")
    or name == "visionzip" or name.startswith("visionzip.")
)
print(json.dumps(loaded))
'''
        result = subprocess.run(
            [sys.executable, "-c", program],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertEqual(json.loads(result.stdout.strip().splitlines()[-1]), [])


class LanguageBindShapeTests(unittest.TestCase):
    def test_hidden_states_and_spatial_attention_share_frame_batch_order(self):
        from videollava.model.multimodal_encoder.languagebind.video.configuration_video import (
            CLIPVisionConfig,
        )
        from videollava.model.multimodal_encoder.languagebind.video.modeling_video import (
            CLIPVisionTransformer,
        )

        config = CLIPVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            projection_dim=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            image_size=28,
            patch_size=14,
            hidden_act="gelu",
            add_time_attn=True,
            num_frames=8,
        )
        tower = CLIPVisionTransformer(config).eval()
        output = tower(
            torch.randn(1, 3, 8, 28, 28),
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )
        self.assertEqual(tuple(output.hidden_states[-2].shape), (1, 8, 5, 16))
        self.assertEqual(tuple(output.attentions[-2].shape), (8, 2, 5, 5))


@unittest.skipUnless(
    CHECKPOINT is not None and CHECKPOINT.is_dir(),
    "set TREVS_TEST_VIDEOLLAVA_CHECKPOINT to enable the checkpoint coverage test",
)
class CheckpointCoverageTests(unittest.TestCase):
    def test_meta_model_is_fully_covered_except_intentional_image_and_rope_keys(self):
        from videollava.model.language_model.sparse_videollava_llama import (
            VideoLlavaConfig,
            VideoLlavaTReVSForCausalLM,
        )

        config_values = json.loads((CHECKPOINT / "config.json").read_text())
        config_values.pop("model_type", None)
        config_values.pop("architectures", None)
        config = VideoLlavaConfig(**config_values)
        config._attn_implementation = "sdpa"
        with torch.device("meta"):
            model = VideoLlavaTReVSForCausalLM(config)

        weight_map = json.loads(
            (CHECKPOINT / "model.safetensors.index.json").read_text()
        )["weight_map"]
        checkpoint_keys = set(weight_map)
        model_keys = set(model.state_dict())
        self.assertEqual(model_keys - checkpoint_keys, set())
        unexpected = checkpoint_keys - model_keys
        disallowed = {
            key
            for key in unexpected
            if not key.startswith("model.image_tower.")
            and not key.endswith(".self_attn.rotary_emb.inv_freq")
        }
        self.assertEqual(disallowed, set())

        tower_checkpoint_keys = {
            key.removeprefix("model.video_tower.")
            for key in checkpoint_keys
            if key.startswith("model.video_tower.")
        }
        self.assertEqual(tower_checkpoint_keys, set(model.get_video_tower().state_dict()))
        self.assertEqual(len(tower_checkpoint_keys), 655)
        self.assertEqual(model.get_video_tower().config.hidden_act, "gelu")
        self.assertEqual(model.get_video_tower().config.num_frames, 8)
        self.assertFalse(hasattr(model.get_model(), "image_tower"))


@unittest.skipUnless(
    DATA_ROOT is not None and DATA_ROOT.is_dir(),
    "set TREVS_TEST_VIDEO_DATA_ROOT to enable the real-video decode test",
)
class DeterministicDecodeTests(unittest.TestCase):
    def test_one_video_from_each_dataset_is_deterministic(self):
        from videollava.model.multimodal_encoder.languagebind.processing_video import (
            build_default_video_processor,
        )

        processor = build_default_video_processor(8)
        for dataset in ("tgif", "msvd", "msrvtt"):
            with self.subTest(dataset=dataset):
                video_dir, question_file, _ = get_dataset_paths(DATA_ROOT, dataset)
                questions = json.loads(question_file.read_text())
                sample = resolve_video_path(video_dir, str(questions[0]["video_name"]))
                first = processor.preprocess(sample)["pixel_values"]
                second = processor.preprocess(sample)["pixel_values"]
                self.assertEqual(tuple(first.shape), (1, 3, 8, 224, 224))
                self.assertTrue(torch.equal(first, second))


class ScriptContractTests(unittest.TestCase):
    _PRESET_ENV_KEYS = (
        "VIDEOLLAVA_TREVS_PRESET",
        "METHOD",
        "TREVS_ROUTE_TOPK",
        "TREVS_ROUTE_FPS",
        "TREVS_PHASE_SCORING",
        "TREVS_USE_SINK_TOKEN",
        "PHASE_TRANSITION_LAYER",
        "PHASE_TRANSITION_N_KEEP",
        "EXPERIMENT",
    )

    def _source_trevs_env(self, overrides):
        environment = dict(os.environ)
        for name in self._PRESET_ENV_KEYS:
            environment.pop(name, None)
        environment.update(overrides)
        names = (
            "VIDEOLLAVA_TREVS_PRESET",
            "METHOD",
            "TREVS_ROUTE_TOPK",
            "TREVS_ROUTE_FPS",
            "TREVS_PHASE_SCORING",
            "TREVS_USE_SINK_TOKEN",
            "PHASE_TRANSITION_LAYER",
            "PHASE_TRANSITION_N_KEEP",
            "STAGE1_TOPK_VISUAL_TOKENS",
            "STAGE1_FPS_VISUAL_TOKENS",
            "STAGE1_VISUAL_TOKENS",
            "STAGE2_VISUAL_TOKENS",
            "LLM_NUM_LAYERS",
            "AVERAGE_VISUAL_TOKENS",
            "AVERAGE_VISUAL_TOKENS_NUMERATOR",
            "EXPERIMENT",
        )
        command = "source scripts/videollava/trevs_env.sh; " + "; ".join(
            f"printf '%s\\n' \"${name}\"" for name in names
        )
        result = subprocess.run(
            ["bash", "-c", command],
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=True,
        )
        return dict(zip(names, result.stdout.splitlines(), strict=True))

    def test_all_scripts_parse_and_default_config_is_observable(self):
        script_dir = REPO_ROOT / "scripts" / "videollava"
        scripts = sorted(str(path) for path in script_dir.glob("*.sh"))
        subprocess.run(["bash", "-n", *scripts], cwd=REPO_ROOT, check=True)
        with tempfile.TemporaryDirectory() as directory:
            environment = dict(os.environ)
            environment.update(
                {
                    "RESULT_ROOT": directory,
                    "CUDA_VISIBLE_DEVICES": "2,5",
                    "VIDEO_QA_JUDGE_MODEL": "gpt-4.1-mock",
                }
            )
            command = (
                "source scripts/videollava/trevs_env.sh; "
                "write_videollava_run_config; "
                "printf '%s' \"$RUN_DIR/run_config.env\""
            )
            result = subprocess.run(
                ["bash", "-c", command],
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=True,
            )
            config_path = Path(result.stdout)
            config_text = config_path.read_text()
            self.assertIn("VIDEOLLAVA_TREVS_PRESET=960", config_text)
            self.assertIn("STAGE1_VISUAL_TOKENS=1920", config_text)
            self.assertIn("STAGE2_VISUAL_TOKENS=640", config_text)
            self.assertIn("LLM_NUM_LAYERS=32", config_text)
            self.assertIn("AVERAGE_VISUAL_TOKENS=960", config_text)
            self.assertIn("AVERAGE_VISUAL_TOKENS_NUMERATOR=30720", config_text)
            self.assertIn("TREVS_USE_SINK_TOKEN=0", config_text)
            self.assertIn("TREVS_PHASE_SCORING=priority_heads", config_text)
            self.assertIn("ATTN_IMPLEMENTATION=sdpa", config_text)
            self.assertIn("GENERATION_DO_SAMPLE=0", config_text)
            self.assertIn("GPU_LOGICAL_MAPPING=2-", config_text)
            self.assertNotIn("OPENAI_API_KEY", config_text)
            self.assertNotIn("OPENAI_BASE_URL", config_text)

    def test_videoqa_runner_resolves_direct_and_packaged_data_layouts(self):
        for layout in ("direct", "packaged"):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                data_root = root / "data"
                if layout == "direct":
                    dataset_root = data_root / "TGIF_Zero_Shot_QA"
                else:
                    dataset_root = (
                        data_root / "GPT_Zero_Shot_QA" / "TGIF_Zero_Shot_QA"
                    )
                (dataset_root / "mp4").mkdir(parents=True)
                (dataset_root / "test_q.json").write_text("[]\n", encoding="utf-8")
                (dataset_root / "test_a.json").write_text("[]\n", encoding="utf-8")
                model_path = root / "model"
                model_path.mkdir()

                call_log = root / "python_calls.txt"
                fake_python = root / "fake_python.sh"
                fake_python.write_text(
                    "#!/usr/bin/env bash\n"
                    "printf '%s\\t' \"$@\" >> \"${FAKE_PYTHON_LOG}\"\n"
                    "printf '\\n' >> \"${FAKE_PYTHON_LOG}\"\n",
                    encoding="utf-8",
                )
                fake_python.chmod(0o755)

                environment = dict(os.environ)
                for name in self._PRESET_ENV_KEYS:
                    environment.pop(name, None)
                environment.update(
                    {
                        "VIDEOLLAVA_TREVS_PRESET": "136",
                        "MODEL_PATH": str(model_path),
                        "DATA_ROOT": str(data_root),
                        "RESULT_ROOT": str(root / "results"),
                        "RUN_DIR": str(root / "run"),
                        "CUDA_VISIBLE_DEVICES": "0",
                        "PYTHON_BIN": str(fake_python),
                        "FAKE_PYTHON_LOG": str(call_log),
                        "MAX_SAMPLES": "1",
                    }
                )
                subprocess.run(
                    [
                        "bash",
                        str(
                            REPO_ROOT
                            / "scripts"
                            / "videollava"
                            / "run_videoqa_multi_gpu.sh"
                        ),
                        "tgif",
                    ],
                    cwd=REPO_ROOT,
                    env=environment,
                    text=True,
                    capture_output=True,
                    check=True,
                )
                calls = call_log.read_text(encoding="utf-8")
                self.assertIn("videollava.eval.video.preflight", calls)
                self.assertIn("videollava.eval.video.run_inference_video_qa", calls)
                self.assertIn("videollava.eval.video.merge_chunks", calls)
                self.assertIn(str(dataset_root / "mp4"), calls)
                self.assertIn(str(dataset_root / "test_q.json"), calls)
                self.assertIn(str(dataset_root / "test_a.json"), calls)

    def test_named_presets_resolve_expected_budgets_and_averages(self):
        expected = {
            "960": {
                "TREVS_ROUTE_TOPK": "180",
                "TREVS_ROUTE_FPS": "60",
                "STAGE1_TOPK_VISUAL_TOKENS": "1440",
                "STAGE1_FPS_VISUAL_TOKENS": "480",
                "STAGE1_VISUAL_TOKENS": "1920",
                "STAGE2_VISUAL_TOKENS": "640",
                "PHASE_TRANSITION_LAYER": "8",
                "PHASE_TRANSITION_N_KEEP": "640",
                "AVERAGE_VISUAL_TOKENS": "960",
            },
            "136": {
                "TREVS_ROUTE_TOPK": "26",
                "TREVS_ROUTE_FPS": "8",
                "STAGE1_TOPK_VISUAL_TOKENS": "208",
                "STAGE1_FPS_VISUAL_TOKENS": "64",
                "STAGE1_VISUAL_TOKENS": "272",
                "STAGE2_VISUAL_TOKENS": "90",
                "PHASE_TRANSITION_LAYER": "8",
                "PHASE_TRANSITION_N_KEEP": "90",
                "AVERAGE_VISUAL_TOKENS": "135.5",
            },
        }
        for preset, expected_values in expected.items():
            with self.subTest(preset=preset):
                values = self._source_trevs_env(
                    {
                        "VIDEOLLAVA_TREVS_PRESET": preset,
                        # Named presets must override stale low-level settings.
                        "TREVS_ROUTE_TOPK": "1",
                        "TREVS_USE_SINK_TOKEN": "1",
                    }
                )
                for name, expected_value in expected_values.items():
                    self.assertEqual(values[name], expected_value)
                self.assertEqual(values["METHOD"], "trevs")
                self.assertEqual(values["TREVS_PHASE_SCORING"], "priority_heads")
                self.assertEqual(values["TREVS_USE_SINK_TOKEN"], "0")
                self.assertEqual(values["LLM_NUM_LAYERS"], "32")
                stage1 = int(values["STAGE1_VISUAL_TOKENS"])
                stage2 = int(values["STAGE2_VISUAL_TOKENS"])
                if preset == "960":
                    self.assertEqual(
                        int(values["TREVS_ROUTE_TOPK"]),
                        3 * int(values["TREVS_ROUTE_FPS"]),
                    )
                    self.assertEqual(stage1, 3 * stage2)
                else:
                    self.assertEqual(
                        (int(values["TREVS_ROUTE_TOPK"]), int(values["TREVS_ROUTE_FPS"])),
                        (26, 8),
                    )
                    self.assertLess(abs(stage1 / stage2 - 3.0), 0.03)
                weighted_total = (
                    int(values["PHASE_TRANSITION_LAYER"])
                    * int(values["STAGE1_VISUAL_TOKENS"])
                    + (
                        int(values["LLM_NUM_LAYERS"])
                        - int(values["PHASE_TRANSITION_LAYER"])
                    )
                    * int(values["STAGE2_VISUAL_TOKENS"])
                )
                self.assertAlmostEqual(
                    weighted_total / int(values["LLM_NUM_LAYERS"]),
                    float(values["AVERAGE_VISUAL_TOKENS"]),
                )
                expected_average_tag = "960" if preset == "960" else "135p5"
                self.assertIn(
                    f"preset-{preset}_avg{expected_average_tag}", values["EXPERIMENT"]
                )

    def test_custom_preset_preserves_manual_controls(self):
        values = self._source_trevs_env(
            {
                "VIDEOLLAVA_TREVS_PRESET": "custom",
                "METHOD": "trevs",
                "TREVS_ROUTE_TOPK": "30",
                "TREVS_ROUTE_FPS": "10",
                "TREVS_PHASE_SCORING": "all_heads",
                "TREVS_USE_SINK_TOKEN": "1",
                "PHASE_TRANSITION_LAYER": "4",
                "PHASE_TRANSITION_N_KEEP": "80",
            }
        )
        self.assertEqual(values["TREVS_ROUTE_TOPK"], "30")
        self.assertEqual(values["TREVS_ROUTE_FPS"], "10")
        self.assertEqual(values["TREVS_PHASE_SCORING"], "all_heads")
        self.assertEqual(values["TREVS_USE_SINK_TOKEN"], "1")
        self.assertEqual(values["PHASE_TRANSITION_LAYER"], "4")
        self.assertEqual(values["PHASE_TRANSITION_N_KEEP"], "80")
        self.assertEqual(values["STAGE1_VISUAL_TOKENS"], "320")
        self.assertEqual(values["STAGE2_VISUAL_TOKENS"], "81")
        self.assertEqual(values["AVERAGE_VISUAL_TOKENS"], "110.875")
        self.assertIn("preset-custom_avg110p875", values["EXPERIMENT"])

    def test_unknown_preset_is_rejected(self):
        environment = dict(os.environ)
        environment["VIDEOLLAVA_TREVS_PRESET"] = "unknown"
        result = subprocess.run(
            ["bash", "-c", "source scripts/videollava/trevs_env.sh"],
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("use 136, 960, dense, or custom", result.stderr)

    def test_custom_preset_rejects_unattainable_stage_two_budget(self):
        environment = dict(os.environ)
        environment.update(
            {
                "VIDEOLLAVA_TREVS_PRESET": "custom",
                "TREVS_ROUTE_TOPK": "6",
                "TREVS_ROUTE_FPS": "2",
                "PHASE_TRANSITION_N_KEEP": "64",
            }
        )
        result = subprocess.run(
            ["bash", "-c", "source scripts/videollava/trevs_env.sh"],
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("must be smaller than the Stage-1", result.stderr)

    def test_scorer_contains_no_embedded_credential(self):
        scorer = (
            REPO_ROOT / "videollava" / "eval" / "video" / "eval_video_qa.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("sk-", scorer)
        self.assertNotIn("api-key", scorer.lower())


if __name__ == "__main__":
    unittest.main()
