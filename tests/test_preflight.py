from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("trevs_reproduce", REPO_ROOT / "scripts/reproduce.py")
assert SPEC is not None and SPEC.loader is not None
reproduce = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reproduce)


class RecordContractTests(unittest.TestCase):
    def test_mmbench_release_contracts_use_logical_tsv_record_counts(self) -> None:
        expected = {
            "mmbench/mmbench_dev_20230712.tsv": 4377,
            "mmbench/mmbench_dev_cn_20231003.tsv": 4329,
        }
        for relative_path, record_count in expected.items():
            with self.subTest(path=relative_path):
                contract = reproduce.FILE_CONTRACTS[relative_path]
                self.assertEqual(contract["container"], "tsv")
                self.assertEqual(contract["expected_records"], record_count)

    def test_jsonl_count_unique_ids_and_corruption_are_strict(self) -> None:
        contract = {
            "container": "jsonl",
            "key_fields": ["question_id"],
            "expected_records": 2,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "questions.jsonl"
            path.write_text(
                json.dumps({"question_id": 1}) + "\n" + json.dumps({"question_id": 2}) + "\n",
                encoding="utf-8",
            )
            summary = reproduce.inspect_record_file(path, contract)
            self.assertTrue(summary["valid"])
            self.assertEqual(summary["unique_ids"], 2)
            path.write_text(
                json.dumps({"question_id": 1}) + "\n" + json.dumps({"question_id": 1}) + "\n{bad\n",
                encoding="utf-8",
            )
            summary = reproduce.inspect_record_file(path, contract)
            self.assertFalse(summary["valid"])
            self.assertEqual(summary["duplicate_ids"], 1)
            self.assertTrue(any("malformed JSON" in error for error in summary["errors"]))

    def test_tsv_empty_or_duplicate_index_fails(self) -> None:
        contract = {"container": "tsv", "key_fields": ["index"], "expected_records": 2}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "questions.tsv"
            path.write_text("index\tquestion\n1\ta\n1\tb\n", encoding="utf-8")
            summary = reproduce.inspect_record_file(path, contract)
            self.assertFalse(summary["valid"])
            self.assertEqual(summary["duplicate_ids"], 1)

    def test_repeated_textvqa_image_id_is_valid_for_distinct_questions(self) -> None:
        contract = {
            "container": "jsonl",
            "key_fields": ["question_id", "text"],
            "expected_records": 2,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "questions.jsonl"
            path.write_text(
                json.dumps({"question_id": "image-1", "text": "First question?"}) + "\n"
                + json.dumps({"question_id": "image-1", "text": "Second question?"})
                + "\n",
                encoding="utf-8",
            )
            summary = reproduce.inspect_record_file(path, contract)
        self.assertTrue(summary["valid"], summary["errors"])
        self.assertEqual(summary["unique_ids"], 2)


class CheckpointContractTests(unittest.TestCase):
    def write_index(self, root: Path, name: str = "model.safetensors.index.json") -> None:
        shard = "model-00001-of-00001.safetensors"
        (root / shard).write_bytes(b"synthetic")
        (root / name).write_text(
            json.dumps({"weight_map": {"model.embed_tokens.weight": shard}}),
            encoding="utf-8",
        )

    def test_llava_config_uses_official_raw_model_type(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "model_type": "llava",
                "architectures": ["LlavaLlamaForCausalLM"],
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "mm_vision_tower": "upstream/vision-tower",
            }
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (root / "tokenizer.model").write_bytes(b"synthetic")
            (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            self.write_index(root, "pytorch_model.bin.index.json")
            fake_transformers = types.ModuleType("transformers")

            class FakeCLIPVisionConfig:
                @staticmethod
                def from_pretrained(name: str, local_files_only: bool = False) -> object:
                    if name != "upstream/vision-tower" or not local_files_only:
                        raise ValueError("unexpected lookup")
                    return object()

            fake_transformers.CLIPVisionConfig = FakeCLIPVisionConfig
            with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
                report = reproduce.validate_llava_checkpoint(root, "llava15")
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["weights"]["layout"], "pytorch_model.bin.index.json")

    def test_llava_checkpoint_reports_missing_offline_vision_tower_actionably(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "model_type": "llava",
                "architectures": ["LlavaLlamaForCausalLM"],
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "mm_vision_tower": "upstream/missing-tower",
            }
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (root / "tokenizer.model").write_bytes(b"synthetic")
            (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            self.write_index(root)
            fake_transformers = types.ModuleType("transformers")

            class MissingCLIPVisionConfig:
                @staticmethod
                def from_pretrained(name: str, local_files_only: bool = False) -> object:
                    raise OSError("not cached")

            fake_transformers.CLIPVisionConfig = MissingCLIPVisionConfig
            with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
                report = reproduce.validate_llava_checkpoint(root, "llava15")
        self.assertFalse(report["valid"])
        self.assertIn("obtain the tower declared by config.json before preflight", report["errors"][0])

    def test_qwen_grid_and_checkpoint_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "model_type": "qwen2_5_vl",
                "architectures": ["Qwen2_5_VLForConditionalGeneration"],
                "hidden_size": 3584,
                "num_hidden_layers": 28,
                "num_attention_heads": 28,
                "num_key_value_heads": 4,
                "vision_config": {"patch_size": 14, "spatial_merge_size": 2},
            }
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (root / "preprocessor_config.json").write_text("{}", encoding="utf-8")
            self.write_index(root)
            report = reproduce.validate_qwen_checkpoint(root)
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["image_grid_contract"]["image_grid_thw"], [1, 64, 80])
        self.assertEqual(report["image_grid_contract"]["merged_visual_tokens"], 1280)


class GpuProbeTests(unittest.TestCase):
    def test_requested_physical_gpu_is_probed_in_isolated_visibility_namespace(self) -> None:
        payload = {
            "valid": True,
            "torch_cuda_version": "12.1",
            "visible_device_count": 1,
            "name": "Synthetic GPU",
            "compute_capability": [8, 9],
            "total_memory_bytes": 10,
            "free_memory_bytes": 9,
            "allocation_test": True,
        }
        result = subprocess.CompletedProcess(["python"], 0, json.dumps(payload), "")
        with mock.patch.object(reproduce.subprocess, "run", return_value=result) as run:
            report = reproduce.torch_gpu_report(["3"])
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["devices"][0]["index"], 3)
        self.assertEqual(run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "3")
        self.assertIn("-c", run.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
