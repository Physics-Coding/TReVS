from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llava.model.builder import _validate_supported_checkpoint
from llava.model.language_model.llava_llama import LlavaConfig as DenseConfig
from llava.model.language_model.score import _normalize_method_name
from llava.model.language_model.sparse_llava_llama import LlavaConfig as SparseConfig


class LLaVACheckpointContractTests(unittest.TestCase):
    def make_checkpoint(self, root: Path, layout: str) -> Path:
        root.mkdir()
        config = {
            "model_type": "llava",
            "architectures": ["LlavaLlamaForCausalLM"],
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "vocab_size": 128,
            "mm_vision_tower": "openai/clip-vit-large-patch14-336",
        }
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        if layout == "bin":
            index_name = "pytorch_model.bin.index.json"
            shard_name = "pytorch_model-00001-of-00001.bin"
        else:
            index_name = "model.safetensors.index.json"
            shard_name = "model-00001-of-00001.safetensors"
        (root / shard_name).write_bytes(b"synthetic shard")
        (root / index_name).write_text(
            json.dumps({"weight_map": {"model.embed_tokens.weight": shard_name}}),
            encoding="utf-8",
        )
        return root

    def test_official_llava_model_type_accepts_bin_and_safetensors_layouts(self) -> None:
        for layout in ("bin", "safetensors"):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as directory:
                checkpoint = self.make_checkpoint(Path(directory) / "checkpoint", layout)
                raw = _validate_supported_checkpoint(str(checkpoint), None)
                self.assertEqual(raw["model_type"], "llava")
                config_values = dict(raw)
                config_values.pop("model_type")
                config = DenseConfig.from_dict(config_values)
                self.assertEqual(config.model_type, "llava_llama")

    def test_dense_and_sparse_share_one_config_class(self) -> None:
        self.assertIs(DenseConfig, SparseConfig)

    def test_projector_only_base_merge_and_unsupported_architecture_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.make_checkpoint(Path(directory) / "checkpoint", "bin")
            with self.assertRaisesRegex(ValueError, "complete LLaVA checkpoints"):
                _validate_supported_checkpoint(str(checkpoint), "base-model")
            (checkpoint / "pytorch_model-00001-of-00001.bin").unlink()
            with self.assertRaisesRegex(ValueError, "missing 1 indexed weight"):
                _validate_supported_checkpoint(str(checkpoint), None)
            payload = json.loads((checkpoint / "config.json").read_text())
            payload["architectures"] = ["LlavaMistralForCausalLM"]
            (checkpoint / "config.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported checkpoint architecture"):
                _validate_supported_checkpoint(str(checkpoint), None)

    def test_historical_method_aliases_are_rejected(self) -> None:
        self.assertEqual(_normalize_method_name(""), "trevs")
        self.assertEqual(_normalize_method_name("trevs"), "trevs")
        self.assertEqual(_normalize_method_name("dense"), "dense")
        for value in ("rcer", "v3_9", "v3_10", "original_llava", "v1_0"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Unsupported METHOD"):
                _normalize_method_name(value)


if __name__ == "__main__":
    unittest.main()
