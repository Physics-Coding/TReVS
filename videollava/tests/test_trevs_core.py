import ast
import os
from pathlib import Path
import unittest
from unittest import mock

import torch
from transformers import LlamaConfig

from videollava.model.language_model import trevs_router as video_router
from videollava.model.language_model.modelling_sparse_llama import (
    TrevsLlamaForCausalLM,
    compute_lookahead_text_to_vision_attention,
    get_layer_cache_lengths,
)
from videollava.model.language_model.sparse_videollava_llama import (
    VideoLlavaConfig,
    VideoLlavaTReVSForCausalLM,
)


def make_tiny_model() -> TrevsLlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=None,
        pad_token_id=0,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(17)
    return TrevsLlamaForCausalLM(config).eval()


class RouterReferenceTests(unittest.TestCase):
    def test_historical_method_aliases_are_rejected(self):
        for value in ("rcer", "v3_9", "v3_10", "original_llava", "v1_0"):
            with self.subTest(value=value), mock.patch.dict(
                os.environ, {"METHOD": value}, clear=False
            ), self.assertRaisesRegex(ValueError, "Unsupported METHOD"):
                video_router._resolve_method_name()

    def test_router_matches_llava_reference_fixture_for_all_scoring_modes(self):
        # Fixed reference indices cover every supported Stage-1 scoring mode.
        expected = {
            ("rms", "0"): (
                [[0, 2, 3, 4, 5, 6, 9, 12, 14], [0, 1, 6, 7, 9, 11, 13, 14, 15]],
                [[0, 4, 5, 9, 12, 14], [0, 1, 7, 9, 11, 13]],
                [[2, 6, 3], [14, 15, 6]],
            ),
            ("rms", "1"): (
                [[0, 2, 4, 5, 7, 9, 10, 12, 15], [0, 1, 6, 7, 9, 11, 13, 14, 15]],
                [[4, 5, 7, 9, 12, 15], [0, 1, 7, 9, 11, 13]],
                [[2, 0, 10], [14, 15, 6]],
            ),
            ("max", "0"): (
                [[0, 2, 3, 4, 5, 6, 8, 9, 15], [0, 1, 2, 3, 7, 11, 13, 14, 15]],
                [[0, 4, 5, 8, 9, 15], [0, 1, 2, 7, 11, 13]],
                [[3, 2, 6], [14, 3, 15]],
            ),
            ("max", "1"): (
                [[0, 2, 3, 4, 5, 6, 8, 9, 15], [0, 1, 2, 3, 7, 11, 13, 14, 15]],
                [[0, 4, 5, 8, 9, 15], [0, 1, 2, 7, 11, 13]],
                [[3, 2, 6], [14, 3, 15]],
            ),
            ("mean", "0"): (
                [[0, 2, 3, 4, 5, 6, 9, 12, 14], [0, 1, 6, 7, 9, 11, 13, 14, 15]],
                [[0, 4, 5, 9, 12, 14], [0, 1, 7, 9, 11, 13]],
                [[2, 6, 3], [14, 15, 6]],
            ),
            ("mean", "1"): (
                [[0, 2, 5, 7, 9, 10, 12, 14, 15], [0, 1, 6, 7, 9, 11, 13, 14, 15]],
                [[5, 7, 9, 12, 14, 15], [0, 1, 7, 9, 11, 13]],
                [[2, 0, 10], [14, 15, 6]],
            ),
        }
        torch.manual_seed(3)
        vit_attn = torch.rand(2, 4, 17, 17)
        vision = torch.rand(2, 16, 12)
        semantic_vision = torch.rand(2, 16, 12)
        text = torch.rand(2, 5, 12)
        text_mask = torch.tensor(
            [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool
        )
        for text_mode in ("rms", "max", "mean"):
            for consistency in ("0", "1"):
                environment = {
                    "METHOD": "trevs",
                    "TREVS_TEXT_SCORE_MODE": text_mode,
                    "DOUBLE_TRACK_USE_CONSISTENCY_REWARD": consistency,
                    "TREVS_VISUAL_TEMPERATURE": "0.75",
                    "TREVS_TEXT_TEMPERATURE": "1.25",
                }
                with self.subTest(text_mode=text_mode, consistency=consistency), mock.patch.dict(
                    os.environ, environment, clear=False
                ):
                    video_idx, video_stats = video_router.trevs_route(
                        vit_attn,
                        vision,
                        text,
                        k_track1=6,
                        k_track2=3,
                        core_token_mask=text_mask,
                        V_semantic_proj=semantic_vision,
                        semantic_layer=5,
                        return_stats=True,
                    )
                    expected_idx, expected_track1, expected_track2 = expected[
                        (text_mode, consistency)
                    ]
                    self.assertEqual(video_idx.tolist(), expected_idx)
                    self.assertEqual(video_stats["idx_track1"].tolist(), expected_track1)
                    self.assertEqual(video_stats["idx_track2"].tolist(), expected_track2)
                    gathered = torch.gather(
                        vision,
                        1,
                        video_idx.unsqueeze(-1).expand(-1, -1, vision.shape[-1]),
                    )
                    self.assertTrue(torch.equal(video_stats["V_routed"], gathered))

    def test_phase_scoring_matches_llava_reference_fixture(self):
        expected = {
            "priority_heads": (
                [[0, 6, 7, 10, 11], [0, 1, 2, 6, 8]],
                [[1, 2, 3, 4, 5, 8, 9, 12], [3, 4, 5, 7, 9, 10, 11, 12]],
            ),
            "all_heads": (
                [[0, 3, 6, 7, 11], [0, 5, 6, 8, 11]],
                [[1, 2, 4, 5, 8, 9, 10, 12], [1, 2, 3, 4, 7, 9, 10, 12]],
            ),
        }
        torch.manual_seed(5)
        attention_tv = torch.rand(2, 8, 4, 13)
        for mode in ("priority_heads", "all_heads"):
            with self.subTest(mode=mode):
                video_keep, video_drop = video_router.score_phase_attention(
                    attention_tv, n_keep=5, mode=mode
                )
                expected_keep, expected_drop = expected[mode]
                self.assertEqual(video_keep.tolist(), expected_keep)
                self.assertEqual(video_drop.tolist(), expected_drop)

    def test_sink_pruning_matches_reference_fixture_and_preserves_positions(self):
        for use_sink, expected_length in ((False, 6), (True, 7)):
            with self.subTest(use_sink=use_sink):
                hidden = torch.arange(10 * 4, dtype=torch.float32).reshape(1, 10, 4)
                original_hidden = hidden.clone()
                position_ids = torch.arange(10).unsqueeze(0)
                attention_mask = torch.ones(1, 10, dtype=torch.long)
                arguments = dict(
                    hidden_states=hidden,
                    v_token_start=2,
                    n_vis_current=6,
                    idx_keep_local=torch.tensor([[1, 4]]),
                    idx_drop_local=torch.tensor([[0, 2, 3, 5]]),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_sink=use_sink,
                )
                video_result = video_router.apply_sink_token_pruning(**arguments)
                self.assertEqual(video_result[0].shape[1], expected_length)
                expected_positions = (
                    [0, 1, 2, 3, 6, 8, 9]
                    if use_sink
                    else [0, 1, 3, 6, 8, 9]
                )
                self.assertEqual(video_result[2].tolist(), [expected_positions])
                expected_hidden = original_hidden[:, expected_positions, :]
                if use_sink:
                    expected_hidden = expected_hidden.clone()
                    expected_hidden[:, 2, :] = original_hidden[:, [2, 4, 5, 7], :].mean(dim=1)
                self.assertTrue(torch.equal(video_result[0], expected_hidden))
                self.assertTrue(
                    torch.equal(video_result[1], torch.ones(1, expected_length, dtype=torch.long))
                )

    def test_sink_default_is_off(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(video_router.use_sink_token())


class CoreCacheAndRopeTests(unittest.TestCase):
    def test_lookahead_rope_uses_max_position_plus_one(self):
        model = make_tiny_model()
        layer = model.model.layers[2]
        rotary = layer.self_attn.rotary_emb
        original_forward = rotary.forward
        recorded = []

        def recording_forward(value, seq_len=None):
            recorded.append(seq_len)
            return original_forward(value, seq_len=seq_len)

        hidden = torch.randn(1, 7, 32)
        position_ids = torch.tensor([[0, 1, 2, 5, 6, 8, 11]])
        with mock.patch.object(rotary, "forward", side_effect=recording_forward):
            attention_tv = compute_lookahead_text_to_vision_attention(
                decoder_layer=layer,
                hidden_states=hidden,
                attention_mask=None,
                position_ids=position_ids,
                text_start=4,
                text_end=7,
                vision_start=1,
                vision_length=3,
            )
        self.assertEqual(attention_tv.shape, (1, 4, 3, 3))
        self.assertEqual(recorded, [12])

    def test_transition_cache_and_two_decode_steps(self):
        for sink_enabled, short_prefill in (("0", 6), ("1", 7)):
            environment = {
                "METHOD": "trevs",
                "PHASE_TRANSITION_LAYER": "2",
                "PHASE_TRANSITION_N_KEEP": "2",
                "TREVS_PHASE_SCORING": "priority_heads",
                "TREVS_USE_SINK_TOKEN": sink_enabled,
            }
            with self.subTest(sink=sink_enabled), mock.patch.dict(
                os.environ, environment, clear=False
            ):
                model = make_tiny_model()
                prefill = model(
                    inputs_embeds=torch.randn(1, 10, 32),
                    attention_mask=torch.ones(1, 10, dtype=torch.long),
                    position_ids=torch.arange(10).unsqueeze(0),
                    use_cache=True,
                    image_shape=6,
                    pre_prompt_length_list=[2],
                    token_length_list=[10],
                )
                self.assertEqual(
                    get_layer_cache_lengths(prefill.past_key_values),
                    [10, 10, short_prefill, short_prefill],
                )
                metrics = model.model.last_trevs_metrics
                self.assertEqual(metrics["trevs_phase_scoring_layer_idx"], 2)
                self.assertEqual(metrics["trevs_phase_pruned_after_layer_idx"], 1)

                cache = prefill.past_key_values
                for offset in range(2):
                    decoded = model(
                        input_ids=torch.tensor([[3 + offset]]),
                        attention_mask=torch.ones(1, 11 + offset, dtype=torch.long),
                        position_ids=torch.tensor([[10 + offset]]),
                        past_key_values=cache,
                        use_cache=True,
                        image_shape=6,
                        pre_prompt_length_list=[2],
                        token_length_list=[10],
                    )
                    cache = decoded.past_key_values
                self.assertEqual(
                    get_layer_cache_lengths(cache),
                    [12, 12, short_prefill + 2, short_prefill + 2],
                )

    def test_video_7b_layer_count_has_eight_full_and_twenty_four_short_caches(self):
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=32,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
            pad_token_id=0,
        )
        config._attn_implementation = "sdpa"
        environment = {
            "METHOD": "trevs",
            "PHASE_TRANSITION_LAYER": "8",
            "PHASE_TRANSITION_N_KEEP": "2",
            "TREVS_USE_SINK_TOKEN": "0",
        }
        with mock.patch.dict(os.environ, environment, clear=False):
            model = TrevsLlamaForCausalLM(config).eval()
            output = model(
                inputs_embeds=torch.randn(1, 10, 16),
                attention_mask=torch.ones(1, 10, dtype=torch.long),
                position_ids=torch.arange(10).unsqueeze(0),
                use_cache=True,
                image_shape=6,
                pre_prompt_length_list=[2],
                token_length_list=[10],
            )
        self.assertEqual(
            get_layer_cache_lengths(output.past_key_values),
            [10] * 8 + [6] * 24,
        )

    def test_standard_greedy_loop_propagates_metadata(self):
        environment = {
            "METHOD": "trevs",
            "PHASE_TRANSITION_LAYER": "2",
            "PHASE_TRANSITION_N_KEEP": "2",
            "TREVS_USE_SINK_TOKEN": "0",
        }
        with mock.patch.dict(os.environ, environment, clear=False):
            model = make_tiny_model()
            generated = model.generate(
                inputs_embeds=torch.randn(1, 10, 32),
                attention_mask=torch.ones(1, 10, dtype=torch.long),
                position_ids=torch.arange(10).unsqueeze(0),
                image_shape=6,
                pre_prompt_length_list=[2],
                token_length_list=[10],
                max_new_tokens=2,
                do_sample=False,
                num_beams=1,
                return_dict_in_generate=True,
            )
        self.assertEqual(tuple(generated.sequences.shape), (1, 3))
        self.assertEqual(get_layer_cache_lengths(generated.past_key_values), [11, 11, 7, 7])
        self.assertTrue(model.model.last_trevs_metrics["phase_transition_applied"])

    def test_eager_backend_is_rejected(self):
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
        config._attn_implementation = "eager"
        with self.assertRaisesRegex(ValueError, "supports only SDPA or FlashAttention2"):
            TrevsLlamaForCausalLM(config)

    def test_video_generate_rejects_prompt_plus_decode_context_overflow(self):
        config = VideoLlavaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=8,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        config._attn_implementation = "sdpa"
        model = VideoLlavaTReVSForCausalLM(config).eval()
        with self.assertRaisesRegex(ValueError, "prompt plus requested generation"):
            model.generate(
                inputs=torch.tensor([[1, 3, 4]]),
                max_new_tokens=6,
                do_sample=False,
                num_beams=1,
                temperature=0.0,
            )

    def test_video_language_core_has_no_runtime_llava_or_qwen_imports(self):
        root = Path(__file__).parents[1] / "model" / "language_model"
        for path in root.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [node.module or ""]
                else:
                    continue
                self.assertFalse(
                    any(name == "llava" or name.startswith("llava.") for name in names)
                )
                self.assertFalse(
                    any(name == "qwen" or name.startswith("qwen.") for name in names)
                )


if __name__ == "__main__":
    unittest.main()
