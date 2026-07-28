import os
import unittest
from unittest.mock import patch

os.environ["METHOD"] = "trevs"

import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import _prepare_4d_causal_attention_mask

from llava.eval.efficiency_utils import estimate_kv_cache_bytes
from llava.model.language_model.modelling_sparse_llama import (
    LlamaDynamicvitDecoderLayer,
    LlamaDynamicvitModel,
    apply_rotary_pos_emb,
    compute_lookahead_text_to_vision_attention,
    compute_manual_attention_probs,
)


class _DummyCudaEvent:
    def record(self):
        pass

    def elapsed_time(self, _other):
        return 0.0


class LlamaLookaheadPruningTest(unittest.TestCase):
    def _attention_config(self, attn_implementation, num_key_value_heads):
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=64,
        )
        config._attn_implementation = attn_implementation
        return config

    def test_lookahead_text_slice_matches_full_manual_attention(self):
        for attn_implementation in ("sdpa", "flash_attention_2"):
            for num_key_value_heads in (4, 2):
                for mask_kind in ("none", "2d", "4d"):
                    with self.subTest(
                        attn_implementation=attn_implementation,
                        num_key_value_heads=num_key_value_heads,
                        mask_kind=mask_kind,
                    ):
                        torch.manual_seed(7)
                        config = self._attention_config(attn_implementation, num_key_value_heads)
                        layer = LlamaDynamicvitDecoderLayer(config, layer_idx=1).eval()
                        hidden_states = torch.randn(1, 11, config.hidden_size)
                        position_ids = torch.arange(11).unsqueeze(0)
                        padding_mask = torch.ones(1, 11, dtype=torch.long)
                        padding_mask[:, -1] = 0
                        if mask_kind == "4d":
                            attention_mask = _prepare_4d_causal_attention_mask(
                                padding_mask,
                                (1, 11),
                                hidden_states,
                                0,
                            )
                        elif mask_kind == "2d":
                            attention_mask = padding_mask
                        else:
                            attention_mask = None

                        actual = compute_lookahead_text_to_vision_attention(
                            decoder_layer=layer,
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            text_start=7,
                            text_end=10,
                            vision_start=2,
                            vision_length=4,
                        )

                        normalized_states = layer.input_layernorm(hidden_states)
                        self_attn = layer.self_attn
                        query_states = self_attn.q_proj(normalized_states).view(1, 11, 4, 8).transpose(1, 2)
                        key_states = self_attn.k_proj(normalized_states).view(
                            1,
                            11,
                            num_key_value_heads,
                            8,
                        ).transpose(1, 2)
                        cos, sin = self_attn.rotary_emb(key_states, seq_len=11)
                        query_states, key_states = apply_rotary_pos_emb(
                            query_states,
                            key_states,
                            cos,
                            sin,
                            position_ids,
                        )
                        expected = compute_manual_attention_probs(
                            query_states,
                            key_states,
                            attention_mask=attention_mask,
                            is_causal=mask_kind != "4d",
                            num_key_value_groups=self_attn.num_key_value_groups,
                        )[:, :, 7:10, 2:6]

                        self.assertEqual(actual.shape, (1, 4, 3, 4))
                        torch.testing.assert_close(actual, expected)

    def _run_prefill(self, num_layers, full_layers, n_vis, n_keep, use_sink=False, training=False):
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=num_layers,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=256,
            use_cache=True,
        )
        config._attn_implementation = "sdpa"
        model = LlamaDynamicvitModel(config)
        model.train(training)
        sequence_length = 2 + n_vis + 1
        model.init_token_total_shape = sequence_length
        scoring_layer_q_lengths = []
        hook = model.layers[full_layers].self_attn.q_proj.register_forward_pre_hook(
            lambda _module, args: scoring_layer_q_lengths.append(args[0].shape[1])
        )
        env = {
            "PHASE_TRANSITION_LAYER": str(full_layers),
            "PHASE_TRANSITION_N_KEEP": str(n_keep),
            "TREVS_USE_SINK_TOKEN": "1" if use_sink else "0",
        }
        with (
            patch.dict(os.environ, env, clear=False),
            patch.object(torch.cuda, "Event", side_effect=lambda **_kwargs: _DummyCudaEvent()),
            patch.object(torch.cuda, "synchronize", side_effect=lambda: None),
        ):
            output = model(
                inputs_embeds=torch.randn(1, sequence_length, config.hidden_size),
                attention_mask=torch.ones(1, sequence_length, dtype=torch.long),
                position_ids=torch.arange(sequence_length).unsqueeze(0),
                use_cache=True,
                image_shape=n_vis,
                token_length_list=[sequence_length],
                pre_prompt_length_list=[2],
            )[2]
        hook.remove()

        phase_visual_tokens = n_keep + int(use_sink)
        short_length = sequence_length - (n_vis - phase_visual_tokens)
        cache_lengths = [int(layer_cache[0].shape[-2]) for layer_cache in output.past_key_values]
        self.assertEqual(
            cache_lengths,
            [sequence_length] * full_layers + [short_length] * (num_layers - full_layers),
        )
        self.assertEqual(scoring_layer_q_lengths, [sequence_length, short_length])
        expected_equal_tokens = (
            n_vis * full_layers + phase_visual_tokens * (num_layers - full_layers)
        ) / num_layers
        self.assertEqual(model.num_token_pool, expected_equal_tokens)

    def test_prefill_scores_full_sequence_then_runs_scoring_layer_short(self):
        self._run_prefill(
            num_layers=3,
            full_layers=1,
            n_vis=6,
            n_keep=2,
        )

    def test_prefill_sink_and_short_policy_work_with_sdpa_training_path(self):
        self._run_prefill(
            num_layers=3,
            full_layers=1,
            n_vis=6,
            n_keep=2,
            use_sink=True,
            training=True,
        )

    def test_layer_eight_budget_has_eight_full_and_twenty_four_short_blocks(self):
        self._run_prefill(
            num_layers=32,
            full_layers=8,
            n_vis=128,
            n_keep=42,
        )

    def test_next_320_budget_has_640_then_213_visual_tokens(self):
        self._run_prefill(
            num_layers=32,
            full_layers=8,
            n_vis=640,
            n_keep=213,
        )

    def test_kv_cache_estimate_uses_full_layer_count_semantics(self):
        actual = estimate_kv_cache_bytes(
            batch_size=1,
            num_hidden_layers=32,
            num_key_value_heads=4,
            head_dim=8,
            bytes_per_elem=2,
            prefill_seq_len_routed=131,
            prefill_seq_len_phase=45,
            phase_transition_layer=8,
        )
        token_sum = 131 * 8 + 45 * 24
        self.assertEqual(actual, 2 * 1 * 4 * 8 * 2 * token_sum)

        for invalid_layer in (0, 32):
            with self.subTest(invalid_layer=invalid_layer), self.assertRaises(ValueError):
                estimate_kv_cache_bytes(
                    batch_size=1,
                    num_hidden_layers=32,
                    num_key_value_heads=4,
                    head_dim=8,
                    bytes_per_elem=2,
                    prefill_seq_len_routed=131,
                    prefill_seq_len_phase=45,
                    phase_transition_layer=invalid_layer,
                )


if __name__ == "__main__":
    unittest.main()
