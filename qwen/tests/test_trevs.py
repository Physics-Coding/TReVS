import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from qwen.eval.attention_backend import configure_qwen_attention_backend, resolve_qwen_attention_backend
from qwen.model.qwen2_5_vl_custom import (
    _apply_trevs_phase_pruning_qwen,
    _build_text_embeddings,
    _load_hf_qwen_module,
    _prune_sequence_by_image_indices,
    apply_qwen2_5_vl_trevs_patches,
    qwen_for_conditional_generation_forward_trevs,
    qwen_text_model_forward_trevs,
    qwen_vision_forward_trevs,
    qwen_vl_model_forward_trevs,
)
from qwen.model.trevs_router import (
    get_trevs_text_score_mode,
    is_trevs_enabled,
    score_phase_attention,
    trevs_route,
    use_consistency_reward,
)


class QwenTReVSRouterTest(unittest.TestCase):
    def test_attention_backend_is_explicit_for_trevs_and_optional_for_dense(self):
        cases = (
            ({"METHOD": "trevs", "USE_FLASH_ATTN": "0"}, "sdpa"),
            ({"METHOD": "trevs", "USE_FLASH_ATTN": "false"}, "sdpa"),
            ({"METHOD": "dense", "USE_FLASH_ATTN": "1"}, "flash_attention_2"),
            ({"METHOD": "dense", "USE_FLASH_ATTN": "0"}, None),
        )
        for env, expected_backend in cases:
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                self.assertEqual(resolve_qwen_attention_backend(), expected_backend)
                model_kwargs = {"torch_dtype": "auto"}
                configure_qwen_attention_backend(model_kwargs)
                if expected_backend is None:
                    self.assertNotIn("attn_implementation", model_kwargs)
                else:
                    self.assertEqual(model_kwargs["attn_implementation"], expected_backend)

    def test_trevs_attention_backend_rejects_flash_attention(self):
        for flash_value in ("1", "true"):
            with self.subTest(flash_value=flash_value), patch.dict(
                os.environ,
                {"METHOD": "trevs", "USE_FLASH_ATTN": flash_value},
                clear=True,
            ), self.assertRaisesRegex(ValueError, "TReVS requires USE_FLASH_ATTN=0"):
                resolve_qwen_attention_backend()

    def test_dense_keeps_hugging_face_forwards_and_trevs_installs_patches(self):
        hf_qwen = _load_hf_qwen_module()
        forward_targets = (
            (hf_qwen.Qwen2_5_VisionTransformerPretrainedModel, qwen_vision_forward_trevs),
            (hf_qwen.Qwen2_5_VLTextModel, qwen_text_model_forward_trevs),
            (hf_qwen.Qwen2_5_VLModel, qwen_vl_model_forward_trevs),
            (hf_qwen.Qwen2_5_VLForConditionalGeneration, qwen_for_conditional_generation_forward_trevs),
        )
        original_forwards = tuple(model_class.forward for model_class, _ in forward_targets)

        with patch.dict(os.environ, {"METHOD": "dense"}, clear=False):
            self.assertFalse(apply_qwen2_5_vl_trevs_patches())
        self.assertEqual(
            tuple(model_class.forward for model_class, _ in forward_targets),
            original_forwards,
        )
        for original_forward, (_, trevs_forward) in zip(original_forwards, forward_targets):
            self.assertIsNot(original_forward, trevs_forward)

        from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
            Qwen2_5_VLConfig,
            Qwen2_5_VLTextConfig,
            Qwen2_5_VLVisionConfig,
        )

        text_config = Qwen2_5_VLTextConfig(
            vocab_size=64,
            hidden_size=24,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=64,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 1_000_000.0,
                "mrope_section": [2, 2, 2],
            },
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        vision_config = Qwen2_5_VLVisionConfig(
            depth=2,
            hidden_size=24,
            intermediate_size=48,
            num_heads=2,
            in_channels=3,
            patch_size=2,
            spatial_merge_size=2,
            temporal_patch_size=1,
            window_size=8,
            out_hidden_size=24,
            fullatt_block_indexes=[1],
        )
        config = Qwen2_5_VLConfig(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=60,
            video_token_id=61,
            vision_start_token_id=58,
            vision_end_token_id=59,
        )
        dense_model = hf_qwen.Qwen2_5_VLForConditionalGeneration(config).eval()
        dense_input_ids = torch.tensor([[1, 58, 60, 60, 60, 60, 59, 2]])
        with patch.dict(os.environ, {"METHOD": "dense"}, clear=False), torch.no_grad():
            dense_outputs = dense_model(
                input_ids=dense_input_ids,
                attention_mask=torch.ones_like(dense_input_ids),
                pixel_values=torch.randn(16, 12),
                image_grid_thw=torch.tensor([[1, 4, 4]]),
                use_cache=False,
            )
        self.assertEqual(dense_outputs.logits.shape, (1, dense_input_ids.shape[1], text_config.vocab_size))
        self.assertTrue(torch.isfinite(dense_outputs.logits).all())
        with patch.dict(os.environ, {"METHOD": "dense"}, clear=False), torch.no_grad():
            dense_generated_ids = dense_model.generate(
                input_ids=dense_input_ids,
                attention_mask=torch.ones_like(dense_input_ids),
                pixel_values=torch.randn(16, 12),
                image_grid_thw=torch.tensor([[1, 4, 4]]),
                max_new_tokens=2,
                do_sample=False,
                use_cache=True,
            )
        self.assertGreater(dense_generated_ids.shape[1], dense_input_ids.shape[1])
        self.assertLessEqual(dense_generated_ids.shape[1], dense_input_ids.shape[1] + 2)

        with patch.dict(os.environ, {"METHOD": "trevs"}, clear=False):
            self.assertTrue(apply_qwen2_5_vl_trevs_patches())
        for model_class, trevs_forward in forward_targets:
            self.assertIs(model_class.forward, trevs_forward)

        with patch.dict(os.environ, {"METHOD": "dense"}, clear=False):
            self.assertFalse(apply_qwen2_5_vl_trevs_patches())
        self.assertEqual(
            tuple(model_class.forward for model_class, _ in forward_targets),
            original_forwards,
        )

        # Leave the process configured for the remaining TReVS model tests.
        with patch.dict(os.environ, {"METHOD": "trevs"}, clear=False):
            self.assertTrue(apply_qwen2_5_vl_trevs_patches())

    def test_routing_text_uses_only_tokens_after_visual_placeholder_block(self):
        embedding = torch.nn.Embedding(128, 4)
        input_ids = torch.tensor([[1, 2, 99, 99, 10, 11, 12]])
        attention_mask = torch.ones_like(input_ids)
        text_embeds, text_mask = _build_text_embeddings(
            embedding,
            input_ids,
            attention_mask,
            image_token_id=99,
            video_token_id=98,
        )

        self.assertTrue(torch.equal(text_embeds, embedding(torch.tensor([[10, 11, 12]]))))
        self.assertTrue(torch.equal(text_mask, torch.ones(1, 3, dtype=torch.bool)))

    def test_stage1_route_exposes_trevs_controls(self):
        torch.manual_seed(7)
        vision_tokens = torch.randn(1, 8, 6)
        semantic_tokens = torch.randn(1, 8, 6)
        text_tokens = torch.randn(1, 4, 6)
        text_mask = torch.tensor([[True, True, True, False]])
        vit_attention = torch.softmax(torch.randn(1, 2, 8, 8), dim=-1)

        env = {
            "METHOD": "trevs",
            "TREVS_TEXT_SCORE_MODE": "rms",
            "TREVS_VISUAL_TEMPERATURE": "2.0",
            "TREVS_TEXT_TEMPERATURE": "3.0",
            "TREVS_USE_CONSISTENCY_REWARD": "0",
        }
        with patch.dict(os.environ, env, clear=False):
            indices, stats = trevs_route(
                vit_attn=vit_attention,
                V_proj=vision_tokens,
                T_emb=text_tokens,
                n_topk=3,
                n_fps=2,
                core_token_mask=text_mask,
                V_semantic_proj=semantic_tokens,
                semantic_layer=5,
                return_stats=True,
            )

        self.assertEqual(indices.shape, (1, 5))
        self.assertEqual(torch.unique(indices).numel(), 5)
        self.assertTrue(torch.equal(indices, torch.sort(indices, dim=-1).values))
        self.assertEqual(stats["selection_mode"], "trevs")
        self.assertEqual(stats["visual_temperature"], 2.0)
        self.assertEqual(stats["text_temperature"], 3.0)
        self.assertFalse(stats["consistency_reward_enabled"])
        self.assertEqual(stats["semantic_layer"], 5)

    def test_legacy_method_aliases_are_rejected(self):
        for method in ("rcer", "v3_9", "v3_10", "qwen", "original_qwen"):
            with self.subTest(method=method), patch.dict(os.environ, {"METHOD": method}, clear=True):
                with self.assertRaisesRegex(ValueError, "expected trevs or dense"):
                    is_trevs_enabled()

    def test_legacy_environment_names_are_ignored(self):
        legacy_env = {
            "METHOD": "trevs",
            "DOUBLE_TRACK_TEXT_SCORE_MODE": "max",
            "DOUBLE_TRACK_USE_CONSISTENCY_REWARD": "0",
        }
        with patch.dict(os.environ, legacy_env, clear=True):
            self.assertEqual(get_trevs_text_score_mode(), "rms")
            self.assertTrue(use_consistency_reward())

    def test_phase_scoring_modes_return_sorted_partition(self):
        torch.manual_seed(11)
        attention = torch.softmax(torch.randn(2, 4, 3, 7), dim=-1)
        for mode in ("priority_heads", "all_heads"):
            keep, drop = score_phase_attention(attention, n_keep=3, mode=mode)
            self.assertEqual(keep.shape, (2, 3))
            self.assertEqual(drop.shape, (2, 4))
            self.assertTrue(torch.equal(keep, torch.sort(keep, dim=-1).values))
            for batch_idx in range(2):
                merged = torch.cat([keep[batch_idx], drop[batch_idx]]).sort().values
                self.assertTrue(torch.equal(merged, torch.arange(7)))

    def test_sink_on_preserves_original_mrope_positions(self):
        hidden = torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)
        attention_mask = torch.ones(1, 8, dtype=torch.long)
        position_ids = torch.stack(
            [torch.arange(8), torch.arange(10, 18), torch.arange(20, 28)], dim=0
        ).unsqueeze(1)
        text_position_ids = torch.arange(100, 108).unsqueeze(0)
        drop_local = torch.tensor([[0, 2, 3]])

        outputs = _apply_trevs_phase_pruning_qwen(
            hidden_states=hidden.clone(),
            v_token_start=2,
            idx_drop_local=drop_local,
            attention_mask=attention_mask,
            position_ids=position_ids,
            text_position_ids=text_position_ids,
            use_sink=True,
        )
        pruned_hidden, pruned_mask, pruned_positions, pruned_text_positions = outputs
        kept = torch.tensor([0, 1, 2, 3, 6, 7])

        self.assertEqual(pruned_hidden.shape[1], 6)
        self.assertTrue(torch.equal(pruned_mask, attention_mask[:, kept]))
        self.assertTrue(torch.equal(pruned_positions, position_ids[:, :, kept]))
        self.assertTrue(torch.equal(pruned_text_positions, text_position_ids[:, kept]))
        expected_sink = hidden[0, torch.tensor([2, 4, 5])].mean(dim=0)
        self.assertTrue(torch.equal(pruned_hidden[0, 2], expected_sink))

    def test_sink_off_prunes_4d_mask_without_renumbering_positions(self):
        hidden = torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)
        mask = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)
        position_ids = torch.stack(
            [torch.arange(8), torch.arange(10, 18), torch.arange(20, 28)], dim=0
        ).unsqueeze(1)
        text_position_ids = torch.arange(100, 108).unsqueeze(0)
        drop_local = torch.tensor([[0, 2, 3]])

        outputs = _apply_trevs_phase_pruning_qwen(
            hidden_states=hidden.clone(),
            v_token_start=2,
            idx_drop_local=drop_local,
            attention_mask={"full_attention": mask},
            position_ids=position_ids,
            text_position_ids=text_position_ids,
            use_sink=False,
        )
        pruned_hidden, pruned_masks, pruned_positions, pruned_text_positions = outputs
        kept = torch.tensor([0, 1, 3, 6, 7])
        expected_mask = mask[:, :, kept, :][:, :, :, kept]

        self.assertTrue(torch.equal(pruned_hidden, hidden[:, kept]))
        self.assertTrue(torch.equal(pruned_masks["full_attention"], expected_mask))
        self.assertTrue(torch.equal(pruned_positions, position_ids[:, :, kept]))
        self.assertTrue(torch.equal(pruned_text_positions, text_position_ids[:, kept]))

    def test_stage1_pruning_supports_four_channel_positions(self):
        position_ids = torch.arange(4 * 7).reshape(4, 1, 7)
        keep_mask = torch.tensor([[True, False, True, False, True, True, False]])
        pruned = _prune_sequence_by_image_indices(position_ids, keep_mask)
        self.assertTrue(torch.equal(pruned, position_ids[:, :, [0, 2, 4, 5]]))

    def test_tiny_qwen_prefill_and_decode_keep_per_layer_cache_lengths(self):
        apply_qwen2_5_vl_trevs_patches()
        hf_qwen = _load_hf_qwen_module()
        from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig

        config = Qwen2_5_VLTextConfig(
            vocab_size=64,
            hidden_size=24,
            intermediate_size=48,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=64,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 1_000_000.0,
                "mrope_section": [2, 2, 2],
            },
            bos_token_id=1,
            eos_token_id=2,
        )
        packed_positions = torch.stack([torch.arange(10)] * 4).unsqueeze(1)
        prefill_mask = torch.ones(1, 10, dtype=torch.long)

        for sink_value, short_prefill_length in (("1", 9), ("0", 8)):
            with self.subTest(sink_value=sink_value):
                model = hf_qwen.Qwen2_5_VLTextModel(config).eval()
                model.n_image_tokens = 4
                model.image_start_index = 2
                scoring_q_lengths = []
                scoring_q_hook = model.layers[1].self_attn.q_proj.register_forward_pre_hook(
                    lambda _module, args: scoring_q_lengths.append(args[0].shape[1])
                )
                env = {
                    "METHOD": "trevs",
                    "TREVS_PHASE_SCORING": "priority_heads",
                    "TREVS_USE_SINK_TOKEN": sink_value,
                    "PHASE_TRANSITION_LAYER": "1",
                    "PHASE_TRANSITION_N_KEEP": "2",
                }
                with patch.dict(os.environ, env, clear=False), torch.no_grad():
                    prefill = model(
                        inputs_embeds=torch.randn(1, 10, config.hidden_size),
                        position_ids=packed_positions,
                        attention_mask=prefill_mask,
                        use_cache=True,
                    )
                    self.assertEqual(
                        prefill.last_hidden_state.shape,
                        (1, short_prefill_length, config.hidden_size),
                    )
                    self.assertEqual(
                        [layer.get_seq_length() for layer in prefill.past_key_values.layers],
                        [10, short_prefill_length, short_prefill_length],
                    )
                    self.assertEqual(scoring_q_lengths, [10, short_prefill_length])
                    self.assertEqual(
                        model.trevs_phase_stats,
                        {
                            "layer": 1,
                            "full_layers": 1,
                            "scoring_layer_idx": 1,
                            "pruned_after_layer_idx": 0,
                            "n_vis_before": 4,
                            "n_vis_after": 3 if sink_value == "1" else 2,
                            "scoring": "priority_heads",
                            "use_sink_token": sink_value == "1",
                        },
                    )

                    decode = model(
                        inputs_embeds=torch.randn(1, 1, config.hidden_size),
                        position_ids=torch.tensor([[[10]], [[10]], [[10]]]),
                        attention_mask=torch.ones(1, 11, dtype=torch.long),
                        past_key_values=prefill.past_key_values,
                        use_cache=True,
                    )
                    self.assertEqual(decode.last_hidden_state.shape, (1, 1, config.hidden_size))
                    self.assertEqual(
                        [layer.get_seq_length() for layer in decode.past_key_values.layers],
                        [11, short_prefill_length + 1, short_prefill_length + 1],
                    )
                    self.assertEqual(scoring_q_lengths, [10, short_prefill_length, 1])
                scoring_q_hook.remove()

    def test_text_only_prefill_clears_visual_span_from_previous_sample(self):
        apply_qwen2_5_vl_trevs_patches()
        hf_qwen = _load_hf_qwen_module()
        from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig

        config = Qwen2_5_VLTextConfig(
            vocab_size=64,
            hidden_size=24,
            intermediate_size=48,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=64,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 1_000_000.0,
                "mrope_section": [2, 2, 2],
            },
            bos_token_id=1,
            eos_token_id=2,
        )
        language_model = hf_qwen.Qwen2_5_VLTextModel(config).eval()
        language_model.n_image_tokens = 4
        language_model.image_start_index = 2

        class TextOnlyVLModel:
            def __init__(self, text_model):
                self.language_model = text_model
                self.config = SimpleNamespace(image_token_id=62, video_token_id=63)
                self.rope_deltas = None

            def get_input_embeddings(self):
                return self.language_model.embed_tokens

            def compute_3d_position_ids(self, inputs_embeds, **_kwargs):
                positions = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
                positions = positions.unsqueeze(0).expand(inputs_embeds.shape[0], -1)
                return torch.stack([positions] * 4)

        model = TextOnlyVLModel(language_model)
        input_ids = torch.arange(10).unsqueeze(0)
        env = {
            "METHOD": "trevs",
            "TREVS_USE_SINK_TOKEN": "0",
            "PHASE_TRANSITION_LAYER": "1",
            "PHASE_TRANSITION_N_KEEP": "2",
        }
        with patch.dict(os.environ, env, clear=False), torch.no_grad():
            outputs = qwen_vl_model_forward_trevs(
                model,
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
            )

        self.assertEqual(outputs.last_hidden_state.shape, (1, 10, config.hidden_size))
        self.assertEqual(language_model.n_image_tokens, 0)
        self.assertIsNone(language_model.image_start_index)

    def test_qwen_lookahead_phase_layer_rejects_invalid_boundaries(self):
        apply_qwen2_5_vl_trevs_patches()
        hf_qwen = _load_hf_qwen_module()
        from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig

        config = Qwen2_5_VLTextConfig(
            vocab_size=64,
            hidden_size=24,
            intermediate_size=48,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=64,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 1_000_000.0,
                "mrope_section": [2, 2, 2],
            },
            bos_token_id=1,
            eos_token_id=2,
        )

        for phase_layer in ("0", "3"):
            with self.subTest(phase_layer=phase_layer):
                model = hf_qwen.Qwen2_5_VLTextModel(config).eval()
                model.n_image_tokens = 4
                model.image_start_index = 2
                env = {
                    "METHOD": "trevs",
                    "PHASE_TRANSITION_LAYER": phase_layer,
                    "PHASE_TRANSITION_N_KEEP": "2",
                }
                with patch.dict(os.environ, env, clear=False), self.assertRaisesRegex(
                    ValueError, "number of decoder blocks that run on the full sequence"
                ):
                    model(
                        inputs_embeds=torch.randn(1, 10, config.hidden_size),
                        position_ids=torch.stack([torch.arange(10)] * 4).unsqueeze(1),
                        attention_mask=torch.ones(1, 10, dtype=torch.long),
                    )

    def test_tiny_qwen_vision_returns_merged_attention_and_semantic_layer(self):
        apply_qwen2_5_vl_trevs_patches()
        hf_qwen = _load_hf_qwen_module()
        from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig

        config = Qwen2_5_VLVisionConfig(
            depth=2,
            hidden_size=24,
            intermediate_size=48,
            num_heads=2,
            in_channels=3,
            patch_size=2,
            spatial_merge_size=2,
            temporal_patch_size=1,
            window_size=8,
            out_hidden_size=24,
            fullatt_block_indexes=[1],
        )
        model = hf_qwen.Qwen2_5_VisionTransformerPretrainedModel(config).eval()
        raw_patches = torch.randn(16, 12)
        grid_thw = torch.tensor([[1, 4, 4]])

        with patch.dict(os.environ, {"METHOD": "trevs", "TREVS_SEMANTIC_LAYER": "0"}), torch.no_grad():
            outputs = model(raw_patches, grid_thw=grid_thw)

        self.assertEqual(outputs.pooler_output.shape, (4, config.out_hidden_size))
        self.assertEqual(outputs.attentions.shape, (1, config.num_heads, 4, 4))
        self.assertEqual(outputs.hidden_states[0].shape, outputs.pooler_output.shape)


if __name__ == "__main__":
    unittest.main()
