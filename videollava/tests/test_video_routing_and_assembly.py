import os
from types import MethodType
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn

from videollava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from videollava.model.language_model.trevs_router import (
    apply_sink_token_pruning,
    score_phase_attention,
)
from videollava.model.language_model.sparse_videollava_llama import (
    VideoLlavaConfig,
    VideoLlavaTReVSForCausalLM,
)
from videollava.model.llava_arch import (
    LlavaMetaForCausalLM,
    assemble_video_sample,
    route_video_frames,
)


class StageOneRoutingTests(unittest.TestCase):
    def test_eight_frames_exclude_cls_and_keep_exact_budget(self):
        torch.manual_seed(11)
        projected = torch.randn(1, 8, 256, 16)
        full_attention = torch.rand(8, 16, 257, 257)
        text = torch.randn(1, 6, 16)
        text_mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool)
        environment = {
            "METHOD": "trevs",
            "TREVS_TEXT_SCORE_MODE": "rms",
            "DOUBLE_TRACK_USE_CONSISTENCY_REWARD": "1",
        }
        with mock.patch.dict(os.environ, environment, clear=False):
            routed, stats = route_video_frames(
                vit_attn=full_attention,
                projected_patches=projected,
                text_embeddings=text,
                text_mask=text_mask,
                k_track1=96,
                k_track2=32,
            )

        self.assertEqual(tuple(routed.shape), (1, 8, 128, 16))
        self.assertEqual(stats["total_visual_tokens"], 1024)
        self.assertEqual(tuple(stats["idx_track1"].shape), (1, 8, 96))
        self.assertEqual(tuple(stats["idx_track2"].shape), (1, 8, 32))
        self.assertEqual(tuple(stats["idx_routed"].shape), (1, 8, 128))
        self.assertLess(int(stats["idx_routed"].max()), 256)
        self.assertTrue(
            torch.all(stats["idx_routed"][..., 1:] >= stats["idx_routed"][..., :-1])
        )
        for frame_idx in range(8):
            track1 = set(stats["idx_track1"][0, frame_idx].tolist())
            track2 = set(stats["idx_track2"][0, frame_idx].tolist())
            self.assertTrue(track1.isdisjoint(track2))

    def test_stage_one_rejects_patch_only_attention(self):
        with self.assertRaisesRegex(ValueError, r"full CLS\+patch attention"):
            route_video_frames(
                vit_attn=torch.rand(8, 2, 256, 256),
                projected_patches=torch.rand(1, 8, 256, 4),
                text_embeddings=torch.rand(1, 2, 4),
                text_mask=torch.ones(1, 2, dtype=torch.bool),
                k_track1=2,
                k_track2=1,
            )


class AssemblyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.embedding = nn.Embedding(128, 8)
        self.input_ids = torch.tensor(
            [1] + [IMAGE_TOKEN_INDEX] * 8 + [10, 11, 2], dtype=torch.long
        )
        self.labels = self.input_ids.clone()
        self.frame_features = torch.arange(8 * 128 * 8, dtype=torch.float32).reshape(
            8, 128, 8
        )

    def test_eight_placeholders_expand_to_one_contiguous_span(self):
        embeddings, labels, visual_start, frame_lengths = assemble_video_sample(
            self.input_ids,
            self.labels,
            self.frame_features,
            self.embedding,
        )
        self.assertEqual(visual_start, 1)
        self.assertEqual(frame_lengths, [128] * 8)
        self.assertEqual(embeddings.shape[0], 1 + 1024 + 3)
        self.assertEqual(visual_start + sum(frame_lengths), 1025)
        self.assertTrue(
            torch.equal(embeddings[visual_start:1025], self.frame_features.reshape(1024, 8))
        )
        self.assertTrue(torch.all(labels[visual_start:1025] == IGNORE_INDEX))
        self.assertEqual(labels[:visual_start].tolist(), [1])
        self.assertEqual(labels[1025:].tolist(), [10, 11, 2])

    def test_placeholder_count_and_contiguity_are_strict(self):
        with self.assertRaisesRegex(ValueError, "Expected 8 visual placeholders"):
            assemble_video_sample(
                self.input_ids[:-1].masked_fill(
                    torch.arange(self.input_ids.numel() - 1) == 8, 9
                ),
                self.labels[:-1],
                self.frame_features,
                self.embedding,
            )
        nonconsecutive = torch.tensor(
            [1] + [IMAGE_TOKEN_INDEX] * 4 + [9] + [IMAGE_TOKEN_INDEX] * 4 + [2]
        )
        with self.assertRaisesRegex(ValueError, "must be consecutive"):
            assemble_video_sample(
                nonconsecutive,
                nonconsecutive.clone(),
                self.frame_features,
                self.embedding,
            )


class _DummyBackbone:
    def __init__(self, hidden_size: int):
        self.embed_tokens = nn.Embedding(128, hidden_size)
        self.trevs_route_stats = None
        self.last_multimodal_metrics = None


class _DummyAssembler(LlavaMetaForCausalLM):
    def __init__(self, frame_features: torch.Tensor, context_length: int = 4096):
        self._backbone = _DummyBackbone(frame_features.shape[-1])
        self._frame_features = frame_features.unsqueeze(0)
        self.config = SimpleNamespace(
            tokenizer_padding_side="right",
            max_position_embeddings=context_length,
        )

    def get_model(self):
        return self._backbone

    def _encode_and_route_videos(self, videos, input_ids):
        del videos, input_ids
        return self._frame_features, {
            "selection_mode": "trevs",
            "total_visual_tokens": int(self._frame_features.shape[1] * self._frame_features.shape[2]),
        }


class MultimodalMetadataTests(unittest.TestCase):
    def test_prefill_metadata_and_positions_are_rebuilt(self):
        frame_features = torch.randn(8, 128, 8)
        assembler = _DummyAssembler(frame_features)
        input_ids = torch.tensor(
            [[1] + [IMAGE_TOKEN_INDEX] * 8 + [10, 11]], dtype=torch.long
        )
        result = assembler.prepare_sparse_inputs_labels_for_multimodal(
            input_ids=input_ids,
            position_ids=torch.full_like(input_ids, 99),
            attention_mask=torch.ones_like(input_ids),
            past_key_values=None,
            labels=None,
            images=torch.zeros(1, 3, 8, 224, 224),
        )
        _, positions, mask, _, inputs_embeds, labels, image_shape, lengths, starts = result
        self.assertIsNone(labels)
        self.assertEqual(image_shape, 1024)
        self.assertEqual(starts, [1])
        self.assertEqual(lengths, [1027])
        self.assertEqual(tuple(inputs_embeds.shape), (1, 1027, 8))
        self.assertEqual(mask.sum().item(), 1027)
        self.assertTrue(torch.equal(positions[0], torch.arange(1027)))
        metrics = assembler.get_model().last_multimodal_metrics
        self.assertEqual(metrics["visual_start"], 1)
        self.assertEqual(metrics["text_start"], 1025)
        self.assertEqual(metrics["frame_token_lengths"], [128] * 8)

    def test_context_exhaustion_raises_instead_of_truncating(self):
        assembler = _DummyAssembler(torch.randn(8, 128, 8), context_length=1000)
        input_ids = torch.tensor(
            [[1] + [IMAGE_TOKEN_INDEX] * 8 + [10, 11]], dtype=torch.long
        )
        with self.assertRaisesRegex(ValueError, "cannot be truncated"):
            assembler.prepare_sparse_inputs_labels_for_multimodal(
                input_ids=input_ids,
                position_ids=None,
                attention_mask=None,
                past_key_values=None,
                labels=None,
                images=torch.zeros(1, 3, 8, 224, 224),
            )

    def test_tiny_wrapper_runs_assembly_phase_transition_and_greedy_generation(self):
        config = VideoLlavaConfig(
            vocab_size=128,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=2048,
            bos_token_id=1,
            eos_token_id=None,
            pad_token_id=0,
        )
        config._attn_implementation = "sdpa"
        model = VideoLlavaTReVSForCausalLM(config).eval()
        frame_features = torch.randn(1, 8, 128, 16)

        def fake_encode_and_route(self, videos, input_ids):
            del videos, input_ids
            return frame_features, {
                "selection_mode": "trevs",
                "total_visual_tokens": 1024,
            }

        model._encode_and_route_videos = MethodType(fake_encode_and_route, model)
        input_ids = torch.tensor(
            [[1] + [IMAGE_TOKEN_INDEX] * 8 + [10, 11]], dtype=torch.long
        )
        environment = {
            "METHOD": "trevs",
            "PHASE_TRANSITION_LAYER": "2",
            "PHASE_TRANSITION_N_KEEP": "341",
            "TREVS_USE_SINK_TOKEN": "0",
            "TREVS_PHASE_SCORING": "priority_heads",
        }
        with mock.patch.dict(os.environ, environment, clear=False):
            output = model.generate(
                inputs=input_ids,
                images=[torch.zeros(3, 8, 224, 224)],
                max_new_tokens=1,
                do_sample=False,
                num_beams=1,
                temperature=0.0,
                use_cache=True,
            )
        self.assertEqual(tuple(output.shape), (1, 2))
        self.assertEqual(model.image_shape, 1024)
        self.assertEqual(model.pre_prompt_length_list, [1])
        self.assertEqual(model.token_length_list, [1027])
        self.assertEqual(model.get_model().last_cache_lengths, [1027, 1027, 344, 344])
        self.assertEqual(model.get_model().last_trevs_metrics["n_vis_before_phase"], 1024)
        self.assertEqual(model.get_model().last_trevs_metrics["n_vis_phase"], 341)


class StageTwoPruningTests(unittest.TestCase):
    def test_global_1024_to_341_sink_off_and_342_sink_on(self):
        torch.manual_seed(23)
        attention_tv = torch.rand(1, 8, 3, 1024)
        keep, drop = score_phase_attention(
            attention_tv, n_keep=341, mode="priority_heads"
        )
        self.assertEqual(tuple(keep.shape), (1, 341))
        self.assertEqual(tuple(drop.shape), (1, 683))

        total_length = 2 + 1024 + 3
        hidden = torch.arange(total_length, dtype=torch.float32).reshape(1, total_length, 1)
        positions = torch.arange(total_length).unsqueeze(0)
        mask = torch.ones(1, total_length, dtype=torch.long)
        for use_sink, expected_visual in ((False, 341), (True, 342)):
            with self.subTest(use_sink=use_sink):
                pruned_hidden, pruned_mask, pruned_positions = apply_sink_token_pruning(
                    hidden_states=hidden,
                    v_token_start=2,
                    n_vis_current=1024,
                    idx_keep_local=keep,
                    idx_drop_local=drop,
                    attention_mask=mask,
                    position_ids=positions,
                    use_sink=use_sink,
                )
                self.assertEqual(pruned_hidden.shape[1], 2 + expected_visual + 3)
                self.assertEqual(pruned_mask.shape[1], pruned_hidden.shape[1])
                self.assertTrue(torch.equal(pruned_hidden[:, :2], hidden[:, :2]))
                self.assertTrue(torch.equal(pruned_hidden[:, -3:], hidden[:, -3:]))
                self.assertTrue(torch.equal(pruned_positions[:, :2], positions[:, :2]))
                self.assertTrue(
                    torch.equal(pruned_positions[:, -3:], positions[:, -3:])
                )
                self.assertGreater(int(pruned_positions.max()), pruned_positions.shape[1] - 1)

    def test_four_dimensional_mask_is_pruned_on_query_and_key_axes(self):
        hidden = torch.randn(1, 10, 4)
        positions = torch.arange(10).unsqueeze(0)
        mask = torch.zeros(1, 1, 10, 10)
        result = apply_sink_token_pruning(
            hidden_states=hidden,
            v_token_start=2,
            n_vis_current=6,
            idx_keep_local=torch.tensor([[1, 4]]),
            idx_drop_local=torch.tensor([[0, 2, 3, 5]]),
            attention_mask=mask,
            position_ids=positions,
            use_sink=False,
        )
        self.assertEqual(tuple(result[0].shape), (1, 6, 4))
        self.assertEqual(tuple(result[1].shape), (1, 1, 6, 6))
        self.assertEqual(tuple(result[2].shape), (1, 6))


if __name__ == "__main__":
    unittest.main()
