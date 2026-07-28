import os
import unittest
from unittest.mock import patch

os.environ["METHOD"] = "trevs"

import torch

from llava.model.language_model.trevs_router import (
    get_trevs_stage1_scoring,
    score_phase_attention,
    trevs_route,
)


class TrevsStage1ScoringTest(unittest.TestCase):
    def test_default_preserves_trevs_scoring(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TREVS_STAGE1_SCORING", None)
            self.assertEqual(get_trevs_stage1_scoring(), "trevs")

    def test_invalid_stage1_scoring_is_rejected(self):
        with patch.dict(os.environ, {"TREVS_STAGE1_SCORING": "semantic_only"}, clear=False):
            with self.assertRaisesRegex(ValueError, "Invalid stage-1 scoring mode"):
                get_trevs_stage1_scoring()

    def test_cls_mode_uses_only_mean_cls_to_patch_attention(self):
        n_vis = 6
        vit_attn = torch.zeros(1, 2, n_vis + 1, n_vis + 1)
        cls_scores_head0 = torch.tensor([0.1, 0.8, 0.3, 0.2, 0.9, 0.4])
        cls_scores_head1 = torch.tensor([0.2, 0.6, 0.1, 0.3, 0.7, 0.5])
        vit_attn[0, 0, 0, 1:] = cls_scores_head0
        vit_attn[0, 1, 0, 1:] = cls_scores_head1
        vision_features = torch.randn(1, n_vis, 4)

        with patch.dict(os.environ, {"TREVS_STAGE1_SCORING": "cls"}, clear=False):
            indices_a, stats_a = trevs_route(
                vit_attn=vit_attn,
                V_proj=vision_features,
                T_emb=torch.randn(1, 3, 4),
                k_track1=2,
                k_track2=0,
                return_stats=True,
            )
            indices_b, stats_b = trevs_route(
                vit_attn=vit_attn,
                V_proj=vision_features,
                T_emb=torch.randn(1, 7, 4) * 1000,
                V_semantic_proj=torch.randn_like(vision_features) * 1000,
                k_track1=2,
                k_track2=0,
                return_stats=True,
            )

        expected = torch.tensor([[1, 4]])
        torch.testing.assert_close(indices_a, expected)
        torch.testing.assert_close(indices_b, expected)
        self.assertEqual(stats_a["selection_mode"], "cls")
        self.assertEqual(stats_a["text_score_mode"], "disabled")
        self.assertFalse(stats_a["consistency_reward_enabled"])
        self.assertEqual(stats_a["semantic_guidance_source"], "disabled")
        self.assertEqual(stats_b["selection_mode"], "cls")

    def test_cls_mode_keeps_topk_and_fps_tracks_disjoint(self):
        n_vis = 8
        vit_attn = torch.zeros(1, 2, n_vis + 1, n_vis + 1)
        vit_attn[:, :, 0, 1:] = torch.tensor([0.1, 0.9, 0.2, 0.3, 0.8, 0.4, 0.5, 0.6])

        with patch.dict(os.environ, {"TREVS_STAGE1_SCORING": "cls"}, clear=False):
            indices, stats = trevs_route(
                vit_attn=vit_attn,
                V_proj=torch.randn(1, n_vis, 4),
                T_emb=torch.randn(1, 3, 4),
                k_track1=2,
                k_track2=3,
                return_stats=True,
            )

        torch.testing.assert_close(stats["idx_track1"], torch.tensor([[1, 4]]))
        self.assertEqual(stats["idx_track2"].shape, (1, 3))
        self.assertEqual(indices.shape, (1, 5))
        self.assertEqual(torch.unique(indices).numel(), 5)
        self.assertTrue(
            set(stats["idx_track1"][0].tolist()).isdisjoint(stats["idx_track2"][0].tolist())
        )

    def test_cls_mode_rejects_patch_only_attention(self):
        n_vis = 6
        with patch.dict(os.environ, {"TREVS_STAGE1_SCORING": "cls"}, clear=False):
            with self.assertRaisesRegex(ValueError, "requires full ViT attention with a CLS token"):
                trevs_route(
                    vit_attn=torch.zeros(1, 2, n_vis, n_vis),
                    V_proj=torch.randn(1, n_vis, 4),
                    T_emb=torch.randn(1, 3, 4),
                    k_track1=2,
                    k_track2=0,
                )

    def test_all_heads_phase_scoring_matches_mean_heads_then_max_text(self):
        attention = torch.tensor(
            [
                [
                    [[0.1, 0.8, 0.2, 0.3], [0.7, 0.1, 0.4, 0.2]],
                    [[0.5, 0.2, 0.6, 0.1], [0.3, 0.9, 0.2, 0.4]],
                ]
            ]
        )
        expected_scores = attention.mean(dim=1).max(dim=1).values
        expected_keep = torch.sort(torch.topk(expected_scores, k=2, dim=-1).indices, dim=-1).values

        actual_keep, actual_drop = score_phase_attention(
            attn_tv=attention,
            n_keep=2,
            mode="all_heads",
        )

        torch.testing.assert_close(actual_keep, expected_keep)
        self.assertEqual(actual_drop.shape, (1, 2))
        self.assertEqual(torch.unique(torch.cat([actual_keep, actual_drop], dim=1)).numel(), 4)


if __name__ == "__main__":
    unittest.main()
