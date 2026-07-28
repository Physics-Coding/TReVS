import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ["METHOD"] = "trevs"

import torch

from llava.constants import IMAGE_TOKEN_INDEX
from llava.model import llava_arch


class _DummyNextModel:
    def __init__(self):
        self.config = SimpleNamespace(
            mm_patch_merge_type="spatial_unpad",
            image_aspect_ratio="anyres",
            image_grid_pinpoints=[[672, 672]],
            tokenizer_model_max_length=None,
            tune_mm_mlp_adapter=False,
            mm_use_im_start_end=False,
        )
        self.model = SimpleNamespace(
            embed_tokens=torch.nn.Embedding(32, 8),
            image_newline=torch.zeros(8),
            trevs_route_stats=None,
            efficiency_ctx=None,
            init_token_total_shape=0,
        )
        self.device = torch.device("cpu")
        self._vision_tower = SimpleNamespace(
            num_patches_per_side=24,
            config=SimpleNamespace(image_size=336),
        )

    def get_model(self):
        return self.model

    def get_vision_tower(self):
        return self._vision_tower

    def encode_images_with_vit_attn_and_semantic_layer(self, images, semantic_layer=None):
        batch_size = images.shape[0]
        features = torch.arange(batch_size * 576 * 8, dtype=torch.float32).reshape(batch_size, 576, 8)
        vit_attention = torch.zeros(batch_size, 1, 1, 1)
        return features, vit_attention, None

    prepare_sparse_inputs_labels_for_multimodal = (
        llava_arch.LlavaMetaForCausalLM.prepare_sparse_inputs_labels_for_multimodal
    )


def _route_first_128(**kwargs):
    features = kwargs["V_proj"]
    batch_size = features.shape[0]
    indices = torch.arange(128).unsqueeze(0).expand(batch_size, -1)
    routed = torch.gather(features, 1, indices.unsqueeze(-1).expand(-1, -1, features.shape[-1]))
    return indices, {
        "idx_track1": indices[:, :96],
        "idx_track2": indices[:, 96:],
        "idx_routed": indices,
        "V_routed": routed,
        "selection_mode": "test",
    }


class LlavaNextStage1RoutingTest(unittest.TestCase):
    def test_trevs_keeps_exactly_128_tokens_per_crop_without_unpadding(self):
        for image_size in ((672, 672), (672, 336)):
            with self.subTest(image_size=image_size):
                model = _DummyNextModel()
                input_ids = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2]])
                images = torch.zeros(1, 5, 1, 1, 1)

                with patch.object(llava_arch, "trevs_route", side_effect=_route_first_128):
                    outputs = model.prepare_sparse_inputs_labels_for_multimodal(
                        input_ids=input_ids,
                        position_ids=None,
                        attention_mask=None,
                        past_key_values=None,
                        labels=None,
                        images=images,
                        image_sizes=[image_size],
                    )

                inputs_embeds = outputs[4]
                image_shape = outputs[6]
                token_length_list = outputs[7]
                self.assertEqual(image_shape, 5 * 128)
                self.assertEqual(inputs_embeds.shape[1], 2 + image_shape)
                self.assertEqual(token_length_list, [2 + image_shape])


if __name__ == "__main__":
    unittest.main()
