#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from abc import ABC, abstractmethod
import os
import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
from .language_model.trevs_router import trevs_route
from .language_model.score import TREVS_ENABLED, TREVS_ROUTE_TOPK, TREVS_ROUTE_FPS, TREVS_TOTAL_TOKENS

TREVS_SEMANTIC_LAYER_ENV = "TREVS_SEMANTIC_LAYER"


def _get_trevs_semantic_layer():
    raw = os.getenv(TREVS_SEMANTIC_LAYER_ENV, "").strip().lower()
    if not raw or raw in {"none", "off", "default", "current"}:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"Invalid {TREVS_SEMANTIC_LAYER_ENV}={raw!r}; expected an integer layer index, "
            "or one of none/off/default/current."
        ) from exc


def _resolve_vit_hidden_state_index(layer_idx: int, num_hidden_states: int) -> int:
    resolved = layer_idx + num_hidden_states if layer_idx < 0 else layer_idx
    if resolved < 0 or resolved >= num_hidden_states:
        raise ValueError(
            f"{TREVS_SEMANTIC_LAYER_ENV}={layer_idx} resolves to hidden_states[{resolved}], "
            f"outside valid range [0, {num_hidden_states - 1}]."
        )
    return resolved


def _get_batch_input_ids(input_ids, attention_mask, batch_idx: int, device):
    cur_ids = input_ids[batch_idx].to(device=device, dtype=torch.long).view(-1)
    if attention_mask is not None:
        cur_attention_mask = attention_mask[batch_idx].to(device=device, dtype=torch.bool).view(-1)
        if cur_attention_mask.numel() == cur_ids.numel():
            cur_ids = cur_ids[cur_attention_mask]
    return cur_ids


def _extract_post_image_text_ids(cur_ids: torch.Tensor):
    image_positions = torch.where(cur_ids == IMAGE_TOKEN_INDEX)[0]
    if image_positions.numel() > 0:
        cur_ids = cur_ids[int(image_positions[0].item()) + 1:]
    cur_text_ids = cur_ids[(cur_ids != IMAGE_TOKEN_INDEX) & (cur_ids >= 0)]
    return cur_text_ids


def _build_routing_text_embeddings(model, input_ids, attention_mask, batch_size: int, device):
    text_ids_per_batch = []
    text_lengths = []
    max_text_len = 0
    for batch_idx in range(batch_size):
        cur_ids = _get_batch_input_ids(input_ids, attention_mask, batch_idx, device)
        cur_text_ids = _extract_post_image_text_ids(cur_ids)
        text_ids_per_batch.append(cur_text_ids)
        cur_text_len = int(cur_text_ids.numel())
        text_lengths.append(cur_text_len)
        max_text_len = max(max_text_len, cur_text_len)

    if len(text_ids_per_batch) == 0 or max_text_len == 0:
        return None, None

    text_ids_padded = torch.full((batch_size, max_text_len), 0, dtype=torch.long, device=device)
    text_mask_padded = torch.zeros(batch_size, max_text_len, dtype=torch.bool, device=device)
    for batch_idx in range(batch_size):
        text_len = text_lengths[batch_idx]
        if text_len == 0:
            continue
        text_ids_padded[batch_idx, :text_len] = text_ids_per_batch[batch_idx]
        text_mask_padded[batch_idx, :text_len] = True

    with torch.no_grad():
        text_embeds = model.embed_tokens(text_ids_padded)
    return text_embeds, text_mask_padded


def _repeat_routing_text_by_split_sizes(text_embeds, text_mask, split_sizes):
    if text_embeds is None or text_mask is None:
        return None, None
    repeat_counts = torch.as_tensor(split_sizes, dtype=torch.long, device=text_embeds.device)
    return (
        torch.repeat_interleave(text_embeds, repeat_counts, dim=0),
        torch.repeat_interleave(text_mask, repeat_counts, dim=0),
    )


def _count_image_tokens_per_sample(input_ids, attention_mask, batch_size: int):
    image_counts = []
    for batch_idx in range(batch_size):
        cur_ids = input_ids[batch_idx].view(-1)
        if attention_mask is not None:
            cur_attention_mask = attention_mask[batch_idx].to(device=cur_ids.device, dtype=torch.bool).view(-1)
            cur_ids = cur_ids[cur_attention_mask]
        image_counts.append(int((cur_ids == IMAGE_TOKEN_INDEX).sum().item()))
    return image_counts


def _repeat_routing_text_by_image_splits(text_embeds, text_mask, image_counts, split_sizes):
    if text_embeds is None or text_mask is None:
        return None, None
    num_image_groups = len(split_sizes)
    if sum(image_counts) == num_image_groups:
        image_repeat_counts = torch.as_tensor(image_counts, dtype=torch.long, device=text_embeds.device)
        text_embeds = torch.repeat_interleave(text_embeds, image_repeat_counts, dim=0)
        text_mask = torch.repeat_interleave(text_mask, image_repeat_counts, dim=0)
    elif text_embeds.shape[0] == num_image_groups:
        pass
    elif text_embeds.shape[0] == 1:
        text_embeds = torch.repeat_interleave(text_embeds, num_image_groups, dim=0)
        text_mask = torch.repeat_interleave(text_mask, num_image_groups, dim=0)
    else:
        raise ValueError(
            f"Cannot align {text_embeds.shape[0]} routing text batch entries with {num_image_groups} image groups."
        )
    return _repeat_routing_text_by_split_sizes(text_embeds, text_mask, split_sizes)


def _index_mask_from_indices(indices, num_tokens: int):
    index_mask = torch.zeros(indices.shape[0], num_tokens, dtype=torch.bool, device=indices.device)
    if indices.numel() > 0:
        index_mask.scatter_(1, indices, True)
    return index_mask


def _store_trevs_route_stats(model, idx_routed, route_stats):
    model.trevs_route_stats = {
        "selection_mode": route_stats.get("selection_mode", "trevs"),
        "idx_track1": route_stats["idx_track1"],
        "idx_track2": route_stats["idx_track2"],
        "idx_routed": idx_routed,
    }
    if "text_score_mode" in route_stats:
        model.trevs_route_stats["text_score_mode"] = route_stats["text_score_mode"]
    for stat_key in (
        "method",
        "visual_temperature",
        "text_temperature",
        "consistency_reward_enabled",
        "semantic_layer",
        "semantic_guidance_source",
    ):
        if stat_key in route_stats:
            model.trevs_route_stats[stat_key] = route_stats[stat_key]


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            self.mm_projector = self.mm_projector.to(torch.bfloat16)


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def encode_images_with_vit_attn(self, images):
        """Encode images and return full ViT attention for two-track TReVS routing."""
        efficiency_ctx = getattr(self.get_model(), "efficiency_ctx", None)
        if efficiency_ctx is not None and efficiency_ctx.vit_start_event is None:
            efficiency_ctx.vit_start_event = torch.cuda.Event(enable_timing=True)
            efficiency_ctx.vit_start_event.record()
        result = self.get_model().get_vision_tower()(images, return_vit_attn=True)
        if efficiency_ctx is not None and efficiency_ctx.vit_end_event is None:
            efficiency_ctx.vit_end_event = torch.cuda.Event(enable_timing=True)
            efficiency_ctx.vit_end_event.record()
            efficiency_ctx.projector_start_event = torch.cuda.Event(enable_timing=True)
            efficiency_ctx.projector_start_event.record()
        if isinstance(result, tuple):
            image_features, vit_attn = result
        else:
            image_features = result
            vit_attn = None
        image_features = self.get_model().mm_projector(image_features)
        if efficiency_ctx is not None and efficiency_ctx.projector_end_event is None:
            efficiency_ctx.projector_end_event = torch.cuda.Event(enable_timing=True)
            efficiency_ctx.projector_end_event.record()
        return image_features, vit_attn

    def encode_images_with_vit_attn_and_semantic_layer(self, images, semantic_layer=None):
        """Encode images and optionally project a separate ViT layer for text-semantic routing."""
        if semantic_layer is None:
            image_features, vit_attn = self.encode_images_with_vit_attn(images)
            return image_features, vit_attn, None

        efficiency_ctx = getattr(self.get_model(), "efficiency_ctx", None)
        if efficiency_ctx is not None and efficiency_ctx.vit_start_event is None:
            efficiency_ctx.vit_start_event = torch.cuda.Event(enable_timing=True)
            efficiency_ctx.vit_start_event.record()

        vision_tower = self.get_model().get_vision_tower()
        image_forward_outs = vision_tower.vision_tower(
            images.to(device=vision_tower.device, dtype=vision_tower.dtype),
            output_hidden_states=True,
            output_attentions=True,
        )

        if efficiency_ctx is not None and efficiency_ctx.vit_end_event is None:
            efficiency_ctx.vit_end_event = torch.cuda.Event(enable_timing=True)
            efficiency_ctx.vit_end_event.record()
            efficiency_ctx.projector_start_event = torch.cuda.Event(enable_timing=True)
            efficiency_ctx.projector_start_event.record()

        image_features = vision_tower.feature_select(image_forward_outs).to(images.dtype)
        vit_attn = None
        if image_forward_outs.attentions is not None:
            vit_attn = image_forward_outs.attentions[vision_tower.select_layer]

        selected_layer = _resolve_vit_hidden_state_index(
            vision_tower.select_layer,
            num_hidden_states=len(image_forward_outs.hidden_states),
        )
        resolved_layer = _resolve_vit_hidden_state_index(
            semantic_layer,
            num_hidden_states=len(image_forward_outs.hidden_states),
        )
        semantic_features = image_forward_outs.hidden_states[resolved_layer]
        if vision_tower.select_feature == "patch":
            semantic_features = semantic_features[:, 1:]
        elif vision_tower.select_feature != "cls_patch":
            raise ValueError(f"Unexpected select feature: {vision_tower.select_feature}")

        projector = self.get_model().mm_projector
        projector_param = next(projector.parameters())
        image_features = projector(image_features.to(device=projector_param.device, dtype=projector_param.dtype))
        if resolved_layer == selected_layer:
            semantic_features = image_features
        else:
            semantic_features = projector(semantic_features.to(device=projector_param.device, dtype=projector_param.dtype))

        if efficiency_ctx is not None and efficiency_ctx.projector_end_event is None:
            efficiency_ctx.projector_end_event = torch.cuda.Event(enable_timing=True)
            efficiency_ctx.projector_end_event.record()
        return image_features, vit_attn, semantic_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        # Adapter training with image start/end tokens is unsupported here.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Materialize optional sequence tensors for assembly, then restore each
        # caller-visible None value before returning.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # Remove padded tokens before multimodal sequence assembly.
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels


    def prepare_sparse_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        """Assemble multimodal inputs after order-preserving Stage-1 routing.

        Routing text is the valid prompt suffix after the first image marker.
        Score Top-K and disjoint FPS select visual tokens before text and image
        embeddings are interleaved; returned metadata identifies the routed
        visual span for decoder phase pruning.
        """
        vision_tower = self.get_vision_tower()
        self.get_model().trevs_route_stats = None
        efficiency_ctx = getattr(self.get_model(), "efficiency_ctx", None)
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            self.model.generate_process_count += 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels,self.image_shape,self.token_length_list,self.pre_prompt_length_list

        if efficiency_ctx is not None and efficiency_ctx.prefill_start_event is None:
            efficiency_ctx.prefill_start_event = torch.cuda.Event(enable_timing=True)
            efficiency_ctx.prefill_start_event.record()

        use_trevs = TREVS_ENABLED
        trevs_semantic_layer = _get_trevs_semantic_layer() if use_trevs else None
        vit_attn = None
        semantic_features = None
        routed_index_masks = None

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            if use_trevs:
                image_features, vit_attn, semantic_features = self.encode_images_with_vit_attn_and_semantic_layer(
                    concat_images,
                    semantic_layer=trevs_semantic_layer,
                )
            else:
                image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            if use_trevs and vit_attn is not None:
                device = image_features.device
                n_vis_input = int(image_features.shape[1])
                T_emb, text_mask_padded = _build_routing_text_embeddings(
                    self.get_model(),
                    input_ids,
                    attention_mask,
                    input_ids.shape[0],
                    device,
                )
                image_counts = _count_image_tokens_per_sample(input_ids, attention_mask, input_ids.shape[0])
                T_emb, text_mask_padded = _repeat_routing_text_by_image_splits(
                    T_emb,
                    text_mask_padded,
                    image_counts,
                    split_sizes,
                )
                if T_emb is not None and text_mask_padded is not None:
                    if TREVS_TOTAL_TOKENS > image_features.shape[1]:
                        raise ValueError(
                            f"TREVS_ROUTE_TOPK + TREVS_ROUTE_FPS = {TREVS_TOTAL_TOKENS} exceeds "
                            f"the number of visual tokens {image_features.shape[1]}."
                        )
                    if efficiency_ctx is not None and efficiency_ctx.route_start_event is None:
                        efficiency_ctx.route_start_event = torch.cuda.Event(enable_timing=True)
                        efficiency_ctx.route_start_event.record()
                    idx_routed, route_stats = trevs_route(
                        vit_attn=vit_attn,
                        V_proj=image_features,
                        T_emb=T_emb,
                        k_track1=TREVS_ROUTE_TOPK,
                        k_track2=TREVS_ROUTE_FPS,
                        core_token_mask=text_mask_padded,
                        V_semantic_proj=semantic_features,
                        semantic_layer=trevs_semantic_layer,
                        timing_enabled=efficiency_ctx is not None,
                        return_stats=True,
                    )
                    if efficiency_ctx is not None and efficiency_ctx.route_end_event is None:
                        efficiency_ctx.route_end_event = torch.cuda.Event(enable_timing=True)
                        efficiency_ctx.route_end_event.record()
                    routed_index_masks = _index_mask_from_indices(idx_routed, image_features.shape[1])
                    _store_trevs_route_stats(self.get_model(), idx_routed, route_stats)
                    if efficiency_ctx is not None:
                        efficiency_ctx.route_k1_start_event = route_stats.get("route_k1_start_event")
                        efficiency_ctx.route_k1_end_event = route_stats.get("route_k1_end_event")
                        efficiency_ctx.route_k2_start_event = route_stats.get("route_k2_start_event")
                        efficiency_ctx.route_k2_end_event = route_stats.get("route_k2_end_event")
                        efficiency_ctx.metrics["routing_text_len"] = int(text_mask_padded[0].sum().item())
                        efficiency_ctx.metrics["n_vis_input"] = n_vis_input
                        efficiency_ctx.metrics["n_vis_routed"] = int(TREVS_TOTAL_TOKENS)
                        efficiency_ctx.metrics["trevs_route_topk"] = int(TREVS_ROUTE_TOPK)
                        efficiency_ctx.metrics["trevs_route_fps"] = int(TREVS_ROUTE_FPS)
                        efficiency_ctx.metrics["trevs_semantic_layer"] = (
                            int(trevs_semantic_layer) if trevs_semantic_layer is not None else None
                        )
            image_features = torch.split(image_features, split_sizes, dim=0)
            if routed_index_masks is not None:
                routed_index_masks = torch.split(routed_index_masks, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            if use_trevs:
                # Preserve the exact per-crop route budget; unpadding would drop selected padding positions.
                mm_patch_merge_type = mm_patch_merge_type.replace('_unpad', '')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                if routed_index_masks is None:
                    image_features = [x.flatten(0, 1) for x in image_features]
                else:
                    image_features = [x.flatten(0, 1)[m.flatten(0, 1)] for x, m in zip(image_features, routed_index_masks)]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    index_mask = routed_index_masks[image_idx] if routed_index_masks is not None else None
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        base_index_mask = index_mask[0] if index_mask is not None else None
                        image_feature = image_feature[1:]
                        if index_mask is not None:
                            index_mask = index_mask[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                            if index_mask is not None:
                                index_mask = index_mask.view(num_patch_height, num_patch_width, height, width)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            if index_mask is not None:
                                index_mask = index_mask.permute(0, 2, 1, 3).contiguous().unsqueeze(0)
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            if index_mask is not None:
                                index_mask = index_mask.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            if index_mask is not None:
                                index_mask = unpad_image(index_mask, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            if index_mask is not None:
                                index_mask = torch.cat((
                                    index_mask,
                                    torch.ones(*index_mask.shape[:-1], 1, dtype=torch.bool, device=index_mask.device)
                                ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                            if index_mask is not None:
                                index_mask = index_mask.flatten(1, 2).squeeze(0)
                                image_feature = image_feature[index_mask]
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            if index_mask is not None:
                                index_mask = index_mask.permute(0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                            if index_mask is not None:
                                index_mask = index_mask.flatten(0, 3)
                                image_feature = image_feature[index_mask]
                        if base_index_mask is not None:
                            base_image_feature = base_image_feature[base_index_mask]
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if index_mask is not None:
                            index_mask = index_mask[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                            if index_mask is not None:
                                index_mask = torch.cat((
                                    index_mask,
                                    torch.ones(1, dtype=torch.bool, device=index_mask.device)
                                ), dim=0)
                        if index_mask is not None:
                            image_feature = image_feature[index_mask]
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            if use_trevs:
                image_features, vit_attn, semantic_features = self.encode_images_with_vit_attn_and_semantic_layer(
                    images,
                    semantic_layer=trevs_semantic_layer,
                )
            else:
                image_features = self.encode_images(images)

        # Adapter training with image start/end tokens is unsupported here.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Materialize optional sequence tensors for assembly, then restore each
        # caller-visible None value before returning.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # Remove padded tokens before routing and multimodal sequence assembly.
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        
        # Stage 1 embeds the complete valid prompt suffix after the first image
        # marker for semantic evidence. Token counts come from the active preset.
        if use_trevs and vit_attn is not None and not isinstance(image_features, list):
            device = image_features.device
            n_vis_input = int(image_features.shape[1])
            T_emb, text_mask_padded = _build_routing_text_embeddings(
                self.get_model(),
                input_ids,
                None,
                len(input_ids),
                device,
            )
            if T_emb is not None and text_mask_padded is not None:
                if TREVS_TOTAL_TOKENS > image_features.shape[1]:
                    raise ValueError(
                        f"TREVS_ROUTE_TOPK + TREVS_ROUTE_FPS = {TREVS_TOTAL_TOKENS} exceeds "
                        f"the number of visual tokens {image_features.shape[1]}."
                    )
                if efficiency_ctx is not None and efficiency_ctx.route_start_event is None:
                    efficiency_ctx.route_start_event = torch.cuda.Event(enable_timing=True)
                    efficiency_ctx.route_start_event.record()
                idx_routed, route_stats = trevs_route(
                    vit_attn=vit_attn,
                    V_proj=image_features,
                    T_emb=T_emb,
                    k_track1=TREVS_ROUTE_TOPK,
                    k_track2=TREVS_ROUTE_FPS,
                    core_token_mask=text_mask_padded,
                    V_semantic_proj=semantic_features,
                    semantic_layer=trevs_semantic_layer,
                    timing_enabled=efficiency_ctx is not None,
                    return_stats=True,
                )
                if efficiency_ctx is not None and efficiency_ctx.route_end_event is None:
                    efficiency_ctx.route_end_event = torch.cuda.Event(enable_timing=True)
                    efficiency_ctx.route_end_event.record()
                image_features = route_stats["V_routed"]
                _store_trevs_route_stats(self.get_model(), idx_routed, route_stats)
                if efficiency_ctx is not None:
                    efficiency_ctx.route_k1_start_event = route_stats.get("route_k1_start_event")
                    efficiency_ctx.route_k1_end_event = route_stats.get("route_k1_end_event")
                    efficiency_ctx.route_k2_start_event = route_stats.get("route_k2_start_event")
                    efficiency_ctx.route_k2_end_event = route_stats.get("route_k2_end_event")
                    efficiency_ctx.metrics["routing_text_len"] = int(text_mask_padded[0].sum().item())
                    efficiency_ctx.metrics["n_vis_input"] = n_vis_input
                    efficiency_ctx.metrics["n_vis_routed"] = int(image_features.shape[1])
                    efficiency_ctx.metrics["trevs_route_topk"] = int(TREVS_ROUTE_TOPK)
                    efficiency_ctx.metrics["trevs_route_fps"] = int(TREVS_ROUTE_FPS)
                    efficiency_ctx.metrics["trevs_semantic_layer"] = (
                        int(trevs_semantic_layer) if trevs_semantic_layer is not None else None
                    )

        new_input_embeds = []  
        new_labels = []
        cur_image_idx = 0
        pre_prompt_length_list = []     
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            # Sentinels bound every text segment around the image placeholders.
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]] 
            # Stage 2 uses the first image position as the visual-span offset.
            pre_prompt_length_list.append(image_token_indices[1])
            cur_input_ids_noim = [] 
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim)) 
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0) 
            cur_new_input_embeds = [] 
            cur_new_labels = []   
            # Reconstruct the prompt by interleaving text segments and routed vision tokens.
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features) 
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]    

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)  
            cur_new_labels = torch.cat(cur_new_labels) 

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Pad the assembled samples to a common batch length.
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)
        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        token_length_list = []  
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            token_length_list.append(cur_len)
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)  # [B, L, D]

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if isinstance(image_features, list):
            image_shape = int(image_features[0].shape[0]) if len(image_features) > 0 else 0
        else:
            image_shape = int(image_features.shape[1])
        self.image_shape = image_shape
        self.token_length_list = token_length_list
        self.pre_prompt_length_list = pre_prompt_length_list
        self.model.init_token_total_shape = max_len      
        if efficiency_ctx is not None:
            efficiency_ctx.metrics["prefill_seq_len_routed"] = int(max_len)
            efficiency_ctx.metrics.setdefault("n_vis_input", image_shape)
            efficiency_ctx.metrics["n_vis_routed"] = image_shape
            efficiency_ctx.metrics.setdefault("routing_text_len", 0)
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels,image_shape,token_length_list,pre_prompt_length_list

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
