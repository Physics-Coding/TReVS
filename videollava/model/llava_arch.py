"""Video-only multimodal assembly for the isolated Video-LLaVA TReVS stack."""

from __future__ import annotations

from abc import ABC, abstractmethod
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from videollava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    NUM_VIDEO_FRAMES,
    VIDEO_PATCHES_PER_FRAME,
)

from .language_model.score import (
    TREVS_ENABLED,
    TREVS_ROUTE_FPS,
    TREVS_ROUTE_TOPK,
    TREVS_TOTAL_TOKENS,
)
from .language_model.trevs_router import trevs_route
from .multimodal_encoder.builder import build_video_tower
from .multimodal_projector.builder import build_vision_projector


TREVS_SEMANTIC_LAYER_ENV = "TREVS_SEMANTIC_LAYER"


def get_trevs_semantic_layer() -> Optional[int]:
    raw = os.getenv(TREVS_SEMANTIC_LAYER_ENV, "").strip().lower()
    if not raw or raw in {"none", "off", "default", "current"}:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"Invalid {TREVS_SEMANTIC_LAYER_ENV}={raw!r}; expected an integer layer "
            "index or one of none/off/default/current."
        ) from exc


def _resolve_hidden_state_index(layer_idx: int, count: int) -> int:
    resolved = layer_idx + count if layer_idx < 0 else layer_idx
    if not 0 <= resolved < count:
        raise ValueError(
            f"{TREVS_SEMANTIC_LAYER_ENV}={layer_idx} resolves to hidden_states[{resolved}], "
            f"outside [0, {count - 1}]."
        )
    return resolved


def _normalize_video_batch(images) -> torch.Tensor:
    if torch.is_tensor(images):
        videos = images
        if videos.ndim == 4:
            videos = videos.unsqueeze(0)
    elif isinstance(images, (list, tuple)) and images:
        samples = []
        for video in images:
            if not torch.is_tensor(video):
                raise TypeError("Every video input must be a torch.Tensor.")
            if video.ndim == 5 and video.shape[0] == 1:
                video = video.squeeze(0)
            if video.ndim != 4:
                raise ValueError(
                    f"Each video must have shape [3, T, H, W], got {tuple(video.shape)}."
                )
            samples.append(video)
        videos = torch.stack(samples, dim=0)
    else:
        raise ValueError("Video-LLaVA requires one or more video tensors.")

    if videos.ndim != 5 or videos.shape[1] != 3:
        raise ValueError(f"Expected videos [B, 3, T, H, W], got {tuple(videos.shape)}.")
    if videos.shape[2] != NUM_VIDEO_FRAMES:
        raise ValueError(
            f"Video-LLaVA requires {NUM_VIDEO_FRAMES} frames, got {videos.shape[2]}."
        )
    if tuple(videos.shape[-2:]) != (224, 224):
        raise ValueError(f"Video frames must be 224x224, got {tuple(videos.shape[-2:])}.")
    return videos


def _extract_routing_text_ids(cur_input_ids: torch.Tensor) -> torch.Tensor:
    placeholder_positions = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
    if placeholder_positions.numel() == 0:
        return cur_input_ids[cur_input_ids >= 0]
    suffix = cur_input_ids[int(placeholder_positions[-1].item()) + 1 :]
    return suffix[suffix >= 0]


def build_routing_text_embeddings(
    embed_tokens: Callable[[torch.Tensor], torch.Tensor],
    input_ids: Sequence[torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    text_ids = [_extract_routing_text_ids(ids.to(device=device)) for ids in input_ids]
    lengths = [int(ids.numel()) for ids in text_ids]
    if not lengths or max(lengths) == 0:
        raise ValueError("TReVS routing requires at least one text token after the video placeholders.")
    padded = torch.zeros(len(text_ids), max(lengths), dtype=torch.long, device=device)
    mask = torch.zeros(len(text_ids), max(lengths), dtype=torch.bool, device=device)
    for batch_idx, ids in enumerate(text_ids):
        if ids.numel():
            padded[batch_idx, : ids.numel()] = ids
            mask[batch_idx, : ids.numel()] = True
    with torch.no_grad():
        embeddings = embed_tokens(padded)
    return embeddings, mask


def route_video_frames(
    vit_attn: torch.Tensor,
    projected_patches: torch.Tensor,
    text_embeddings: torch.Tensor,
    text_mask: torch.Tensor,
    semantic_projected_patches: Optional[torch.Tensor] = None,
    semantic_layer: Optional[int] = None,
    k_track1: int = TREVS_ROUTE_TOPK,
    k_track2: int = TREVS_ROUTE_FPS,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Apply the live LLaVA Stage-1 router independently to every video frame."""

    if projected_patches.ndim != 4:
        raise ValueError(
            "Projected video patches must have shape [B, T, N, D], got "
            f"{tuple(projected_patches.shape)}."
        )
    batch_size, frame_count, patch_count, hidden_size = projected_patches.shape
    if frame_count != NUM_VIDEO_FRAMES or patch_count != VIDEO_PATCHES_PER_FRAME:
        raise ValueError(
            f"Expected [B, {NUM_VIDEO_FRAMES}, {VIDEO_PATCHES_PER_FRAME}, D] patch features, "
            f"got {tuple(projected_patches.shape)}."
        )
    if vit_attn.ndim != 4 or vit_attn.shape[0] != batch_size * frame_count:
        raise ValueError(
            f"Expected one ViT attention matrix per frame, got {tuple(vit_attn.shape)}."
        )
    if tuple(vit_attn.shape[-2:]) != (
        VIDEO_PATCHES_PER_FRAME + 1,
        VIDEO_PATCHES_PER_FRAME + 1,
    ):
        raise ValueError(
            "Stage 1 requires full CLS+patch attention with shape [..., 257, 257], "
            f"got {tuple(vit_attn.shape)}."
        )
    if text_embeddings.shape[0] != batch_size or text_mask.shape[0] != batch_size:
        raise ValueError("Routing text batch size does not match the video batch size.")
    if k_track1 + k_track2 > patch_count:
        raise ValueError(
            f"Per-frame route budget {k_track1}+{k_track2} exceeds {patch_count} patches."
        )

    flattened = projected_patches.reshape(
        batch_size * frame_count, patch_count, hidden_size
    )
    repeated_text = torch.repeat_interleave(text_embeddings, frame_count, dim=0)
    repeated_mask = torch.repeat_interleave(text_mask, frame_count, dim=0)
    flattened_semantic = None
    if semantic_projected_patches is not None:
        if semantic_projected_patches.shape != projected_patches.shape:
            raise ValueError("Semantic-layer patch features must match projected patch features.")
        flattened_semantic = semantic_projected_patches.reshape_as(flattened)

    routed_indices, flat_stats = trevs_route(
        vit_attn=vit_attn,
        V_proj=flattened,
        T_emb=repeated_text,
        k_track1=k_track1,
        k_track2=k_track2,
        core_token_mask=repeated_mask,
        V_semantic_proj=flattened_semantic,
        semantic_layer=semantic_layer,
        return_stats=True,
    )
    tokens_per_frame = k_track1 + k_track2
    routed = flat_stats["V_routed"].reshape(
        batch_size, frame_count, tokens_per_frame, hidden_size
    )
    stats = dict(flat_stats)
    stats.update(
        {
            "idx_track1": flat_stats["idx_track1"].reshape(
                batch_size, frame_count, k_track1
            ),
            "idx_track2": flat_stats["idx_track2"].reshape(
                batch_size, frame_count, k_track2
            ),
            "idx_routed": routed_indices.reshape(
                batch_size, frame_count, tokens_per_frame
            ),
            "frame_count": frame_count,
            "patches_per_frame": patch_count,
            "tokens_per_frame": tokens_per_frame,
            "total_visual_tokens": frame_count * tokens_per_frame,
        }
    )
    return routed, stats


def assemble_video_sample(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    frame_features: torch.Tensor,
    embed_tokens: Callable[[torch.Tensor], torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, int, List[int]]:
    """Replace eight consecutive sentinels with eight ordered frame feature blocks."""

    if input_ids.ndim != 1 or labels.ndim != 1 or input_ids.shape != labels.shape:
        raise ValueError("Input IDs and labels must be equal-length rank-1 tensors.")
    if frame_features.ndim != 3 or frame_features.shape[0] != NUM_VIDEO_FRAMES:
        raise ValueError(
            f"Expected {NUM_VIDEO_FRAMES} frame feature blocks, got {tuple(frame_features.shape)}."
        )
    placeholder_positions = torch.where(input_ids == IMAGE_TOKEN_INDEX)[0]
    if placeholder_positions.numel() != NUM_VIDEO_FRAMES:
        raise ValueError(
            f"Expected {NUM_VIDEO_FRAMES} visual placeholders, got {placeholder_positions.numel()}."
        )
    if not torch.equal(
        placeholder_positions,
        torch.arange(
            int(placeholder_positions[0]),
            int(placeholder_positions[0]) + NUM_VIDEO_FRAMES,
            device=placeholder_positions.device,
        ),
    ):
        raise ValueError("The eight Video-LLaVA visual placeholders must be consecutive.")

    visual_start = int(placeholder_positions[0].item())
    frame_lengths = [int(frame.shape[0]) for frame in frame_features]
    if any(length <= 0 for length in frame_lengths):
        raise ValueError(f"Every frame must retain at least one visual token: {frame_lengths}.")

    text_before = input_ids[:visual_start]
    text_after = input_ids[visual_start + NUM_VIDEO_FRAMES :]
    label_before = labels[:visual_start]
    label_after = labels[visual_start + NUM_VIDEO_FRAMES :]
    text_ids = torch.cat((text_before, text_after), dim=0)
    text_embeddings = embed_tokens(text_ids)
    split = text_before.numel()
    before_embeddings = text_embeddings[:split].to(frame_features.device)
    after_embeddings = text_embeddings[split:].to(frame_features.device)
    visual_embeddings = frame_features.reshape(-1, frame_features.shape[-1])
    assembled = torch.cat((before_embeddings, visual_embeddings, after_embeddings), dim=0)
    visual_labels = torch.full(
        (sum(frame_lengths),),
        IGNORE_INDEX,
        dtype=labels.dtype,
        device=frame_features.device,
    )
    assembled_labels = torch.cat(
        (
            label_before.to(frame_features.device),
            visual_labels,
            label_after.to(frame_features.device),
        ),
        dim=0,
    )
    return assembled, assembled_labels, visual_start, frame_lengths


class LlavaMetaModel:
    def __init__(self, config):
        super().__init__(config)
        self.video_tower = None
        self.mm_projector = None
        if getattr(config, "mm_video_tower", None):
            self.video_tower = build_video_tower(config)
            self.mm_projector = build_vision_projector(config)

    def get_video_tower(self):
        if self.video_tower is None:
            raise RuntimeError("This model was created without a video tower.")
        return self.video_tower


class LlavaMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        raise NotImplementedError

    def get_video_tower(self):
        return self.get_model().get_video_tower()

    def _encode_video_patches(
        self, videos: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[int]]:
        tower = self.get_video_tower()
        if self.get_model().mm_projector is None:
            raise RuntimeError("This model was created without an mm_projector.")

        if TREVS_ENABLED:
            outputs = tower.video_tower(
                videos.to(device=tower.device, dtype=tower.dtype),
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
            )
            selected = tower.feature_select(outputs)
            if outputs.attentions is None:
                raise RuntimeError("The video tower did not return attention for TReVS routing.")
            vit_attn = outputs.attentions[tower.select_layer]
            semantic_layer = get_trevs_semantic_layer()
            semantic = None
            if semantic_layer is not None:
                resolved = _resolve_hidden_state_index(
                    semantic_layer, len(outputs.hidden_states)
                )
                semantic = outputs.hidden_states[resolved]
        else:
            selected = tower(videos, return_vit_attn=False)
            vit_attn = None
            semantic = None
            semantic_layer = None

        expected_shape = (
            videos.shape[0],
            NUM_VIDEO_FRAMES,
            VIDEO_PATCHES_PER_FRAME + 1,
        )
        if selected.ndim != 4 or tuple(selected.shape[:3]) != expected_shape:
            raise ValueError(
                f"Expected video hidden states {expected_shape}+[D], got {tuple(selected.shape)}."
            )

        projector = self.get_model().mm_projector
        patches = projector(selected[:, :, 1:, :].to(device=next(projector.parameters()).device))
        semantic_patches = None
        if semantic is not None:
            semantic_patches = projector(
                semantic[:, :, 1:, :].to(device=next(projector.parameters()).device)
            )
        return patches, vit_attn, semantic_patches, semantic_layer

    def _encode_and_route_videos(
        self, videos: torch.Tensor, input_ids: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        patches, vit_attn, semantic_patches, semantic_layer = self._encode_video_patches(videos)
        if not TREVS_ENABLED:
            stats = {
                "selection_mode": "dense",
                "frame_count": int(patches.shape[1]),
                "patches_per_frame": int(patches.shape[2]),
                "tokens_per_frame": int(patches.shape[2]),
                "total_visual_tokens": int(patches.shape[1] * patches.shape[2]),
            }
            return patches, stats

        text_embeddings, text_mask = build_routing_text_embeddings(
            self.get_model().embed_tokens,
            input_ids,
            patches.device,
        )
        return route_video_frames(
            vit_attn=vit_attn.to(device=patches.device),
            projected_patches=patches,
            text_embeddings=text_embeddings.to(device=patches.device),
            text_mask=text_mask.to(device=patches.device),
            semantic_projected_patches=semantic_patches,
            semantic_layer=semantic_layer,
        )

    def prepare_sparse_inputs_labels_for_multimodal(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values,
        labels: Optional[torch.Tensor],
        images,
        image_sizes=None,
    ):
        del position_ids, image_sizes
        if input_ids is None:
            raise ValueError("Multimodal assembly requires input token IDs.")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Formal Video-LLaVA inference requires token batch size 1.")
        if past_key_values is not None and input_ids.shape[1] == 1:
            return (
                input_ids,
                None,
                attention_mask,
                past_key_values,
                None,
                labels,
                getattr(self, "image_shape", 0),
                getattr(self, "token_length_list", []),
                getattr(self, "pre_prompt_length_list", []),
            )

        videos = _normalize_video_batch(images)
        if videos.shape[0] != input_ids.shape[0]:
            raise ValueError("The number of videos must match the token batch size.")
        source_mask = (
            attention_mask.to(device=input_ids.device, dtype=torch.bool)
            if attention_mask is not None
            else torch.ones_like(input_ids, dtype=torch.bool)
        )
        unpadded_ids = [ids[mask] for ids, mask in zip(input_ids, source_mask)]
        source_labels = labels if labels is not None else torch.full_like(input_ids, IGNORE_INDEX)
        unpadded_labels = [value[mask] for value, mask in zip(source_labels, source_mask)]

        frame_features, route_stats = self._encode_and_route_videos(videos, unpadded_ids)
        if frame_features.shape[0] != len(unpadded_ids):
            raise ValueError("The video tower returned a different batch size than the tokenizer.")
        assembled_embeddings = []
        assembled_labels = []
        visual_starts = []
        frame_lengths_by_sample = []
        for batch_idx in range(len(unpadded_ids)):
            embeddings, sample_labels, visual_start, frame_lengths = assemble_video_sample(
                unpadded_ids[batch_idx],
                unpadded_labels[batch_idx],
                frame_features[batch_idx],
                self.get_model().embed_tokens,
            )
            assembled_embeddings.append(embeddings)
            assembled_labels.append(sample_labels)
            visual_starts.append(visual_start)
            frame_lengths_by_sample.append(frame_lengths)

        image_shapes = [sum(lengths) for lengths in frame_lengths_by_sample]
        if len(set(image_shapes)) != 1:
            raise ValueError(f"Batch entries have different visual span lengths: {image_shapes}.")
        image_shape = image_shapes[0]
        expected_visual_tokens = int(route_stats["total_visual_tokens"])
        if image_shape != expected_visual_tokens:
            raise ValueError(
                f"Assembled visual span has {image_shape} tokens, expected {expected_visual_tokens}."
            )

        context_limit = int(
            getattr(
                self.config,
                "tokenizer_model_max_length",
                getattr(self.config, "max_position_embeddings", 0),
            )
            or 0
        )
        token_length_list = [int(value.shape[0]) for value in assembled_embeddings]
        if context_limit and max(token_length_list) > context_limit:
            raise ValueError(
                "Video token expansion exceeds the model context and cannot be truncated without "
                f"invalidating TReVS span metadata: {max(token_length_list)} > {context_limit}."
            )

        max_length = max(token_length_list)
        hidden_size = assembled_embeddings[0].shape[-1]
        embedding_device = assembled_embeddings[0].device
        padded_embeddings = torch.zeros(
            len(assembled_embeddings),
            max_length,
            hidden_size,
            dtype=assembled_embeddings[0].dtype,
            device=embedding_device,
        )
        padded_labels = torch.full(
            (len(assembled_labels), max_length),
            IGNORE_INDEX,
            dtype=assembled_labels[0].dtype,
            device=embedding_device,
        )
        rebuilt_mask = torch.zeros(
            len(assembled_embeddings), max_length, dtype=torch.long, device=embedding_device
        )
        rebuilt_positions = torch.zeros(
            len(assembled_embeddings), max_length, dtype=torch.long, device=embedding_device
        )
        padding_side = getattr(self.config, "tokenizer_padding_side", "right")
        for batch_idx, (embeddings, sample_labels) in enumerate(
            zip(assembled_embeddings, assembled_labels)
        ):
            length = embeddings.shape[0]
            if padding_side == "left":
                offset = max_length - length
                visual_starts[batch_idx] += offset
            elif padding_side == "right":
                offset = 0
            else:
                raise ValueError(f"Unsupported tokenizer_padding_side={padding_side!r}.")
            padded_embeddings[batch_idx, offset : offset + length] = embeddings
            padded_labels[batch_idx, offset : offset + length] = sample_labels
            rebuilt_mask[batch_idx, offset : offset + length] = 1
            rebuilt_positions[batch_idx, offset : offset + length] = torch.arange(
                length, device=embedding_device
            )

        self.image_shape = image_shape
        self.token_length_list = token_length_list
        self.pre_prompt_length_list = visual_starts
        self.frame_token_lengths = frame_lengths_by_sample
        self.get_model().trevs_route_stats = route_stats
        self.get_model().last_multimodal_metrics = {
            "frame_count": NUM_VIDEO_FRAMES,
            "frame_token_lengths": frame_lengths_by_sample[0],
            "n_vis_input": NUM_VIDEO_FRAMES * VIDEO_PATCHES_PER_FRAME,
            "n_vis_routed": image_shape,
            "visual_start": visual_starts[0],
            "text_start": visual_starts[0] + image_shape,
            "prefill_sequence_length": token_length_list[0],
        }
        return (
            None,
            rebuilt_positions,
            rebuilt_mask,
            past_key_values,
            padded_embeddings,
            None if labels is None else padded_labels,
            image_shape,
            token_length_list,
            visual_starts,
        )


__all__ = [
    "LlavaMetaForCausalLM",
    "LlavaMetaModel",
    "assemble_video_sample",
    "build_routing_text_embeddings",
    "get_trevs_semantic_layer",
    "route_video_frames",
]
