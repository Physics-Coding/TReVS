"""Checkpoint-compatible LanguageBind spatiotemporal vision transformer."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import CLIPAttention, CLIPMLP, CLIPVisionEmbeddings

from .configuration_video import (
    CLIPVisionConfig,
    build_official_video_config,
    validate_official_video_config,
)
from .processing_video import LanguageBindVideoProcessor


class CLIPEncoderLayer(nn.Module):
    """One temporal-attention block followed by the original spatial block."""

    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        self.add_time_attn = config.add_time_attn
        self.num_frames = config.num_frames
        if self.add_time_attn:
            self.temporal_embedding = nn.Parameter(
                torch.zeros(1, config.num_frames, config.hidden_size)
            )
            nn.init.normal_(self.temporal_embedding, std=config.hidden_size**-0.5)
            self.temporal_attn = CLIPAttention(config)
            self.temporal_layer_norm1 = nn.LayerNorm(
                self.embed_dim, eps=config.layer_norm_eps
            )

    @staticmethod
    def _to_temporal(hidden_states: torch.Tensor, num_frames: int) -> Tuple[torch.Tensor, int, int]:
        batch_times_frames, num_tokens, width = hidden_states.shape
        if batch_times_frames % num_frames:
            raise ValueError(
                f"Flattened frame batch {batch_times_frames} is not divisible by {num_frames}"
            )
        batch_size = batch_times_frames // num_frames
        temporal = hidden_states.reshape(batch_size, num_frames, num_tokens, width)
        temporal = temporal.permute(0, 2, 1, 3).reshape(batch_size * num_tokens, num_frames, width)
        return temporal, batch_size, num_tokens

    @staticmethod
    def _from_temporal(
        hidden_states: torch.Tensor,
        batch_size: int,
        num_tokens: int,
        num_frames: int,
    ) -> torch.Tensor:
        width = hidden_states.shape[-1]
        spatial = hidden_states.reshape(batch_size, num_tokens, num_frames, width)
        return spatial.permute(0, 2, 1, 3).reshape(batch_size * num_frames, num_tokens, width)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        if self.add_time_attn:
            temporal, batch_size, num_tokens = self._to_temporal(
                hidden_states, self.num_frames
            )
            temporal = temporal + self.temporal_embedding[:, : self.num_frames]
            residual = self._from_temporal(
                temporal, batch_size, num_tokens, self.num_frames
            )
            temporal = self.temporal_layer_norm1(temporal)
            temporal = self.temporal_attn(
                hidden_states=temporal,
                attention_mask=None,
                causal_attention_mask=None,
                output_attentions=False,
            )[0]
            hidden_states = residual + self._from_temporal(
                temporal, batch_size, num_tokens, self.num_frames
            )

        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, spatial_attention, *_ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)

        if output_attentions:
            return hidden_states, spatial_attention
        return (hidden_states,)


class CLIPEncoder(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = (
            self.config.output_attentions if output_attentions is None else output_attentions
        )
        output_hidden_states = (
            self.config.output_hidden_states
            if output_hidden_states is None
            else output_hidden_states
        )
        return_dict = self.config.use_return_dict if return_dict is None else return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states += (hidden_states,)
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions += (layer_outputs[1],)

        if output_hidden_states:
            encoder_states += (hidden_states,)

        if not return_dict:
            return tuple(
                item for item in (hidden_states, encoder_states, all_attentions) if item is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


class CLIPVisionTransformer(nn.Module):
    """The exact module hierarchy stored under ``model.video_tower.video_tower``."""

    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    @staticmethod
    def _reshape_frames(hidden_states: torch.Tensor, batch_size: int, num_frames: int) -> torch.Tensor:
        return hidden_states.reshape(batch_size, num_frames, *hidden_states.shape[1:])

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        if pixel_values is None:
            raise ValueError("pixel_values must be provided")
        if pixel_values.ndim != 5:
            raise ValueError(
                "LanguageBind video input must have shape [B, C, T, H, W], "
                f"got {tuple(pixel_values.shape)}"
            )
        batch_size, channels, num_frames, height, width = pixel_values.shape
        if channels != self.config.num_channels:
            raise ValueError(f"Expected {self.config.num_channels} channels, got {channels}")
        if num_frames != self.config.num_frames:
            raise ValueError(f"Expected {self.config.num_frames} frames, got {num_frames}")
        if (height, width) != (self.config.image_size, self.config.image_size):
            raise ValueError(
                f"Expected {self.config.image_size}x{self.config.image_size}, got {height}x{width}"
            )

        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        flattened = pixel_values.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames, channels, height, width
        )
        hidden_states = self.pre_layrnorm(self.embeddings(flattened))
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        last_hidden_state = self._reshape_frames(
            encoder_outputs.last_hidden_state, batch_size, num_frames
        )
        pooled_output = self.post_layernorm(
            encoder_outputs.last_hidden_state[:, 0, :]
        ).reshape(batch_size, num_frames, -1).mean(dim=1)
        reshaped_hidden_states = None
        if encoder_outputs.hidden_states is not None:
            reshaped_hidden_states = tuple(
                self._reshape_frames(state, batch_size, num_frames)
                for state in encoder_outputs.hidden_states
            )

        if not return_dict:
            output = (last_hidden_state, pooled_output)
            if reshaped_hidden_states is not None:
                output += (reshaped_hidden_states,)
            if encoder_outputs.attentions is not None:
                output += (encoder_outputs.attentions,)
            return output

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=reshaped_hidden_states,
            attentions=encoder_outputs.attentions,
        )


class LanguageBindVideoTower(nn.Module):
    """Video-LLaVA wrapper whose nested name matches the local 7B checkpoint."""

    def __init__(self, video_tower, args, delay_load: bool = False, cache_dir=None):
        super().__init__()
        del delay_load, cache_dir
        self.video_tower_name = str(video_tower)
        self.select_layer = int(getattr(args, "mm_vision_select_layer", -2))
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")
        if self.select_feature not in ("patch", "cls_patch"):
            raise ValueError(f"Unsupported mm_vision_select_feature: {self.select_feature}")

        config = build_official_video_config()
        mm_hidden_size = getattr(args, "mm_hidden_size", config.hidden_size)
        if mm_hidden_size != config.hidden_size:
            raise ValueError(
                f"Checkpoint mm_hidden_size={mm_hidden_size} does not match video tower "
                f"hidden_size={config.hidden_size}"
            )
        if not -config.num_hidden_layers <= self.select_layer < config.num_hidden_layers:
            raise ValueError(f"Invalid mm_vision_select_layer: {self.select_layer}")

        # This child must exist during the outer model's from_pretrained call so
        # embedded model.video_tower.video_tower.* tensors are not discarded.
        self.video_tower = CLIPVisionTransformer(config)
        self.video_tower.requires_grad_(False)
        self.video_processor = LanguageBindVideoProcessor(config)
        self.is_loaded = True

    def load_model(self, device_map=None) -> None:
        """Confirm local initialization; never download a second tower."""

        del device_map
        validate_official_video_config(self.video_tower.config)
        self.is_loaded = True

    def feature_select(self, video_forward_outs, layer: Optional[int] = None) -> torch.Tensor:
        if video_forward_outs.hidden_states is None:
            raise ValueError("Video tower output_hidden_states must be enabled")
        return video_forward_outs.hidden_states[self.select_layer if layer is None else layer]

    @torch.no_grad()
    def forward(self, videos, return_vit_attn: bool = False):
        if isinstance(videos, list):
            features = []
            attentions = []
            for video in videos:
                result = self.forward(video.unsqueeze(0), return_vit_attn=return_vit_attn)
                if return_vit_attn:
                    feature, attention = result
                    features.append(feature)
                    attentions.append(attention)
                else:
                    features.append(result)
            return (features, attentions) if return_vit_attn else features

        original_dtype = videos.dtype
        outputs = self.video_tower(
            videos.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True,
            output_attentions=return_vit_attn,
            return_dict=True,
        )
        features = self.feature_select(outputs).to(dtype=original_dtype)
        if not return_vit_attn:
            return features
        if outputs.attentions is None:
            raise RuntimeError("Video attention output was requested but not returned")
        return features, outputs.attentions[self.select_layer]

    @property
    def dummy_feature(self) -> torch.Tensor:
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self) -> torch.dtype:
        return self.video_tower.embeddings.class_embedding.dtype

    @property
    def device(self) -> torch.device:
        return self.video_tower.embeddings.class_embedding.device

    @property
    def config(self) -> CLIPVisionConfig:
        return self.video_tower.config

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    @property
    def num_patches_per_side(self) -> int:
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self) -> int:
        return self.num_patches_per_side**2
