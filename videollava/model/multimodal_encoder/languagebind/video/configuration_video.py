"""Configuration for the Video-LLaVA LanguageBind video encoder."""

from __future__ import annotations

import copy
from typing import Mapping, Optional

from transformers import CLIPVisionConfig as TransformersCLIPVisionConfig
from transformers import PretrainedConfig


# Video-LLaVA-7B embeds weights for this exact LanguageBind ViT-L/14 video
# architecture.  Keeping the values here avoids a network request for the
# original LanguageBind repository during inference.
OFFICIAL_VIDEO_TOWER_CONFIG = {
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "projection_dim": 768,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "num_channels": 3,
    "image_size": 224,
    "patch_size": 14,
    "hidden_act": "gelu",
    "layer_norm_eps": 1e-5,
    "attention_dropout": 0.0,
    "initializer_range": 0.02,
    "initializer_factor": 1.0,
    "add_time_attn": True,
    "num_frames": 8,
    "force_patch_dropout": 0.0,
    "video_decode_backend": "decord",
}


class CLIPVisionConfig(TransformersCLIPVisionConfig):
    """CLIP vision config extended with LanguageBind temporal attention."""

    model_type = "languagebind_video_vision_model"

    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
        projection_dim: int = 768,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        num_channels: int = 3,
        image_size: int = 224,
        patch_size: int = 14,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-5,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        initializer_factor: float = 1.0,
        add_time_attn: bool = True,
        num_frames: int = 8,
        force_patch_dropout: float = 0.0,
        video_decode_backend: str = "decord",
        **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            projection_dim=projection_dim,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_channels=num_channels,
            image_size=image_size,
            patch_size=patch_size,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            attention_dropout=attention_dropout,
            initializer_range=initializer_range,
            initializer_factor=initializer_factor,
            **kwargs,
        )
        self.add_time_attn = add_time_attn
        self.num_frames = num_frames
        self.force_patch_dropout = force_patch_dropout
        self.video_decode_backend = video_decode_backend


class LanguageBindVideoConfig(PretrainedConfig):
    """Minimal composition config retained for processor compatibility."""

    model_type = "LanguageBindVideo"
    is_composition = True

    def __init__(
        self,
        vision_config: Optional[Mapping] = None,
        projection_dim: int = 768,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vision_config = CLIPVisionConfig(**dict(vision_config or {}))
        self.projection_dim = projection_dim
        self.initializer_factor = 1.0

    def to_dict(self) -> dict:
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict()
        output["model_type"] = self.model_type
        return output


def validate_official_video_config(config: CLIPVisionConfig) -> None:
    """Fail early if a production tower cannot match embedded checkpoint keys."""

    mismatches = []
    for field, expected in OFFICIAL_VIDEO_TOWER_CONFIG.items():
        actual = getattr(config, field, None)
        if actual != expected:
            mismatches.append(f"{field}={actual!r} (expected {expected!r})")
    if mismatches:
        raise ValueError("Invalid Video-LLaVA video tower config: " + ", ".join(mismatches))


def build_official_video_config() -> CLIPVisionConfig:
    config = CLIPVisionConfig(**OFFICIAL_VIDEO_TOWER_CONFIG)
    validate_official_video_config(config)
    return config
