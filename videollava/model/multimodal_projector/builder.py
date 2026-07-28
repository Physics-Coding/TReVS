"""Checkpoint-compatible multimodal projector factory."""

from __future__ import annotations

import re

from torch import nn


class IdentityMap(nn.Module):
    def forward(self, value, *args, **kwargs):
        return value

    @property
    def config(self) -> dict:
        return {"mm_projector_type": "identity"}


class SimpleResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)
        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, value):
        return value + self.proj(self.pre_norm(value))


def build_vision_projector(config, delay_load: bool = False, **kwargs) -> nn.Module:
    del delay_load, kwargs
    projector_type = getattr(config, "mm_projector_type", "linear")
    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_match = re.fullmatch(r"mlp(\d+)x_gelu", projector_type)
    if mlp_match:
        depth = int(mlp_match.group(1))
        if depth < 1:
            raise ValueError(f"MLP projector depth must be positive: {projector_type}")
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, depth):
            modules.extend((nn.GELU(), nn.Linear(config.hidden_size, config.hidden_size)))
        return nn.Sequential(*modules)

    if projector_type == "identity":
        return IdentityMap()
    raise ValueError(f"Unknown projector type: {projector_type}")
