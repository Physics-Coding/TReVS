"""Independent Video-LLaVA language-model implementation for TReVS."""

from .score import (
    METHOD,
    TREVS_ENABLED,
    TREVS_ROUTE_FPS,
    TREVS_ROUTE_TOPK,
    TREVS_TOTAL_TOKENS,
)

__all__ = [
    "METHOD",
    "TREVS_ENABLED",
    "TREVS_ROUTE_FPS",
    "TREVS_ROUTE_TOPK",
    "TREVS_TOTAL_TOKENS",
]
