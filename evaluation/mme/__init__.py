"""MME answer conversion and official metric computation."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "EVAL_GROUPS",
    "MMEMetricsCalculator",
    "convert_answers",
    "load_ground_truth",
    "score_results",
]


def __getattr__(name: str):
    if name in {"EVAL_GROUPS", "MMEMetricsCalculator", "score_results"}:
        return getattr(import_module(".calculation", __name__), name)
    if name in {"convert_answers", "load_ground_truth"}:
        return getattr(import_module(".convert_answer_to_mme", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
