"""Factory for the local, checkpoint-backed LanguageBind video tower."""

from __future__ import annotations

from .languagebind import LanguageBindVideoTower


def build_video_tower(video_tower_cfg, **kwargs) -> LanguageBindVideoTower:
    tower_name = getattr(
        video_tower_cfg,
        "mm_video_tower",
        getattr(video_tower_cfg, "video_tower", None),
    )
    if not tower_name:
        raise ValueError("The model config does not define mm_video_tower")
    if not str(tower_name).endswith(("LanguageBind_Video", "LanguageBind_Video_merge")):
        raise ValueError(f"Unsupported video tower: {tower_name}")
    return LanguageBindVideoTower(tower_name, args=video_tower_cfg, **kwargs)


def build_image_tower(*args, **kwargs):
    raise RuntimeError("This isolated Video-LLaVA package intentionally does not build an image tower")
