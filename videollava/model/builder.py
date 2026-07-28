"""Strict local checkpoint loader for the isolated Video-LLaVA TReVS model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoTokenizer

from videollava.constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
)

from .language_model.sparse_videollava_llama import (
    VideoLlavaConfig,
    VideoLlavaTReVSForCausalLM,
)


SUPPORTED_ATTENTION_IMPLEMENTATIONS = ("sdpa", "flash_attention_2")


def _validate_loading_info(loading_info: dict) -> None:
    missing = loading_info.get("missing_keys", [])
    mismatched = loading_info.get("mismatched_keys", [])
    unexpected = loading_info.get("unexpected_keys", [])
    disallowed_unexpected = [
        key
        for key in unexpected
        if not key.startswith("model.image_tower.")
        and not key.endswith(".self_attn.rotary_emb.inv_freq")
    ]
    if missing or mismatched or disallowed_unexpected:
        raise RuntimeError(
            "Video-LLaVA checkpoint does not exactly cover the isolated inference model: "
            f"missing={missing[:8]}, mismatched={mismatched[:8]}, "
            f"unexpected={disallowed_unexpected[:8]}."
        )


def load_pretrained_model(
    model_path: str,
    device: str = "cuda:0",
    attn_implementation: str = "sdpa",
) -> Tuple[object, VideoLlavaTReVSForCausalLM, object, int]:
    checkpoint = Path(model_path).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Video-LLaVA checkpoint directory does not exist: {checkpoint}")
    if attn_implementation not in SUPPORTED_ATTENTION_IMPLEMENTATIONS:
        raise ValueError(
            f"Unsupported attention implementation {attn_implementation!r}; "
            f"expected one of {SUPPORTED_ATTENTION_IMPLEMENTATIONS}."
        )
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for Video-LLaVA but no CUDA device is available.")

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        use_fast=False,
        local_files_only=True,
    )
    with (checkpoint / "config.json").open("r", encoding="utf-8") as handle:
        config_values = json.load(handle)
    config_values.pop("model_type", None)
    config_values.pop("architectures", None)
    config = VideoLlavaConfig(**config_values)
    config._name_or_path = str(checkpoint)
    config._attn_implementation = attn_implementation
    config.tokenizer_padding_side = "right"

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model, loading_info = VideoLlavaTReVSForCausalLM.from_pretrained(
        checkpoint,
        config=config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map={"": device},
        local_files_only=True,
        output_loading_info=True,
    )
    _validate_loading_info(loading_info)

    if getattr(config, "mm_use_im_patch_token", False):
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if getattr(config, "mm_use_im_start_end", False):
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    video_tower = model.get_video_tower()
    if hasattr(model.get_model(), "image_tower") and model.get_model().image_tower is not None:
        raise RuntimeError("Video-only loading unexpectedly instantiated an image tower.")
    video_tower.load_model()
    model.eval()
    context_length = int(
        getattr(config, "max_sequence_length", config.max_position_embeddings)
    )
    return tokenizer, model, video_tower.video_processor, context_length


__all__ = ["load_pretrained_model"]
