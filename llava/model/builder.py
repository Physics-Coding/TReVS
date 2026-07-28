"""Load the LLaMA/Vicuna LLaVA inference models supported by this package."""

from __future__ import annotations

import ast
import json
import logging
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

from llava.constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
)
from llava.model import LlavaLlamaDynamicForCausalLM, LlavaLlamaForCausalLM


LOGGER = logging.getLogger(__name__)
SUPPORTED_CHECKPOINT_MODEL_TYPE = "llava"
SUPPORTED_ARCHITECTURE = "LlavaLlamaForCausalLM"


def _maybe_override_image_grid_pinpoints(model) -> None:
    raw_pinpoints = os.getenv("LLAVA_IMAGE_GRID_PINPOINTS", "").strip()
    if not raw_pinpoints:
        return
    try:
        image_grid_pinpoints = ast.literal_eval(raw_pinpoints)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(
            f"Invalid LLAVA_IMAGE_GRID_PINPOINTS={raw_pinpoints!r}; "
            "expected a list like [[672, 672]]."
        ) from exc
    if not isinstance(image_grid_pinpoints, (list, tuple)):
        raise ValueError("LLAVA_IMAGE_GRID_PINPOINTS must be a list of [width, height] pairs.")
    normalized_pinpoints = []
    for pinpoint in image_grid_pinpoints:
        if not isinstance(pinpoint, (list, tuple)) or len(pinpoint) != 2:
            raise ValueError("LLAVA_IMAGE_GRID_PINPOINTS must contain [width, height] pairs.")
        width, height = int(pinpoint[0]), int(pinpoint[1])
        if width <= 0 or height <= 0:
            raise ValueError("Image grid dimensions must be positive.")
        normalized_pinpoints.append([width, height])
    model.config.image_grid_pinpoints = normalized_pinpoints
    LOGGER.info("Overrode image_grid_pinpoints with %s", normalized_pinpoints)


def _validate_weight_layout(model_path: Path) -> None:
    index_path = next(
        (
            model_path / name
            for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json")
            if (model_path / name).is_file()
        ),
        None,
    )
    if index_path is not None:
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed checkpoint weight index: {index_path}") from exc
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Checkpoint weight index has no nonempty weight_map: {index_path}")
        missing = sorted(
            shard
            for shard in {str(value) for value in weight_map.values()}
            if not (model_path / shard).is_file()
        )
        if missing:
            raise ValueError(f"Checkpoint is missing {len(missing)} indexed weight shard(s): {missing[:5]}")
        return
    if any((model_path / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
        return
    raise ValueError(
        "This inference-only package requires a complete checkpoint with a weight index and all "
        "shards, or a complete monolithic model.safetensors/pytorch_model.bin file."
    )


def _validate_supported_checkpoint(model_path: str, model_base: str | None) -> dict:
    if model_base is not None:
        raise ValueError(
            "This inference-only package accepts complete LLaVA checkpoints; "
            "projector-only, LoRA, and base-model merge loading are not supported."
        )
    checkpoint = Path(model_path)
    config_path = checkpoint / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read a valid checkpoint config: {config_path}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Checkpoint config must be a JSON object: {config_path}")
    model_type = config.get("model_type")
    architectures = config.get("architectures")
    if model_type != SUPPORTED_CHECKPOINT_MODEL_TYPE or architectures != [SUPPORTED_ARCHITECTURE]:
        raise ValueError(
            "Unsupported checkpoint architecture. Expected a full LLaMA/Vicuna LLaVA "
            f"checkpoint with model_type={SUPPORTED_CHECKPOINT_MODEL_TYPE!r} and "
            f"architectures={[SUPPORTED_ARCHITECTURE]!r}; received "
            f"model_type={model_type!r}, architectures={architectures!r}."
        )
    if config.get("s2", False):
        raise ValueError("S2 vision towers are not supported by this reproducibility package.")
    _validate_weight_layout(checkpoint)
    return config


def load_pretrained_model(
    model_path,
    model_base,
    model_name,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    use_flash_attn=False,
    dynamic_sparse=True,
    **kwargs,
):
    del model_name
    model_path = os.path.expanduser(model_path)
    raw_config = _validate_supported_checkpoint(model_path, model_base)
    load_kwargs = {"device_map": device_map, **kwargs}
    if device != "cuda":
        load_kwargs["device_map"] = {"": device}
    if load_8bit:
        load_kwargs["load_in_8bit"] = True
    elif load_4bit:
        load_kwargs["load_in_4bit"] = True
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        load_kwargs["torch_dtype"] = torch.float16
    if use_flash_attn:
        load_kwargs["attn_implementation"] = "flash_attention_2"

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model_class = LlavaLlamaDynamicForCausalLM if dynamic_sparse else LlavaLlamaForCausalLM
    config_values = dict(raw_config)
    config_values.pop("model_type", None)
    config = model_class.config_class.from_dict(config_values)
    LOGGER.info("Loading %s inference model", "sparse" if dynamic_sparse else "dense")
    model = model_class.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        config=config,
        **load_kwargs,
    )

    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    _maybe_override_image_grid_pinpoints(model)

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device_map)
    if device_map != "auto":
        vision_tower.to(device=device_map, dtype=torch.float16)
    image_processor = vision_tower.image_processor
    setattr(model, "tokenizer", tokenizer)
    context_len = getattr(model.config, "max_sequence_length", 2048)
    return tokenizer, model, image_processor, context_len
