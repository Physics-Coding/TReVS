"""Qwen2.5-VL forward patches for TReVS inference.

The ViT patch exposes spatially merged patch attention for Stage-1 routing.
During decoder prefill, Stage 2 uses block ``k`` Q/K as cache-free lookahead
to score block ``k - 1`` output, physically removes visual tokens, and then
executes block ``k`` on the shortened sequence. The implementation preserves
the four-channel position representation (one text channel plus three M-RoPE
channels), prunes both axes of 4D masks, and builds masks from each layer's own
cache length. These heterogeneous cache lengths require the SDPA backend.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from qwen.model.trevs_router import (
    batch_index_select,
    get_trevs_phase_scoring,
    get_nonnegative_int_env,
    is_trevs_enabled,
    score_phase_attention,
    trevs_route,
    use_sink_token,
)


_PATCHED = False
_ORIGINAL_FORWARDS = {}
TREVS_SEMANTIC_LAYER_ENV = "TREVS_SEMANTIC_LAYER"


def _get_trevs_semantic_layer() -> Optional[int]:
    raw = os.getenv(TREVS_SEMANTIC_LAYER_ENV, "").strip().lower()
    if not raw or raw in {"none", "off", "default", "current"}:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"Invalid {TREVS_SEMANTIC_LAYER_ENV}={raw!r}; expected a Qwen ViT block index or none/off/default/current."
        ) from exc


def _resolve_layer_index(layer_idx: int, num_layers: int) -> int:
    resolved = layer_idx + num_layers if layer_idx < 0 else layer_idx
    if resolved < 0 or resolved >= num_layers:
        raise ValueError(
            f"{TREVS_SEMANTIC_LAYER_ENV}={layer_idx} resolves to block {resolved}, but Qwen ViT has {num_layers} blocks."
        )
    return resolved


def _load_hf_qwen_module():
    # This process-wide import shim targets the locked Qwen Transformers ABI.
    # Qwen and LLaVA run in separate environments; no model forward is patched here.
    import transformers.generation as generation
    import transformers.utils as transformers_utils
    from transformers.utils import import_utils

    import_utils.is_sklearn_available = lambda: False
    transformers_utils.is_sklearn_available = lambda: False
    from transformers.generation.utils import GenerationMixin

    if not hasattr(generation, "GenerationMixin"):
        generation.GenerationMixin = GenerationMixin
    import_utils.is_torch_available = lambda: True
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as hf_qwen

    return hf_qwen


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _make_mm_token_type_ids(model, input_ids: torch.Tensor, mm_token_type_ids: Optional[torch.Tensor]) -> torch.Tensor:
    if mm_token_type_ids is not None:
        return mm_token_type_ids
    token_type_ids = torch.zeros_like(input_ids, dtype=torch.int)
    token_type_ids[input_ids == model.config.image_token_id] = 1
    token_type_ids[input_ids == model.config.video_token_id] = 2
    return token_type_ids


def _build_text_embeddings(
    embed_tokens: nn.Embedding,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    image_token_id: int,
    video_token_id: int,
):
    """Embed valid text after the last multimodal placeholder for Stage 1.

    Padding and image/video placeholders never enter semantic similarity. An
    empty suffix produces a placeholder embedding with an all-false mask; the
    router handles that mask through its explicit empty-text fallback.
    """
    text_ids = []
    text_lengths = []
    max_text_len = 0
    for batch_idx in range(input_ids.shape[0]):
        valid_mask = torch.ones_like(input_ids[batch_idx], dtype=torch.bool)
        if attention_mask is not None:
            valid_mask = attention_mask[batch_idx].bool()
        cur_ids = input_ids[batch_idx][valid_mask]
        multimodal_mask = (cur_ids == image_token_id) | (cur_ids == video_token_id)
        if multimodal_mask.any():
            last_multimodal_index = torch.nonzero(multimodal_mask, as_tuple=False)[-1, 0]
            cur_ids = cur_ids[last_multimodal_index + 1 :]
        cur_text_ids = cur_ids[(cur_ids != image_token_id) & (cur_ids != video_token_id)]
        text_ids.append(cur_text_ids)
        text_lengths.append(int(cur_text_ids.numel()))
        max_text_len = max(max_text_len, int(cur_text_ids.numel()))

    if max_text_len == 0:
        text_ids_padded = torch.zeros(input_ids.shape[0], 1, dtype=torch.long, device=input_ids.device)
        text_mask = torch.zeros(input_ids.shape[0], 1, dtype=torch.bool, device=input_ids.device)
    else:
        text_ids_padded = torch.zeros(input_ids.shape[0], max_text_len, dtype=torch.long, device=input_ids.device)
        text_mask = torch.zeros(input_ids.shape[0], max_text_len, dtype=torch.bool, device=input_ids.device)
        for batch_idx, cur_ids in enumerate(text_ids):
            cur_len = text_lengths[batch_idx]
            if cur_len > 0:
                text_ids_padded[batch_idx, :cur_len] = cur_ids
                text_mask[batch_idx, :cur_len] = True

    with torch.no_grad():
        text_embeds = embed_tokens(text_ids_padded)
    return text_embeds, text_mask


def _prune_sequence_by_image_indices(
    tensor: Optional[torch.Tensor],
    keep_mask: torch.Tensor,
    batch_first_dims: int = 2,
):
    if tensor is None:
        return None
    if tensor.ndim == 2 and batch_first_dims == 2:
        return tensor[keep_mask].reshape(tensor.shape[0], -1)
    if tensor.ndim == 3 and tensor.shape[1] == keep_mask.shape[0] and tensor.shape[2] == keep_mask.shape[1]:
        expanded_mask = keep_mask.unsqueeze(0).expand(tensor.shape[0], -1, -1)
        return tensor[expanded_mask].reshape(tensor.shape[0], tensor.shape[1], -1)
    raise NotImplementedError(f"Unsupported tensor shape for sequence pruning: {tuple(tensor.shape)}")


def _merge_patch_attention(attn_weights: torch.Tensor, spatial_merge_unit: int, reverse_indices: torch.Tensor) -> torch.Tensor:
    """Map raw-patch attention to the merged visual-token order used by the LLM."""
    H, raw_q, raw_k = attn_weights.shape
    if raw_q != raw_k or raw_q % spatial_merge_unit != 0:
        raise ValueError(f"Unexpected Qwen vision attention shape: {tuple(attn_weights.shape)}.")
    n_vis = raw_q // spatial_merge_unit
    merged = attn_weights.reshape(H, n_vis, spatial_merge_unit, n_vis, spatial_merge_unit).mean(dim=(2, 4))
    reverse_indices = reverse_indices.to(device=merged.device)
    merged = merged[:, reverse_indices, :]
    merged = merged[:, :, reverse_indices]
    return merged


def _manual_vision_attention_probs(attn_module, hidden_states, cu_seqlens, position_embeddings):
    """Expose block-local ViT probabilities without changing the normal block output."""
    hf_qwen = _load_hf_qwen_module()
    seq_length = hidden_states.shape[0]
    query_states, key_states, _ = (
        attn_module.qkv(hidden_states).reshape(seq_length, 3, attn_module.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    )
    cos, sin = position_embeddings
    query_states, key_states = hf_qwen.apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
    query_states = query_states.transpose(0, 1)
    key_states = key_states.transpose(0, 1)
    head_dim = query_states.shape[-1]
    attn_weights = torch.matmul(query_states, key_states.transpose(1, 2)) / math.sqrt(head_dim)
    mask = torch.full((1, seq_length, seq_length), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    for i in range(1, len(cu_seqlens)):
        mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0
    attn_weights = attn_weights + mask
    return torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)


def qwen_vision_forward_trevs(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs):
    """Run the Qwen ViT while probing merged patch attention for Stage 1.

    The probe reads the final full-attention block without changing its normal
    output. Raw patch attention is averaged across both spatial-merge axes and
    restored with ``reverse_indices`` to match pooled LLM visual-token order.
    A configured semantic-layer state is merged separately and is used only as
    routing evidence; the normal pooled output remains the injected feature.
    """
    hf_qwen = _load_hf_qwen_module()
    trevs_semantic_layer = _get_trevs_semantic_layer() if is_trevs_enabled() else None
    resolved_semantic_layer = (
        _resolve_layer_index(trevs_semantic_layer, len(self.blocks)) if trevs_semantic_layer is not None else None
    )

    hidden_states = self.patch_embed(hidden_states)
    rotary_pos_emb = self.rot_pos_emb(grid_thw)
    window_index, cu_window_seqlens = self.get_window_index(grid_thw)
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens,
        device=hidden_states.device,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    selected_layer = self.fullatt_block_indexes[-1] if len(self.fullatt_block_indexes) > 0 else len(self.blocks) - 1
    selected_attn = None
    semantic_hidden_states = None
    for layer_num, blk in enumerate(self.blocks):
        cu_seqlens_now = cu_seqlens if layer_num in self.fullatt_block_indexes else cu_window_seqlens
        if layer_num == selected_layer:
            selected_attn = _manual_vision_attention_probs(
                blk.attn,
                blk.norm1(hidden_states),
                cu_seqlens_now,
                position_embeddings,
            )
        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens_now,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if layer_num == resolved_semantic_layer:
            semantic_hidden_states = hidden_states

    merged_hidden_states = self.merger(hidden_states)
    reverse_indices = torch.argsort(window_index)
    merged_hidden_states = merged_hidden_states[reverse_indices, :]
    if selected_attn is None:
        raise RuntimeError("Failed to collect Qwen vision attention for TReVS routing.")
    merged_attn = _merge_patch_attention(selected_attn, self.spatial_merge_unit, reverse_indices).unsqueeze(0)
    semantic_merged_hidden_states = None
    if semantic_hidden_states is not None:
        if resolved_semantic_layer == len(self.blocks) - 1:
            semantic_merged_hidden_states = merged_hidden_states
        else:
            semantic_merged_hidden_states = self.merger(semantic_hidden_states)[reverse_indices, :]

    return hf_qwen.BaseModelOutputWithPooling(
        last_hidden_state=hidden_states,
        pooler_output=merged_hidden_states,
        hidden_states=(semantic_merged_hidden_states,) if semantic_merged_hidden_states is not None else None,
        attentions=merged_attn,
    )


def _build_causal_mask_mapping(language_model, attention_mask, inputs_embeds, past_key_values, text_position_ids):
    hf_qwen = _load_hf_qwen_module()
    if isinstance(attention_mask, dict):
        return attention_mask
    mask_kwargs = {
        "config": language_model.config,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "position_ids": text_position_ids,
    }
    causal_mask_mapping = {"full_attention": hf_qwen.create_causal_mask(**mask_kwargs)}
    if getattr(language_model, "has_sliding_layers", False):
        causal_mask_mapping["sliding_attention"] = hf_qwen.create_sliding_window_causal_mask(**mask_kwargs)
    return causal_mask_mapping


def _cache_layer_seq_len(past_key_values, layer_idx: int) -> int:
    if past_key_values is None or not hasattr(past_key_values, "layers"):
        return 0
    if layer_idx >= len(past_key_values.layers):
        return 0
    layer = past_key_values.layers[layer_idx]
    if not hasattr(layer, "get_seq_length"):
        return 0
    seq_len = layer.get_seq_length()
    if isinstance(seq_len, torch.Tensor):
        seq_len = int(seq_len.item())
    return int(seq_len)


def _build_layer_causal_mask(
    language_model,
    hidden_states: torch.Tensor,
    attention_mask_2d: Optional[torch.Tensor],
    past_key_values,
    layer_idx: int,
):
    """Build a causal mask from one decoder layer's local KV-cache length."""
    if isinstance(attention_mask_2d, dict):
        return attention_mask_2d[language_model.config.layer_types[layer_idx]]

    bsz, q_len, _ = hidden_states.shape
    past_len = _cache_layer_seq_len(past_key_values, layer_idx)
    kv_len = past_len + q_len
    device = hidden_states.device
    dtype = hidden_states.dtype

    query_pos = torch.arange(past_len, past_len + q_len, device=device).view(q_len, 1)
    key_pos = torch.arange(kv_len, device=device).view(1, kv_len)
    allowed = key_pos <= query_pos

    layer_type = language_model.config.layer_types[layer_idx]
    if layer_type == "sliding_attention" and getattr(language_model.config, "sliding_window", None) is not None:
        sliding_window = int(language_model.config.sliding_window)
        allowed = allowed & (key_pos > (query_pos - sliding_window))

    mask = torch.zeros((bsz, 1, q_len, kv_len), device=device, dtype=dtype)
    mask = mask.masked_fill(~allowed.view(1, 1, q_len, kv_len), torch.finfo(dtype).min)

    if attention_mask_2d is not None and attention_mask_2d.ndim == 2 and attention_mask_2d.shape[1] == kv_len:
        key_valid = attention_mask_2d.to(device=device, dtype=torch.bool).view(bsz, 1, 1, kv_len)
        mask = mask.masked_fill(~key_valid, torch.finfo(dtype).min)

    return mask


def _manual_llm_attention_probs(self_attn, hidden_states, position_embeddings, attention_mask):
    """Compute M-RoPE attention probabilities without running or caching the block."""
    hf_qwen = _load_hf_qwen_module()
    bsz, q_len, _ = hidden_states.shape
    query_states = self_attn.q_proj(hidden_states)
    key_states = self_attn.k_proj(hidden_states)
    query_states = query_states.view(bsz, q_len, -1, self_attn.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self_attn.head_dim).transpose(1, 2)
    cos, sin = position_embeddings
    mrope_section = self_attn.config.rope_parameters.get("mrope_section")
    if mrope_section is None:
        rotary_half = self_attn.head_dim // 2
        base = rotary_half // 3
        mrope_section = [base, base, rotary_half - 2 * base]
    query_states, key_states = hf_qwen.apply_multimodal_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        mrope_section,
    )
    key_states = _repeat_kv(key_states, self_attn.num_key_value_groups)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self_attn.scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    if query_states.dtype == torch.float16:
        attn_weights = torch.where(torch.isinf(attn_weights), torch.full_like(attn_weights, torch.finfo(attn_weights.dtype).min), attn_weights)
    return torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)


def _prune_attention_mask_qwen(attention_mask, keep_indices: torch.Tensor):
    """Gather every sequence axis represented by a Qwen attention mask."""
    if attention_mask is None:
        return None
    if isinstance(attention_mask, dict):
        return {key: _prune_attention_mask_qwen(value, keep_indices) for key, value in attention_mask.items()}
    if attention_mask.ndim == 2:
        return batch_index_select(attention_mask, keep_indices)
    if attention_mask.ndim == 4:
        batch_size, num_heads, query_length, key_length = attention_mask.shape
        if query_length != key_length or query_length < keep_indices.shape[1]:
            raise ValueError(
                f"Cannot TReVS-prune attention mask with shape {tuple(attention_mask.shape)} "
                f"using {keep_indices.shape[1]} kept positions."
            )
        query_index = keep_indices[:, None, :, None].expand(batch_size, num_heads, -1, key_length)
        pruned = torch.gather(attention_mask, dim=2, index=query_index)
        key_index = keep_indices[:, None, None, :].expand(batch_size, num_heads, pruned.shape[2], -1)
        return torch.gather(pruned, dim=3, index=key_index)
    raise NotImplementedError(f"Unsupported attention mask rank for Qwen TReVS pruning: {attention_mask.ndim}.")


def _apply_trevs_phase_pruning_qwen(
    hidden_states: torch.Tensor,
    v_token_start: int,
    idx_drop_local: torch.Tensor,
    attention_mask,
    position_ids: torch.Tensor,
    text_position_ids: Optional[torch.Tensor] = None,
    use_sink: bool = True,
):
    """Physically prune Qwen states while preserving all position channels.

    Optional sink mode retains the mean dropped state at the first dropped
    position. The three M-RoPE channels, separate text-position channel, and
    both query/key axes of a 4D mask use the same keep indices.
    """
    B, L_total, _ = hidden_states.shape
    idx_drop_global = idx_drop_local + v_token_start
    sink_count = 1 if use_sink and idx_drop_global.shape[1] > 0 else 0
    removed_count = idx_drop_global.shape[1] - sink_count

    if use_sink:
        for batch_idx in range(B):
            drop_positions = idx_drop_global[batch_idx]
            if drop_positions.numel() == 0:
                continue
            sink_token = hidden_states[batch_idx, drop_positions, :].mean(dim=0, keepdim=True)
            sink_pos = drop_positions[0]
            hidden_states[batch_idx, sink_pos, :] = sink_token.squeeze(0)

    all_idx = torch.arange(L_total, device=hidden_states.device).unsqueeze(0).expand(B, -1)
    keep_mask = torch.ones(B, L_total, dtype=torch.bool, device=hidden_states.device)
    if removed_count > 0:
        keep_mask.scatter_(1, idx_drop_global[:, sink_count:], False)
    keep_indices = all_idx[keep_mask].reshape(B, L_total - removed_count)

    hidden_states = batch_index_select(hidden_states, keep_indices)
    attention_mask = _prune_attention_mask_qwen(attention_mask, keep_indices)
    position_ids = torch.gather(position_ids, 2, keep_indices.unsqueeze(0).expand(position_ids.shape[0], -1, -1))
    if text_position_ids is not None:
        text_position_ids = batch_index_select(text_position_ids, keep_indices)
    return hidden_states, attention_mask, position_ids, text_position_ids


def qwen_text_model_forward_trevs(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    **kwargs,
):
    """Run Qwen SDPA decoding with one cache-free lookahead transition.

    Per-layer prompt caches may have different lengths after physical pruning,
    requiring layer-local 4D additive masks; FlashAttention2 cannot represent
    this contract in the active Qwen TReVS path.
    """
    hf_qwen = _load_hf_qwen_module()
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    trevs_prefill_candidate = (
        is_trevs_enabled()
        and inputs_embeds.shape[1] != 1
        and getattr(self, "n_image_tokens", 0) > 0
        and getattr(self, "image_start_index", None) is not None
    )

    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = hf_qwen.DynamicCache(config=self.config)

    if position_ids is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        position_ids = position_ids.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    # Keep the text channel separate while preserving all three M-RoPE channels.
    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = None

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    trevs_prefill = trevs_prefill_candidate
    trevs_phase_scoring = get_trevs_phase_scoring()
    trevs_use_sink_token = use_sink_token()
    phase_layer = get_nonnegative_int_env("PHASE_TRANSITION_LAYER", 8)
    phase_n_keep = get_nonnegative_int_env("PHASE_TRANSITION_N_KEEP", 39)
    phase_done = False

    if trevs_prefill and not 1 <= phase_layer < len(self.layers):
        raise ValueError(
            "PHASE_TRANSITION_LAYER must satisfy 1 <= PHASE_TRANSITION_LAYER "
            f"< num_hidden_layers ({len(self.layers)}); got {phase_layer}. The value is the number of "
            "decoder blocks that run on the full sequence before lookahead pruning."
        )

    for layer_idx, decoder_layer in enumerate(self.layers):
        if trevs_prefill and layer_idx == phase_layer and not phase_done:
            # Block k supplies Q/K to score block k-1 output before block k runs.
            # This lookahead does not invoke the block or update its KV cache.
            score_layer_mask = _build_layer_causal_mask(
                self,
                hidden_states,
                attention_mask,
                past_key_values,
                layer_idx,
            )
            normed_hidden = decoder_layer.input_layernorm(hidden_states)
            attn_for_prune = _manual_llm_attention_probs(
                decoder_layer.self_attn,
                normed_hidden,
                position_embeddings,
                score_layer_mask,
            )
            n_vis = int(self.n_image_tokens)
            vis_start = int(self.image_start_index)
            text_start = vis_start + n_vis
            text_end = hidden_states.shape[1]
            attn_tv = attn_for_prune[:, :, text_start:text_end, vis_start : vis_start + n_vis]
            if attn_tv.shape[2] > 0 and attn_tv.shape[3] > 0:
                actual_n_keep = min(phase_n_keep, n_vis - 1)
                _, idx_drop_local = score_phase_attention(
                    attn_tv,
                    n_keep=actual_n_keep,
                    mode=trevs_phase_scoring,
                )
                hidden_states, attention_mask, position_ids, text_position_ids = _apply_trevs_phase_pruning_qwen(
                    hidden_states=hidden_states,
                    v_token_start=vis_start,
                    idx_drop_local=idx_drop_local,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    text_position_ids=text_position_ids,
                    use_sink=trevs_use_sink_token,
                )
                self.n_image_tokens = actual_n_keep + int(trevs_use_sink_token)
                self.trevs_phase_stats = {
                    "layer": layer_idx,
                    "full_layers": phase_layer,
                    "scoring_layer_idx": layer_idx,
                    "pruned_after_layer_idx": layer_idx - 1,
                    "n_vis_before": n_vis,
                    "n_vis_after": self.n_image_tokens,
                    "scoring": trevs_phase_scoring,
                    "use_sink_token": trevs_use_sink_token,
                }
                position_embeddings = self.rotary_emb(hidden_states, position_ids)
                phase_done = True

        # Pre-transition and post-transition layers retain different prompt-cache
        # lengths, so each layer requires a mask built from its own cache state.
        layer_mask = _build_layer_causal_mask(
            self,
            hidden_states,
            attention_mask,
            past_key_values,
            layer_idx,
        )
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=layer_mask,
            position_embeddings=position_embeddings,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

    hidden_states = self.norm(hidden_states)
    return hf_qwen.BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


def qwen_vl_model_forward_trevs(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    rope_deltas=None,
    mm_token_type_ids=None,
    second_per_grid_ts=None,
    **kwargs,
):
    """Assemble a routed Qwen prompt while preserving multimodal positions.

    M-RoPE positions and the generation rope delta are computed from the full
    prompt first. One order-preserving keep set then prunes input IDs, token
    types, masks, and every position channel before routed image embeddings are
    inserted.
    """
    hf_qwen = _load_hf_qwen_module()
    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    if pixel_values is None:
        self.language_model.n_image_tokens = 0
        self.language_model.image_start_index = None

    mm_token_type_ids = _make_mm_token_type_ids(self, input_ids, mm_token_type_ids) if input_ids is not None else mm_token_type_ids

    if position_ids is None:
        position_ids = self.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            mm_token_type_ids=mm_token_type_ids,
        )

    if pixel_values is not None:
        self.trevs_route_stats = None
        self.language_model.trevs_phase_stats = None
        pixel_values = pixel_values.type(self.visual.dtype)
        vision_outputs = self.visual(pixel_values, grid_thw=image_grid_thw)
        image_embeds = vision_outputs.pooler_output
        vit_attn = vision_outputs.attentions
        semantic_image_embeds = (
            vision_outputs.hidden_states[0]
            if vision_outputs.hidden_states is not None and len(vision_outputs.hidden_states) > 0
            else None
        )
        n_image_tokens = int((input_ids == self.config.image_token_id).sum().item())
        if image_embeds.shape[0] != n_image_tokens:
            raise ValueError(f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_embeds.shape[0]}")
        if n_image_tokens != 1280:
            raise ValueError(
                f"Qwen2.5-VL TReVS expects exactly 1280 visual tokens, got {n_image_tokens}. "
                "Use AutoProcessor(min_pixels=1280*28*28, max_pixels=1280*28*28)."
            )

        image_mask_2d = input_ids == self.config.image_token_id
        image_indices = torch.nonzero(image_mask_2d, as_tuple=False)
        image_start_index = int(image_indices[0, 1].item())
        if is_trevs_enabled():
            route_topk = get_nonnegative_int_env("TREVS_ROUTE_TOPK", 96)
            route_fps = get_nonnegative_int_env("TREVS_ROUTE_FPS", 32)
            semantic_layer = _get_trevs_semantic_layer()
            text_embeds, text_mask = _build_text_embeddings(
                self.get_input_embeddings(),
                input_ids,
                attention_mask,
                self.config.image_token_id,
                self.config.video_token_id,
            )
            idx_routed, route_stats = trevs_route(
                vit_attn=vit_attn,
                V_proj=image_embeds.unsqueeze(0),
                T_emb=text_embeds.to(image_embeds.device),
                n_topk=route_topk,
                n_fps=route_fps,
                core_token_mask=text_mask.to(image_embeds.device),
                V_semantic_proj=(
                    semantic_image_embeds.unsqueeze(0) if semantic_image_embeds is not None else None
                ),
                semantic_layer=semantic_layer,
                return_stats=True,
            )
            keep_indices_local = idx_routed[0]
            image_embeds = route_stats["V_routed"].squeeze(0)
            # Apply the routed order-preserving keep set to every prompt tensor.
            keep_mask = torch.ones_like(input_ids, dtype=torch.bool)
            all_image_local = torch.arange(n_image_tokens, device=input_ids.device)
            remove_local = all_image_local[~torch.isin(all_image_local, keep_indices_local.to(input_ids.device))]
            remove_positions = image_indices[remove_local]
            keep_mask[remove_positions[:, 0], remove_positions[:, 1]] = False
            input_ids = _prune_sequence_by_image_indices(input_ids, keep_mask)
            mm_token_type_ids = _prune_sequence_by_image_indices(mm_token_type_ids, keep_mask)
            attention_mask = _prune_sequence_by_image_indices(attention_mask, keep_mask) if attention_mask is not None else None
            position_ids = _prune_sequence_by_image_indices(position_ids, keep_mask)
            inputs_embeds = self.get_input_embeddings()(input_ids)
            # Generation keeps the original prompt mask, so its decode positions require the
            # rope delta computed before internal visual-token pruning.
            self.trevs_route_stats = {
                key: value
                for key, value in route_stats.items()
                if key != "V_routed"
            }
            self.trevs_route_stats["idx_routed"] = idx_routed
        else:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        self.language_model.n_image_tokens = int((input_ids == self.config.image_token_id).sum().item())
        self.language_model.image_start_index = image_start_index

    if pixel_values_videos is not None:
        video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw).pooler_output
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        **kwargs,
    )
    return hf_qwen.Qwen2_5_VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )


def qwen_for_conditional_generation_forward_trevs(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    rope_deltas=None,
    mm_token_type_ids=None,
    second_per_grid_ts=None,
    logits_to_keep=0,
    **kwargs,
):
    hf_qwen = _load_hf_qwen_module()
    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        mm_token_type_ids=mm_token_type_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        **kwargs,
    )
    hidden_states = outputs[0]
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :])
    loss = None
    if labels is not None:
        if hasattr(self, "loss_function"):
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs)
        else:
            shift_logits = logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
            loss = nn.CrossEntropyLoss()(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
    return hf_qwen.Qwen2_5_VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=outputs.rope_deltas,
    )


def apply_qwen2_5_vl_trevs_patches():
    """Idempotently install TReVS forwards or restore all native dense forwards.

    Exactly four model forwards are replaced for ``METHOD=trevs``. Repeated
    calls reuse the first saved native implementations; ``METHOD=dense``
    restores those implementations before model loading.
    """
    global _PATCHED
    hf_qwen = _load_hf_qwen_module()
    forward_targets = (
        (hf_qwen.Qwen2_5_VisionTransformerPretrainedModel, qwen_vision_forward_trevs),
        (hf_qwen.Qwen2_5_VLTextModel, qwen_text_model_forward_trevs),
        (hf_qwen.Qwen2_5_VLModel, qwen_vl_model_forward_trevs),
        (hf_qwen.Qwen2_5_VLForConditionalGeneration, qwen_for_conditional_generation_forward_trevs),
    )
    for model_class, _ in forward_targets:
        _ORIGINAL_FORWARDS.setdefault(model_class, model_class.forward)

    if not is_trevs_enabled():
        for model_class, _ in forward_targets:
            model_class.forward = _ORIGINAL_FORWARDS[model_class]
        _PATCHED = False
        return False
    for model_class, trevs_forward in forward_targets:
        model_class.forward = trevs_forward
    _PATCHED = True
    return True
