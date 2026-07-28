"""Self-contained TReVS routing for Video-LLaVA.

Stage 1 combines full-ViT CLS-to-patch saliency with projector-space
text-to-vision cosine evidence. Both signals become positive median/MAD
evidence; score Top-K forms Track 1 and FPS samples only from its disjoint
remainder for Track 2. Final indices are sorted to preserve source-token order.

Stage 2 supports all-head or high-variance priority-head scoring and optional
mean-sink retention. Its semantics mirror the LLaVA path without importing or
monkey-patching that package, preserving Video-LLaVA's isolated runtime. The
caller flattens batch and frame axes so Stage-1 Top-K/FPS budgets apply per
frame. Stage 2 scores the concatenated global visual span once; the packaged
Video-LLaVA default disables sink retention.
"""

from typing import Dict, Optional, Tuple

import os

import torch
import torch.nn.functional as F

from .utils import batch_index_select


EPS = 1e-6
TREVS_TEXT_SCORE_MODE_ENV = "TREVS_TEXT_SCORE_MODE"
TREVS_VISUAL_TEMPERATURE_ENV = "TREVS_VISUAL_TEMPERATURE"
TREVS_TEXT_TEMPERATURE_ENV = "TREVS_TEXT_TEMPERATURE"
TREVS_PHASE_SCORING_ENV = "TREVS_PHASE_SCORING"
TREVS_USE_SINK_TOKEN_ENV = "TREVS_USE_SINK_TOKEN"
DOUBLE_TRACK_USE_CONSISTENCY_REWARD_ENV = "DOUBLE_TRACK_USE_CONSISTENCY_REWARD"
TREVS_TEXT_SCORE_MODE_VALUES = ("rms", "max", "mean")
TREVS_PHASE_SCORING_VALUES = ("priority_heads", "all_heads")


def _get_bool_env(name: str, default: bool) -> bool:
    env_value = os.getenv(name, "").strip().lower()
    if not env_value:
        return default
    if env_value in {"1", "true", "yes", "on"}:
        return True
    if env_value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Invalid {name}={env_value!r}; expected one of 1/0, true/false, yes/no, or on/off."
    )


def _get_positive_float_env(name: str, default: float) -> float:
    env_value = os.getenv(name, "").strip()
    if not env_value:
        return float(default)
    try:
        value = float(env_value)
    except ValueError as exc:
        raise ValueError(f"Invalid {name}={env_value!r}; expected a positive float.") from exc
    if value <= 0.0:
        raise ValueError(f"Invalid {name}={env_value!r}; expected a positive float.")
    return value


def get_trevs_text_score_mode() -> str:
    env_value = os.getenv(TREVS_TEXT_SCORE_MODE_ENV, "").strip().lower()
    if not env_value:
        return "rms"
    if env_value not in TREVS_TEXT_SCORE_MODE_VALUES:
        raise ValueError(
            f"Invalid text score mode {env_value!r}; supported modes: {list(TREVS_TEXT_SCORE_MODE_VALUES)}."
        )
    return env_value


def get_trevs_phase_scoring() -> str:
    env_value = os.getenv(TREVS_PHASE_SCORING_ENV, "").strip().lower()
    if not env_value:
        return "priority_heads"
    if env_value not in TREVS_PHASE_SCORING_VALUES:
        raise ValueError(
            f"Invalid phase scoring mode {env_value!r}; supported modes: {list(TREVS_PHASE_SCORING_VALUES)}."
        )
    return env_value


def use_consistency_reward() -> bool:
    return _get_bool_env(DOUBLE_TRACK_USE_CONSISTENCY_REWARD_ENV, True)


def use_sink_token() -> bool:
    return _get_bool_env(TREVS_USE_SINK_TOKEN_ENV, False)


def _resolve_method_name() -> str:
    method = os.getenv("METHOD", "trevs").strip().lower()
    if not method:
        return "trevs"
    if method in {"trevs", "dense"}:
        return method
    raise ValueError(f"Unsupported METHOD={method!r}. Expected 'trevs' or 'dense'.")


def farthest_point_sampling(
    features: torch.Tensor,
    n_samples: int,
    reference_features: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return candidate-local FPS indices using ``1 - cosine_similarity``.

    L2-normalized Track-1 features initialize the reference set when present;
    otherwise candidate index zero is a deterministic anchor. Each new FPS
    point joins that set. The caller maps these local indices back to visual
    indices and sorts the final Top-K/FPS union only after selection.
    """
    batch, n_tokens, channels = features.shape
    device = features.device

    if n_samples == 0:
        return torch.empty(batch, 0, dtype=torch.long, device=device)
    if n_tokens < n_samples:
        raise ValueError(f"Cannot FPS-sample {n_samples} points from only {n_tokens} candidates.")

    feat_norm = F.normalize(features.float(), p=2, dim=-1)
    idx_selected = torch.zeros(batch, n_samples, dtype=torch.long, device=device)
    selected_mask = torch.zeros(batch, n_tokens, dtype=torch.bool, device=device)

    if reference_features is not None and reference_features.numel() > 0:
        ref_norm = F.normalize(reference_features.float(), p=2, dim=-1)
        similarity_to_ref = torch.bmm(feat_norm, ref_norm.transpose(1, 2))
        min_distances = 1.0 - similarity_to_ref.max(dim=-1).values
        start_idx = 0
    else:
        idx_selected[:, 0] = 0
        selected_mask[:, 0] = True
        first_point = feat_norm[:, 0:1, :]
        min_distances = 1.0 - (feat_norm * first_point).sum(dim=-1)
        start_idx = 1

    for sample_idx in range(start_idx, n_samples):
        candidate_distances = min_distances.masked_fill(selected_mask, -1.0)
        next_idx = candidate_distances.argmax(dim=-1)
        idx_selected[:, sample_idx] = next_idx
        selected_mask.scatter_(1, next_idx.unsqueeze(-1), True)
        new_point = torch.gather(
            feat_norm,
            1,
            next_idx.view(batch, 1, 1).expand(batch, 1, channels),
        )
        new_dist = 1.0 - (feat_norm * new_point).sum(dim=-1)
        min_distances = torch.min(min_distances, new_dist)

    return idx_selected


def split_vit_attention(vit_attn: torch.Tensor, n_vis: int) -> tuple:
    """Return active CLS saliency inputs or the legacy patch-only fallback."""
    if vit_attn.ndim != 4 or vit_attn.shape[-1] != vit_attn.shape[-2]:
        raise ValueError(f"Unexpected vit_attn shape: {tuple(vit_attn.shape)}")

    attn_tokens = vit_attn.shape[-1]
    if attn_tokens == n_vis + 1:
        return vit_attn[:, :, 1:, 1:], vit_attn[:, :, 0, 1:]
    if attn_tokens == n_vis:
        return vit_attn, None
    raise ValueError(f"vit_attn token dimension {attn_tokens} does not match n_vis={n_vis}")


def _select_phase_topk_indices(vision_scores: torch.Tensor, n_keep: int) -> tuple:
    batch, n_vis = vision_scores.shape
    n_keep = min(max(int(n_keep), 0), max(n_vis - 1, 0))

    if n_keep == 0:
        idx_keep = torch.empty(batch, 0, dtype=torch.long, device=vision_scores.device)
    else:
        idx_keep = torch.topk(vision_scores, k=n_keep, dim=-1).indices
        idx_keep = torch.sort(idx_keep, dim=-1).values

    all_idx = torch.arange(n_vis, device=vision_scores.device).unsqueeze(0).expand(batch, -1)
    keep_mask = torch.zeros(batch, n_vis, dtype=torch.bool, device=vision_scores.device)
    if idx_keep.numel() > 0:
        keep_mask.scatter_(1, idx_keep, True)
    idx_drop = all_idx[~keep_mask].reshape(batch, n_vis - n_keep)
    return idx_keep, idx_drop


def score_phase_attention(
    attn_tv: torch.Tensor,
    n_keep: int = 341,
    mode: Optional[str] = None,
) -> tuple:
    """Score the phase transition with all heads or high-variance priority heads.

    Priority-head mode selects half of the heads by visual-key variance,
    averages those heads, and takes maximum relevance over text queries. The
    returned local visual indices are sorted only to restore input order.
    """
    if attn_tv.ndim != 4:
        raise ValueError(
            f"Expected attn_tv with shape [B, H, L_text, N_vis], got {tuple(attn_tv.shape)}."
        )

    phase_scoring = get_trevs_phase_scoring() if mode is None else mode.strip().lower()
    if phase_scoring not in TREVS_PHASE_SCORING_VALUES:
        raise ValueError(
            f"Invalid phase scoring mode {phase_scoring!r}; supported modes: {list(TREVS_PHASE_SCORING_VALUES)}."
        )

    if phase_scoring == "all_heads":
        vision_scores = attn_tv.mean(dim=1).max(dim=1).values
    else:
        n_heads = attn_tv.shape[1]
        head_var = attn_tv.float().var(dim=-1).mean(dim=-1)
        n_priority = max(n_heads // 2, 1)
        priority_heads = torch.topk(head_var, k=n_priority, dim=-1).indices
        gather_idx = priority_heads.unsqueeze(-1).unsqueeze(-1).expand(
            attn_tv.shape[0], n_priority, attn_tv.shape[2], attn_tv.shape[3]
        )
        priority_attn = torch.gather(attn_tv, 1, gather_idx)
        vision_scores = priority_attn.mean(dim=1).max(dim=1).values

    return _select_phase_topk_indices(vision_scores=vision_scores, n_keep=n_keep)


def score_phase_attention_all_heads(attn_tv: torch.Tensor, n_keep: int = 341) -> tuple:
    return score_phase_attention(attn_tv=attn_tv, n_keep=n_keep, mode="all_heads")


def apply_sink_token_pruning(
    hidden_states: torch.Tensor,
    v_token_start: int,
    n_vis_current: int,
    idx_keep_local: torch.Tensor,
    idx_drop_local: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_ids: torch.Tensor,
    use_sink: bool = True,
) -> tuple:
    """Delete visual tokens while preserving positions and mask alignment.

    Optional sink mode retains the mean dropped state at the first dropped
    position. Position IDs are gathered without renumbering, and a 4D mask is
    gathered along both query and key axes.
    """
    del idx_keep_local
    batch, total_length, _ = hidden_states.shape
    if idx_drop_local.shape[0] != batch:
        raise ValueError("Phase-pruning indices must have the same batch size as hidden_states.")
    if not (0 <= int(v_token_start) <= total_length):
        raise ValueError(f"Invalid visual start {v_token_start} for sequence length {total_length}.")
    if int(n_vis_current) < 0 or int(v_token_start) + int(n_vis_current) > total_length:
        raise ValueError(
            f"Invalid visual span [{v_token_start}, {v_token_start + n_vis_current}) "
            f"for sequence length {total_length}."
        )

    idx_drop_global = idx_drop_local + int(v_token_start)
    sink_count = 1 if use_sink and idx_drop_global.shape[1] > 0 else 0
    removed_count = idx_drop_global.shape[1] - sink_count

    if use_sink:
        hidden_states = hidden_states.clone()
        for batch_idx in range(batch):
            drop_positions = idx_drop_global[batch_idx]
            if drop_positions.numel() == 0:
                continue
            hidden_states[batch_idx, drop_positions[0], :] = hidden_states[
                batch_idx, drop_positions, :
            ].mean(dim=0)

    all_idx = torch.arange(total_length, device=hidden_states.device).unsqueeze(0).expand(batch, -1)
    keep_mask = torch.ones(batch, total_length, dtype=torch.bool, device=hidden_states.device)
    if removed_count > 0:
        keep_mask.scatter_(1, idx_drop_global[:, sink_count:], False)
    keep_indices = all_idx[keep_mask].reshape(batch, total_length - removed_count)

    hidden_states = batch_index_select(hidden_states, keep_indices)
    position_ids = batch_index_select(position_ids, keep_indices)
    if attention_mask is not None:
        if attention_mask.dim() == 4:
            attention_mask = batch_index_select(attention_mask, keep_indices)
            attention_mask = batch_index_select(
                attention_mask.transpose(-1, -2), keep_indices
            ).transpose(-1, -2)
        elif attention_mask.dim() == 2:
            attention_mask = batch_index_select(attention_mask, keep_indices)
        else:
            raise ValueError(
                f"Unsupported attention mask rank {attention_mask.dim()} during TReVS pruning."
            )
    return hidden_states, attention_mask, position_ids


def _build_positive_evidence_batch(values: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Convert each score row to nonnegative median/MAD-normalized evidence."""
    if values.ndim == 1:
        values = values.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    values = values.float()
    abs_max = values.abs().amax(dim=-1, keepdim=True)
    median = values.median(dim=-1, keepdim=True).values
    mad = (values - median).abs().median(dim=-1, keepdim=True).values
    positive = F.relu((values - median) / (mad + eps))
    positive = torch.where(abs_max <= eps, torch.zeros_like(positive), positive)
    return positive.squeeze(0) if squeeze_output else positive


def _compute_saliency_raw_scores(vit_attn: torch.Tensor, n_vis: int) -> torch.Tensor:
    """Use active CLS saliency, with patch-only aggregation as a legacy fallback."""
    patch_patch_attn, cls_patch_attn = split_vit_attention(vit_attn, n_vis=n_vis)
    if cls_patch_attn is None:
        return patch_patch_attn.mean(dim=1).sum(dim=1)
    return cls_patch_attn.mean(dim=1)


def _normalize_text_token_mask(
    text_embeds: torch.Tensor, core_token_mask: Optional[torch.Tensor]
) -> torch.Tensor:
    if core_token_mask is None:
        return torch.ones(text_embeds.shape[:2], dtype=torch.bool, device=text_embeds.device)
    if core_token_mask.shape != text_embeds.shape[:2]:
        raise ValueError(
            f"core_token_mask shape {tuple(core_token_mask.shape)} does not match "
            f"text shape {tuple(text_embeds.shape[:2])}."
        )
    text_mask = core_token_mask.to(device=text_embeds.device, dtype=torch.bool)
    empty_mask = text_mask.sum(dim=-1) == 0
    if empty_mask.any():
        text_mask = text_mask.clone()
        text_mask[empty_mask] = True
    return text_mask


def _compute_trevs_route_scores(
    vit_attn: torch.Tensor,
    vision_projected: torch.Tensor,
    text_embeds: torch.Tensor,
    core_token_mask: Optional[torch.Tensor],
    vision_semantic_projected: Optional[torch.Tensor] = None,
    eps: float = EPS,
) -> Dict[str, torch.Tensor]:
    """Fuse visual saliency and projector-space text/vision semantic evidence."""
    batch, n_vis, _ = vision_projected.shape
    if text_embeds.shape[0] != batch:
        raise ValueError(
            f"Batch size mismatch between V_proj ({batch}) and T_emb ({text_embeds.shape[0]})."
        )
    if vision_semantic_projected is None:
        vision_semantic_projected = vision_projected
    if vision_semantic_projected.shape != vision_projected.shape:
        raise ValueError(
            f"V_semantic_proj shape {tuple(vision_semantic_projected.shape)} is incompatible "
            f"with V_proj shape {tuple(vision_projected.shape)}."
        )

    text_mask = _normalize_text_token_mask(text_embeds, core_token_mask)
    device = vision_projected.device
    text_score_mode = get_trevs_text_score_mode()
    saliency_scores = _compute_saliency_raw_scores(vit_attn, n_vis=n_vis).to(device=device)
    vision_norm = F.normalize(vision_semantic_projected.to(device=device).float(), p=2, dim=-1)
    text_norm = F.normalize(text_embeds.float(), p=2, dim=-1)
    token_maps = torch.bmm(text_norm, vision_norm.transpose(1, 2)).float().clamp_min(0.0)
    token_mask = text_mask.unsqueeze(-1)
    token_count = text_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(dtype=token_maps.dtype)

    if text_score_mode == "max":
        semantic_scores = token_maps.masked_fill(~token_mask, 0.0).max(dim=1).values
    elif text_score_mode == "mean":
        semantic_scores = (token_maps * token_mask).sum(dim=1) / token_count
    else:
        semantic_scores = torch.sqrt((token_maps.pow(2) * token_mask).sum(dim=1) / token_count)

    visual_temperature = _get_positive_float_env(TREVS_VISUAL_TEMPERATURE_ENV, 1.0)
    text_temperature = _get_positive_float_env(TREVS_TEXT_TEMPERATURE_ENV, 1.0)
    visual_evidence = _build_positive_evidence_batch(saliency_scores, eps=eps) / visual_temperature
    semantic_evidence = _build_positive_evidence_batch(semantic_scores, eps=eps) / text_temperature
    consistency_reward_enabled = use_consistency_reward()
    consistency_reward = (
        torch.sqrt(visual_evidence * semantic_evidence) if consistency_reward_enabled else 0.0
    )
    route_scores = torch.maximum(visual_evidence, semantic_evidence) + consistency_reward

    return {
        "selection_mode": "trevs",
        "method": "trevs",
        "text_score_mode": text_score_mode,
        "consistency_reward_enabled": consistency_reward_enabled,
        "visual_temperature": visual_temperature,
        "text_temperature": text_temperature,
        "route_scores": route_scores,
        "tie_break_scores": saliency_scores + semantic_scores.to(device=device),
    }


def _select_stage1_topk_indices(
    route_scores: torch.Tensor,
    total_budget: int,
    tie_break_scores: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    empty_idx = torch.empty(
        route_scores.shape[0], 0, dtype=torch.long, device=route_scores.device
    )
    if total_budget <= 0:
        return {"idx_track1": empty_idx}

    effective_scores = route_scores.float()
    if tie_break_scores is not None:
        effective_scores = effective_scores + EPS * tie_break_scores.float()
    idx_track1 = torch.topk(effective_scores, k=total_budget, dim=-1).indices
    return {"idx_track1": torch.sort(idx_track1, dim=-1).values}


def _build_stage1_routing_outputs(
    vision_projected: torch.Tensor,
    idx_track1: torch.Tensor,
    fps_track2_budget: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run disjoint FPS on tokens not already selected by score Top-K."""
    batch, n_vis, _ = vision_projected.shape
    device = vision_projected.device
    all_indices = torch.arange(n_vis, device=device)
    idx_track2_list = []
    idx_routed_list = []

    selected_mask = torch.zeros(batch, n_vis, dtype=torch.bool, device=device)
    if idx_track1.numel() > 0:
        selected_mask.scatter_(1, idx_track1, True)

    for batch_idx in range(batch):
        candidate_indices = all_indices[~selected_mask[batch_idx]]
        if fps_track2_budget == 0:
            idx_track2_b = torch.empty(0, dtype=torch.long, device=device)
        else:
            candidate_features = vision_projected[batch_idx, candidate_indices, :].unsqueeze(0)
            reference_features = (
                vision_projected[batch_idx, idx_track1[batch_idx], :].unsqueeze(0)
                if idx_track1.shape[1] > 0
                else None
            )
            fps_local_idx = farthest_point_sampling(
                candidate_features,
                fps_track2_budget,
                reference_features=reference_features,
            )
            idx_track2_b = candidate_indices[fps_local_idx.squeeze(0)]

        idx_track2_list.append(idx_track2_b)
        idx_routed_b = torch.cat([idx_track1[batch_idx], idx_track2_b], dim=0)
        if torch.unique(idx_routed_b).numel() != idx_routed_b.numel():
            raise ValueError("TReVS routing produced overlapping stage selections.")
        idx_routed_list.append(torch.sort(idx_routed_b, dim=-1).values)

    idx_track2 = torch.stack(idx_track2_list, dim=0)
    idx_routed = torch.stack(idx_routed_list, dim=0)
    return idx_track2, idx_routed, batch_index_select(vision_projected, idx_routed)


def trevs_route(
    vit_attn: torch.Tensor,
    V_proj: torch.Tensor,
    T_emb: torch.Tensor,
    k_track1: int = 96,
    k_track2: int = 32,
    core_token_mask: Optional[torch.Tensor] = None,
    V_semantic_proj: Optional[torch.Tensor] = None,
    semantic_layer: Optional[int] = None,
    timing_enabled: bool = False,
    return_stats: bool = False,
):
    """Run score Top-K followed by disjoint FPS in source-token order."""
    _, n_vis, _ = V_proj.shape
    if k_track1 < 0 or k_track2 < 0:
        raise ValueError(f"TReVS token budgets must be >= 0, got {(k_track1, k_track2)}.")
    if k_track1 + k_track2 > n_vis:
        raise ValueError(
            f"Requested TReVS token budgets {(k_track1, k_track2)} exceed available visual tokens {n_vis}."
        )
    method = _resolve_method_name()
    if method != "trevs":
        raise ValueError(f"trevs_route only supports METHOD=trevs, got {method!r}.")

    route_k1_start_event = route_k1_end_event = None
    route_k2_start_event = route_k2_end_event = None
    if timing_enabled:
        if not V_proj.is_cuda:
            raise ValueError("TReVS CUDA timing requires CUDA vision tensors.")
        route_k1_start_event = torch.cuda.Event(enable_timing=True)
        route_k1_start_event.record()

    track_scores = _compute_trevs_route_scores(
        vit_attn=vit_attn,
        vision_projected=V_proj,
        text_embeds=T_emb,
        core_token_mask=core_token_mask,
        vision_semantic_projected=V_semantic_proj,
        eps=torch.finfo(V_proj.float().dtype).eps,
    )
    idx_track1 = _select_stage1_topk_indices(
        route_scores=track_scores["route_scores"],
        total_budget=k_track1,
        tie_break_scores=track_scores["tie_break_scores"],
    )["idx_track1"]
    if timing_enabled:
        route_k1_end_event = torch.cuda.Event(enable_timing=True)
        route_k1_end_event.record()
        route_k2_start_event = torch.cuda.Event(enable_timing=True)
        route_k2_start_event.record()

    idx_track2, idx_routed, vision_routed = _build_stage1_routing_outputs(
        vision_projected=V_proj,
        idx_track1=idx_track1,
        fps_track2_budget=k_track2,
    )
    if timing_enabled:
        route_k2_end_event = torch.cuda.Event(enable_timing=True)
        route_k2_end_event.record()
    if idx_routed.shape[1] != k_track1 + k_track2:
        raise ValueError("TReVS routing produced an unexpected number of routed tokens.")

    if not return_stats:
        return idx_routed
    return idx_routed, {
        "selection_mode": track_scores["selection_mode"],
        "method": track_scores["method"],
        "text_score_mode": track_scores["text_score_mode"],
        "consistency_reward_enabled": track_scores["consistency_reward_enabled"],
        "visual_temperature": track_scores["visual_temperature"],
        "text_temperature": track_scores["text_temperature"],
        "semantic_layer": semantic_layer,
        "semantic_guidance_source": (
            "vit_layer_projected" if semantic_layer is not None else "default_v_proj"
        ),
        "V_routed": vision_routed,
        "idx_track1": idx_track1,
        "idx_track2": idx_track2,
        "route_k1_start_event": route_k1_start_event,
        "route_k1_end_event": route_k1_end_event,
        "route_k2_start_event": route_k2_start_event,
        "route_k2_end_event": route_k2_end_event,
    }
