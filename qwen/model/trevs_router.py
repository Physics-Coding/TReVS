"""TReVS routing primitives for Qwen2.5-VL.

Stage 1 combines caller-supplied visual saliency with projector-space
text-to-vision cosine evidence. Qwen's active path supplies spatially merged
patch-to-patch attention; CLS-to-patch attention remains supported for callers
that provide a CLS token. Both signals become positive median/MAD evidence;
the route score is their maximum plus an optional geometric-mean consistency
reward. Score Top-K forms the first track, and FPS samples only from its
disjoint remainder. Final indices preserve the original visual-token order.

Stage 2 exposes the same all-head and high-variance priority-head scoring used
by the Qwen forward patch. The forward patch, rather than this module, owns
physical deletion and Qwen's M-RoPE, mask, and cache invariants.
"""

from typing import Dict, Optional, Tuple

import os

import torch
import torch.nn.functional as F


EPS = 1e-6
TREVS_TEXT_SCORE_MODE_ENV = "TREVS_TEXT_SCORE_MODE"
TREVS_VISUAL_TEMPERATURE_ENV = "TREVS_VISUAL_TEMPERATURE"
TREVS_TEXT_TEMPERATURE_ENV = "TREVS_TEXT_TEMPERATURE"
TREVS_PHASE_SCORING_ENV = "TREVS_PHASE_SCORING"
TREVS_USE_SINK_TOKEN_ENV = "TREVS_USE_SINK_TOKEN"
TREVS_USE_CONSISTENCY_REWARD_ENV = "TREVS_USE_CONSISTENCY_REWARD"
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
    raise ValueError(f"Invalid {name}={env_value!r}; expected a boolean value.")


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
    return _get_bool_env(TREVS_USE_CONSISTENCY_REWARD_ENV, True)


def use_sink_token() -> bool:
    return _get_bool_env(TREVS_USE_SINK_TOKEN_ENV, True)


def get_nonnegative_int_env(name: str, default: int) -> int:
    env_value = os.getenv(name, "").strip()
    if not env_value:
        return default
    try:
        value = int(env_value)
    except ValueError as exc:
        raise ValueError(f"Invalid {name}={env_value!r}; expected an integer.") from exc
    if value < 0:
        raise ValueError(f"Invalid {name}={env_value!r}; expected >= 0.")
    return value


def get_trevs_method() -> str:
    method = os.getenv("METHOD", "trevs").strip().lower()
    if method not in {"trevs", "dense"}:
        raise ValueError(f"Unsupported METHOD={method!r}; expected trevs or dense.")
    return method


def is_trevs_enabled() -> bool:
    return get_trevs_method() == "trevs"


def batch_index_select(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        B, N, C = x.shape
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        return x.reshape(B * N, C)[(idx + offset).reshape(-1)].reshape(B, idx.shape[1], C)
    if x.ndim == 2:
        B, N = x.shape
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        return x.reshape(B * N)[(idx + offset).reshape(-1)].reshape(B, idx.shape[1])
    raise NotImplementedError(f"batch_index_select only supports rank 2/3 tensors, got {x.ndim}.")


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
    B, N, D = features.shape
    device = features.device

    if n_samples == 0:
        return torch.empty(B, 0, dtype=torch.long, device=device)
    if N < n_samples:
        raise ValueError(f"Cannot FPS-sample {n_samples} points from only {N} candidates.")

    feat_norm = F.normalize(features.float(), p=2, dim=-1)
    idx_selected = torch.zeros(B, n_samples, dtype=torch.long, device=device)
    selected_mask = torch.zeros(B, N, dtype=torch.bool, device=device)

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

    for i in range(start_idx, n_samples):
        candidate_distances = min_distances.masked_fill(selected_mask, -1.0)
        next_idx = candidate_distances.argmax(dim=-1)
        idx_selected[:, i] = next_idx
        selected_mask.scatter_(1, next_idx.unsqueeze(-1), True)
        new_point = torch.gather(feat_norm, 1, next_idx.view(B, 1, 1).expand(B, 1, D))
        new_dist = 1.0 - (feat_norm * new_point).sum(dim=-1)
        min_distances = torch.min(min_distances, new_dist)

    return idx_selected


def _select_phase_topk_indices(vision_scores: torch.Tensor, n_keep: int) -> Tuple[torch.Tensor, torch.Tensor]:
    B, N_vis = vision_scores.shape
    n_keep = min(max(int(n_keep), 0), max(N_vis - 1, 0))
    if n_keep == 0:
        idx_keep = torch.empty(B, 0, dtype=torch.long, device=vision_scores.device)
    else:
        idx_keep = torch.topk(vision_scores, k=n_keep, dim=-1).indices
        idx_keep = torch.sort(idx_keep, dim=-1).values

    all_idx = torch.arange(N_vis, device=vision_scores.device).unsqueeze(0).expand(B, -1)
    keep_mask = torch.zeros(B, N_vis, dtype=torch.bool, device=vision_scores.device)
    if idx_keep.numel() > 0:
        keep_mask.scatter_(1, idx_keep, True)
    idx_drop = all_idx[~keep_mask].reshape(B, N_vis - n_keep)
    return idx_keep, idx_drop


def score_phase_attention(
    attn_tv: torch.Tensor,
    n_keep: int = 39,
    mode: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select visual tokens from decoder text-to-vision attention.

    Priority-head mode selects the half of heads with highest variance across
    visual keys, averages them, and takes maximum relevance over text queries.
    The selected indices are sorted only to restore source-token order.
    """
    if attn_tv.ndim != 4:
        raise ValueError(f"Expected attn_tv with shape [B, H, L_text, N_vis], got {tuple(attn_tv.shape)}.")

    phase_scoring = get_trevs_phase_scoring() if mode is None else mode.strip().lower()
    if phase_scoring not in TREVS_PHASE_SCORING_VALUES:
        raise ValueError(
            f"Invalid phase scoring mode {phase_scoring!r}; supported modes: {list(TREVS_PHASE_SCORING_VALUES)}."
        )

    if phase_scoring == "all_heads":
        vision_scores = attn_tv.mean(dim=1).max(dim=1).values
    else:
        B, H, L_text, N_vis = attn_tv.shape
        head_var = attn_tv.float().var(dim=-1).mean(dim=-1)
        n_priority = max(H // 2, 1)
        priority_heads = torch.topk(head_var, k=n_priority, dim=-1).indices
        gather_idx = priority_heads.unsqueeze(-1).unsqueeze(-1).expand(B, n_priority, L_text, N_vis)
        priority_attn = torch.gather(attn_tv, 1, gather_idx)
        vision_scores = priority_attn.mean(dim=1).max(dim=1).values

    return _select_phase_topk_indices(vision_scores, n_keep=n_keep)


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
    z_scores = (values - median) / (mad + eps)
    positive = F.relu(z_scores)
    positive = torch.where(abs_max <= eps, torch.zeros_like(positive), positive)
    if squeeze_output:
        return positive.squeeze(0)
    return positive


def _compute_saliency_raw_scores(vit_attn: torch.Tensor, n_vis: int) -> torch.Tensor:
    """Use CLS saliency when present, otherwise aggregate Qwen patch attention."""
    if vit_attn.ndim != 4 or vit_attn.shape[-1] != vit_attn.shape[-2]:
        raise ValueError(f"Unexpected vit_attn shape: {tuple(vit_attn.shape)}")
    attn_tokens = vit_attn.shape[-1]
    if attn_tokens == n_vis + 1:
        cls_patch_attn = vit_attn[:, :, 0, 1:]
        return cls_patch_attn.mean(dim=1)
    if attn_tokens == n_vis:
        patch_patch_attn = vit_attn
        return patch_patch_attn.mean(dim=1).sum(dim=1)
    raise ValueError(f"vit_attn token dimension {attn_tokens} does not match n_vis={n_vis}.")


def _normalize_text_token_mask(T_emb: torch.Tensor, core_token_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if core_token_mask is None:
        return torch.ones(T_emb.shape[:2], dtype=torch.bool, device=T_emb.device)
    if core_token_mask.shape != T_emb.shape[:2]:
        raise ValueError(
            f"core_token_mask shape {tuple(core_token_mask.shape)} does not match text shape {tuple(T_emb.shape[:2])}."
        )
    text_mask = core_token_mask.to(device=T_emb.device, dtype=torch.bool)
    empty_mask = text_mask.sum(dim=-1) == 0
    if empty_mask.any():
        text_mask = text_mask.clone()
        text_mask[empty_mask] = True
    return text_mask


def _compute_trevs_route_scores(
    vit_attn: torch.Tensor,
    V_proj: torch.Tensor,
    T_emb: torch.Tensor,
    core_token_mask: Optional[torch.Tensor],
    V_semantic_proj: Optional[torch.Tensor] = None,
    eps: float = EPS,
) -> Dict[str, torch.Tensor]:
    """Fuse visual saliency and projector-space text/vision semantic evidence."""
    B, N_vis, _ = V_proj.shape
    if T_emb.shape[0] != B:
        raise ValueError(f"Batch size mismatch between V_proj ({B}) and T_emb ({T_emb.shape[0]}).")
    if V_semantic_proj is None:
        V_semantic_proj = V_proj
    if V_semantic_proj.shape != V_proj.shape:
        raise ValueError(
            f"V_semantic_proj shape {tuple(V_semantic_proj.shape)} is incompatible with V_proj shape {tuple(V_proj.shape)}."
        )

    text_mask = _normalize_text_token_mask(T_emb=T_emb, core_token_mask=core_token_mask)
    device = V_proj.device
    text_score_mode = get_trevs_text_score_mode()
    saliency_scores = _compute_saliency_raw_scores(vit_attn, n_vis=N_vis).to(device=device)
    V_norm = F.normalize(V_semantic_proj.to(device=device).float(), p=2, dim=-1)
    T_norm = F.normalize(T_emb.float(), p=2, dim=-1)
    token_maps = torch.bmm(T_norm, V_norm.transpose(1, 2)).float().clamp_min(0.0)
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
    semantic_evidence = _build_positive_evidence_batch(
        semantic_scores.to(device=device), eps=eps
    ) / text_temperature
    consistency_reward_enabled = use_consistency_reward()
    consistency_reward = torch.sqrt(visual_evidence * semantic_evidence) if consistency_reward_enabled else 0.0
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


def trevs_route(
    vit_attn: torch.Tensor,
    V_proj: torch.Tensor,
    T_emb: torch.Tensor,
    n_topk: int,
    n_fps: int,
    core_token_mask: Optional[torch.Tensor],
    V_semantic_proj: Optional[torch.Tensor] = None,
    semantic_layer: Optional[int] = None,
    return_stats: bool = False,
):
    """Select score Top-K plus disjoint FPS tokens in source-token order."""
    _, N_vis, _ = V_proj.shape
    if n_topk < 0 or n_fps < 0:
        raise ValueError(f"TReVS token budgets must be >= 0, got {(n_topk, n_fps)}.")
    if n_topk + n_fps > N_vis:
        raise ValueError(
            f"Requested TReVS token budgets {(n_topk, n_fps)} exceed available visual tokens {N_vis}."
        )

    method = get_trevs_method()
    if method != "trevs":
        raise ValueError(f"trevs_route only supports METHOD=trevs, got {method!r}.")

    eps = torch.finfo(V_proj.float().dtype).eps
    scores = _compute_trevs_route_scores(
        vit_attn,
        V_proj,
        T_emb,
        core_token_mask,
        V_semantic_proj=V_semantic_proj,
        eps=eps,
    )
    effective_scores = scores["route_scores"].float() + EPS * scores["tie_break_scores"].float()
    idx_topk = torch.topk(effective_scores, k=n_topk, dim=-1).indices
    # Restore source-token order after score-based selection.
    idx_topk = torch.sort(idx_topk, dim=-1).values

    B = V_proj.shape[0]
    all_indices = torch.arange(N_vis, device=V_proj.device)
    selected_mask = torch.zeros(B, N_vis, dtype=torch.bool, device=V_proj.device)
    selected_mask.scatter_(1, idx_topk, True)
    idx_fps_list = []
    idx_routed_list = []
    for batch_idx in range(B):
        candidate_indices = all_indices[~selected_mask[batch_idx]]
        if n_fps == 0:
            idx_fps_b = torch.empty(0, dtype=torch.long, device=V_proj.device)
        else:
            candidate_features = V_proj[batch_idx, candidate_indices, :].unsqueeze(0)
            reference_features = V_proj[batch_idx, idx_topk[batch_idx], :].unsqueeze(0) if n_topk > 0 else None
            fps_local_idx = farthest_point_sampling(candidate_features, n_fps, reference_features=reference_features)
            idx_fps_b = candidate_indices[fps_local_idx.squeeze(0)]
        idx_fps_list.append(idx_fps_b)
        idx_routed_b = torch.sort(torch.cat([idx_topk[batch_idx], idx_fps_b], dim=0), dim=-1).values
        if torch.unique(idx_routed_b).numel() != idx_routed_b.numel():
            raise ValueError("TReVS routing produced overlapping Top-K and FPS selections.")
        idx_routed_list.append(idx_routed_b)

    idx_fps = torch.stack(idx_fps_list, dim=0)
    idx_routed = torch.stack(idx_routed_list, dim=0)
    V_routed = batch_index_select(V_proj, idx_routed)
    if idx_routed.shape[1] != n_topk + n_fps:
        raise ValueError("TReVS routing produced an unexpected number of routed tokens.")

    if return_stats:
        return idx_routed, {
            "selection_mode": scores["selection_mode"],
            "method": scores["method"],
            "text_score_mode": scores["text_score_mode"],
            "consistency_reward_enabled": scores["consistency_reward_enabled"],
            "visual_temperature": scores["visual_temperature"],
            "text_temperature": scores["text_temperature"],
            "semantic_layer": semantic_layer,
            "semantic_guidance_source": "vit_layer_merged" if semantic_layer is not None else "default_v_proj",
            "V_routed": V_routed,
            "idx_topk": idx_topk,
            "idx_fps": idx_fps,
        }
    return idx_routed
