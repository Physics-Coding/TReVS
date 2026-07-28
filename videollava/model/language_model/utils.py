"""Small tensor helpers shared by the isolated Video-LLaVA TReVS stack."""

import torch


def batch_index_select(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Select the sequence axis independently for every batch entry."""
    if x.ndim == 4:
        batch, heads, length, channels = x.shape
        new_length = idx.shape[1]
        offset = torch.arange(batch, dtype=torch.long, device=x.device).view(batch, 1) * length
        flat_idx = (idx + offset).reshape(-1)
        return x.reshape(batch * length, heads, channels)[flat_idx].reshape(
            batch, heads, new_length, channels
        )
    if x.ndim == 3:
        batch, length, channels = x.shape
        new_length = idx.shape[1]
        offset = torch.arange(batch, dtype=torch.long, device=x.device).view(batch, 1) * length
        flat_idx = (idx + offset).reshape(-1)
        return x.reshape(batch * length, channels)[flat_idx].reshape(batch, new_length, channels)
    if x.ndim == 2:
        batch, length = x.shape
        new_length = idx.shape[1]
        offset = torch.arange(batch, dtype=torch.long, device=x.device).view(batch, 1) * length
        flat_idx = (idx + offset).reshape(-1)
        return x.reshape(batch * length)[flat_idx].reshape(batch, new_length)
    raise NotImplementedError(f"batch_index_select does not support a rank-{x.ndim} tensor")
