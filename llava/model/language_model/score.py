import torch
import os
import torch.nn.functional as F

def _get_nonnegative_int_env(name: str, default: int) -> int:
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

def _normalize_method_name(method_value: str) -> str:
    normalized = method_value.strip().lower()
    if not normalized:
        return "trevs"
    if normalized in {"trevs", "dense"}:
        return normalized
    raise ValueError(
        f"Unsupported METHOD={method_value!r}. This package only exposes 'trevs' and 'dense'."
    )


METHOD = _normalize_method_name(os.getenv("METHOD", "trevs"))
TREVS_ENABLED = METHOD == "trevs"
TREVS_ROUTE_TOPK = _get_nonnegative_int_env("TREVS_ROUTE_TOPK", 96)
TREVS_ROUTE_FPS = _get_nonnegative_int_env("TREVS_ROUTE_FPS", 32)
TREVS_TOTAL_TOKENS = TREVS_ROUTE_TOPK + TREVS_ROUTE_FPS
