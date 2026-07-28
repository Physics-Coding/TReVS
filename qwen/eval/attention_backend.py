import os
from typing import Optional


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _env_flag(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default).strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise ValueError(f"Unsupported {name}={value!r}. Expected 0/1 or a boolean value.")


def resolve_qwen_attention_backend() -> Optional[str]:
    method = os.environ.get("METHOD", "trevs").strip().lower()
    if method not in {"trevs", "dense"}:
        raise ValueError(f"Unsupported METHOD={method!r}. Expected trevs or dense.")

    use_flash_attn = _env_flag("USE_FLASH_ATTN")
    if method == "trevs":
        if use_flash_attn:
            raise ValueError(
                "TReVS requires USE_FLASH_ATTN=0 because its custom Qwen decoder uses a 4D additive "
                "attention mask. FlashAttention2 is not supported; use the SDPA backend."
            )
        return "sdpa"
    if use_flash_attn:
        return "flash_attention_2"
    return None


def configure_qwen_attention_backend(model_kwargs: dict) -> None:
    backend = resolve_qwen_attention_backend()
    if backend is not None:
        model_kwargs["attn_implementation"] = backend
