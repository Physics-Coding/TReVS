"""Self-contained Video-LLaVA inference package.

The package intentionally performs no model registration at import time.  In
particular, importing :mod:`videollava` must not change the AutoConfig entries
used by the existing LLaVA and Qwen implementations in this repository.
"""

__all__: list[str] = []
