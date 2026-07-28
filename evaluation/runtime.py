"""Small runtime helpers shared by benchmark entry points."""

from __future__ import annotations

import os
import random


def seed_everything(seed: int | None = None) -> int:
    """Initialize Python, NumPy, Torch, and CUDA RNGs and return the seed."""

    if seed is None:
        seed = int(os.environ.get("RANDOM_SEED", "42"))
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
