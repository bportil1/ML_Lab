from __future__ import annotations

import random
from typing import Any

import numpy as np

from .backend import require_torch
from .config import DeviceName


def resolve_device(requested: DeviceName = "auto", *, torch_module: Any | None = None):
    torch = torch_module or require_torch(purpose="neural device selection")
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if requested == "mps" and (
            not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available()
        ):
            raise RuntimeError("MPS was requested but is not available")
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(
    seed: int,
    *,
    deterministic: bool = True,
    torch_module: Any | None = None,
) -> None:
    """Seed Python, NumPy, and (when available) PyTorch consistently."""
    random.seed(seed)
    np.random.seed(seed)
    torch = torch_module
    if torch is None:
        try:
            torch = require_torch(purpose="neural deterministic seeding")
        except RuntimeError:
            return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic and getattr(torch.backends, "cudnn", None) is not None:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def torch_generator(seed: int, *, torch_module: Any | None = None):
    torch = torch_module or require_torch(purpose="neural data loading")
    return torch.Generator().manual_seed(seed)
