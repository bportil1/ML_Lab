from __future__ import annotations


def require_torch(*, purpose: str = "neural operations"):
    """Import PyTorch lazily so non-neural ML_Lab use stays lightweight."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised without neural extras
        raise RuntimeError(
            f"{purpose} require PyTorch: pip install 'ml-lab[neural]'"
        ) from exc
    return torch
