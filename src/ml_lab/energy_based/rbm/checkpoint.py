from __future__ import annotations

from pathlib import Path

from .api import create


CHECKPOINT_FORMAT = "ml_lab.rbm.v1"


def save(model, path: str | Path) -> Path:
    from ml_lab.neural.backend import require_torch

    torch = require_torch(purpose="RBM checkpointing")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "config": model.configuration(),
        "state_dict": model.state_dict(),
    }
    torch.save(payload, path)
    return path


def load(path: str | Path, *, device: str = "cpu"):
    from ml_lab.neural.backend import require_torch

    torch = require_torch(purpose="RBM checkpointing")
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported RBM checkpoint format: {payload.get('format')!r}")
    config = dict(payload["config"])
    model = create(
        config["family"],
        config["visible_dim"],
        config["hidden_dim"],
        sharpness=config["sharpness"],
        dropout=config["dropout"],
        settings=config["settings"],
        device=device,
    )
    model.load_state_dict(payload["state_dict"])
    return model
