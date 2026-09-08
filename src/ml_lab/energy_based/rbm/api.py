from __future__ import annotations

from typing import Any

from .config import RBMConfig, canonical_family


def create(
    family: str,
    visible_dim: int,
    hidden_dim: int,
    *,
    sharpness: float = 1.0,
    dropout: float = 0.0,
    settings=None,
    device: str | None = None,
    dtype=None,
):
    """Construct a stable RBM family model without importing PyTorch until needed."""
    from ml_lab.neural.backend import require_torch
    from .models import BernoulliRBM, GaussianRBM, StudentTPoERBM

    torch = require_torch(purpose="RBM models")
    config = RBMConfig(
        family=family,
        visible_dim=visible_dim,
        hidden_dim=hidden_dim,
        sharpness=sharpness,
        dropout=dropout,
        settings=settings,
    )
    canonical = canonical_family(family)
    model_type = {
        "bernoulli": BernoulliRBM,
        "gaussian": GaussianRBM,
        "student_t_poe": StudentTPoERBM,
    }[canonical]
    model = model_type(
        config.visible_dim,
        config.hidden_dim,
        sharpness=config.sharpness,
        dropout=config.dropout,
        settings=config.resolved_settings(),
    )
    move: dict[str, Any] = {}
    if device is not None:
        move["device"] = torch.device(device)
    if dtype is not None:
        move["dtype"] = dtype
    if move:
        model = model.to(**move)
    return model


def create_from_config(config: RBMConfig, *, device: str | None = None, dtype=None):
    return create(
        config.family,
        config.visible_dim,
        config.hidden_dim,
        sharpness=config.sharpness,
        dropout=config.dropout,
        settings=config.settings,
        device=device,
        dtype=dtype,
    )


def run_chain(model, initial_visible, *, steps: int = 1):
    """Run a Gibbs chain and return the final visible tensor."""
    if int(steps) <= 0:
        raise ValueError("steps must be > 0")
    visible = model._coerce_visible(initial_visible)
    for _ in range(int(steps)):
        visible = model.gibbs_step(visible)
    return visible


def summarize(model, visible=None) -> dict[str, Any]:
    result: dict[str, Any] = {"config": model.configuration()}
    result["parameter_count"] = int(sum(parameter.numel() for parameter in model.parameters()))
    if visible is not None:
        energy = model.energy(visible).detach().cpu()
        positive, negative, gap = model.compute_energy_gap(visible)
        result.update(
            {
                "energy_mean": float(energy.mean()),
                "energy_std": float(energy.std(unbiased=False)),
                "positive_energy_mean": positive,
                "negative_energy_mean": negative,
                "energy_gap": gap,
            }
        )
    return result
