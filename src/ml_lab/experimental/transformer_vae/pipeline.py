from __future__ import annotations

from typing import Any

import numpy as np


def run(
    *,
    mode: str = "describe",
    X: Any = None,
    model_config: dict[str, Any] | None = None,
    training_config: dict[str, Any] | None = None,
    prior_sample_count: int = 0,
) -> dict[str, Any]:
    if mode == "describe":
        return {
            "experiment": "transformer_vae",
            "status": "experimental",
            "source_concepts": [
                "Transformer sequence encoder/decoder",
                "diagonal-Gaussian variational latent space",
                "reparameterization trick",
                "beta-weighted KL regularization",
                "optional KL warm-up",
            ],
            "available_modes": ["describe", "fit_transform"],
            "warnings": [
                "This is an experimental baseline VAE, not part of the stable representation API.",
                "Legacy ICMI/TC/DWKL objectives are intentionally not bundled into this baseline.",
            ],
        }

    if mode != "fit_transform":
        raise ValueError(f"unknown transformer_vae mode: {mode}")
    if X is None:
        raise ValueError("X is required for fit_transform mode")

    from .config import TransformerVAEConfig, TransformerVAETrainingConfig
    from .training import train_transformer_vae

    values = np.asarray(X, dtype=np.float32)
    result = train_transformer_vae(
        values,
        model_config=TransformerVAEConfig(**(model_config or {})),
        training_config=TransformerVAETrainingConfig(**(training_config or {})),
        prior_sample_count=prior_sample_count,
    )
    record = result.to_record()
    record.update(
        {
            "posterior_mean": result.posterior_mean.tolist(),
            "posterior_logvar": result.posterior_logvar.tolist(),
            "reconstruction": result.reconstruction.tolist(),
            "prior_samples": (
                result.prior_samples.tolist() if result.prior_samples is not None else None
            ),
        }
    )
    return record
