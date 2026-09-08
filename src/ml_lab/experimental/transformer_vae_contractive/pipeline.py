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
            "experiment": "transformer_vae_contractive",
            "status": "experimental",
            "objective": "reconstruction + beta*KL + lambda*||d mu(x)/d x||_F^2",
            "contractive_target": "posterior_mean_jacobian_wrt_input",
            "available_estimators": ["exact", "hutchinson"],
            "available_modes": ["describe", "fit_transform"],
            "source_concepts": [
                "contractive autoencoder first-order Jacobian regularization",
                "Transformer variational autoencoder",
                "local tangent/first-order representation sensitivity",
            ],
            "warnings": [
                "This is an experimental contractive VAE, not part of the stable representation API.",
                "Exact Jacobian penalties require one vector-Jacobian product per latent dimension and can be expensive.",
                "The contractive term regularizes the encoder posterior mean map, not decoder manifold curvature.",
            ],
        }

    if mode != "fit_transform":
        raise ValueError(f"unknown transformer_vae_contractive mode: {mode}")
    if X is None:
        raise ValueError("X is required for fit_transform mode")

    from .config import ContractiveTransformerVAETrainingConfig, TransformerVAEConfig
    from .training import train_contractive_transformer_vae

    values = np.asarray(X, dtype=np.float32)
    result = train_contractive_transformer_vae(
        values,
        model_config=TransformerVAEConfig(**(model_config or {})),
        training_config=ContractiveTransformerVAETrainingConfig(**(training_config or {})),
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
