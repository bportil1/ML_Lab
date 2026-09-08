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
            "experiment": "transformer_vae_decomposed",
            "status": "experimental",
            "source_concepts": [
                "Transformer variational autoencoder",
                "index-code mutual information (ICMI)",
                "total correlation (TC)",
                "dimension-wise KL (DWKL)",
                "beta-TCVAE-style KL decomposition",
            ],
            "objective": "reconstruction + alpha*ICMI + beta*TC + gamma*DWKL",
            "density_estimator": "minibatch_mixture",
            "available_modes": ["describe", "fit_transform"],
            "warnings": [
                "ICMI/TC/DWKL are minibatch density estimates, not exact full-dataset quantities.",
                "The pairwise density estimator is quadratic in minibatch size and linear in latent dimension.",
                "This research objective remains outside the stable representation API.",
            ],
        }

    if mode != "fit_transform":
        raise ValueError(f"unknown transformer_vae_decomposed mode: {mode}")
    if X is None:
        raise ValueError("X is required for fit_transform mode")

    from .config import DecomposedTransformerVAETrainingConfig, TransformerVAEConfig
    from .training import train_decomposed_transformer_vae

    values = np.asarray(X, dtype=np.float32)
    result = train_decomposed_transformer_vae(
        values,
        model_config=TransformerVAEConfig(**(model_config or {})),
        training_config=DecomposedTransformerVAETrainingConfig(**(training_config or {})),
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
