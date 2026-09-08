from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ml_lab.experimental.transformer_vae.config import (
    TransformerVAEConfig,
    TransformerVAETrainingConfig,
)

ContractiveEstimator = Literal["exact", "hutchinson"]


@dataclass(frozen=True, slots=True)
class ContractiveTransformerVAETrainingConfig(TransformerVAETrainingConfig):
    """Training controls for a contractive Transformer VAE.

    The contractive term penalizes the first-order Jacobian of the posterior mean
    ``mu(x)`` with respect to the model input. ``exact`` computes the Frobenius
    norm explicitly by latent dimension; ``hutchinson`` uses stochastic vector-
    Jacobian products to estimate the same squared Frobenius norm.
    """

    contractive_weight: float = 1e-3
    contractive_warmup_epochs: int = 0
    contractive_estimator: ContractiveEstimator = "exact"
    hutchinson_samples: int = 1
    normalize_by_active_inputs: bool = True

    def __post_init__(self) -> None:
        TransformerVAETrainingConfig.__post_init__(self)
        if self.contractive_weight < 0:
            raise ValueError("contractive_weight cannot be negative")
        if self.contractive_warmup_epochs < 0:
            raise ValueError("contractive_warmup_epochs cannot be negative")
        if self.contractive_estimator not in {"exact", "hutchinson"}:
            raise ValueError("contractive_estimator must be 'exact' or 'hutchinson'")
        if self.hutchinson_samples < 1:
            raise ValueError("hutchinson_samples must be positive")


__all__ = [
    "ContractiveEstimator",
    "ContractiveTransformerVAETrainingConfig",
    "TransformerVAEConfig",
]
