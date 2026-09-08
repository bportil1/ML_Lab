"""Experimental Transformer VAE with decomposed ICMI/TC/DWKL regularization."""

from .config import DecomposedTransformerVAETrainingConfig, TransformerVAEConfig
from .objectives import DecomposedKLEstimate, estimate_decomposed_kl
from .pipeline import run
from .training import (
    DecomposedTransformerVAETrainingResult,
    train_decomposed_transformer_vae,
)

__all__ = [
    "TransformerVAEConfig",
    "DecomposedTransformerVAETrainingConfig",
    "DecomposedKLEstimate",
    "estimate_decomposed_kl",
    "DecomposedTransformerVAETrainingResult",
    "train_decomposed_transformer_vae",
    "run",
]
