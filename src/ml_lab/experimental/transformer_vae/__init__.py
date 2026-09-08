"""Experimental Transformer variational autoencoder.

The package is deliberately outside the stable representation API while the VAE
objective, latent diagnostics, and generative behavior mature.
"""

from .config import TransformerVAEConfig, TransformerVAETrainingConfig
from .model import build_transformer_vae
from .pipeline import run
from .training import TransformerVAETrainingResult, train_transformer_vae

__all__ = [
    "TransformerVAEConfig",
    "TransformerVAETrainingConfig",
    "TransformerVAETrainingResult",
    "build_transformer_vae",
    "train_transformer_vae",
    "run",
]
