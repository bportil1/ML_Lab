"""Stable dimensional-representation and compression operations.

Importing this module does not import PyTorch. Neural dependencies are loaded only
when a neural autoencoder path is explicitly invoked.
"""

from .api import autoencode, pca, run, transformer_autoencode
from .config import (
    AutoencoderTrainingConfig,
    MLPAutoencoderConfig,
    PCARepresentationConfig,
    TransformerAutoencoderConfig,
)
from .data import load_csv_features
from .pca import run_pca
from .results import RepresentationResult


def build_mlp_autoencoder(*args, **kwargs):
    from .autoencoder import build_mlp_autoencoder as _build

    return _build(*args, **kwargs)


def build_transformer_autoencoder(*args, **kwargs):
    from .transformer_autoencoder import build_transformer_autoencoder as _build

    return _build(*args, **kwargs)


def train_transformer_autoencoder(*args, **kwargs):
    from .transformer_training import train_transformer_autoencoder as _train

    return _train(*args, **kwargs)


def train_mlp_autoencoder(*args, **kwargs):
    from .training import train_mlp_autoencoder as _train

    return _train(*args, **kwargs)


__all__ = [
    "AutoencoderTrainingConfig",
    "MLPAutoencoderConfig",
    "PCARepresentationConfig",
    "TransformerAutoencoderConfig",
    "RepresentationResult",
    "autoencode",
    "build_mlp_autoencoder",
    "build_transformer_autoencoder",
    "load_csv_features",
    "pca",
    "run",
    "run_pca",
    "train_mlp_autoencoder",
    "train_transformer_autoencoder",
    "transformer_autoencode",
]
