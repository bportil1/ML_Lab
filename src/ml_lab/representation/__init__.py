"""Stable dimensional-representation and compression operations.

Importing this module does not import PyTorch. Neural dependencies are loaded only
when the MLP autoencoder path is explicitly invoked.
"""

from .api import autoencode, pca, run
from .config import AutoencoderTrainingConfig, MLPAutoencoderConfig, PCARepresentationConfig
from .data import load_csv_features
from .pca import run_pca
from .results import RepresentationResult


def build_mlp_autoencoder(*args, **kwargs):
    from .autoencoder import build_mlp_autoencoder as _build

    return _build(*args, **kwargs)


def train_mlp_autoencoder(*args, **kwargs):
    from .training import train_mlp_autoencoder as _train

    return _train(*args, **kwargs)


__all__ = [
    "AutoencoderTrainingConfig",
    "MLPAutoencoderConfig",
    "PCARepresentationConfig",
    "RepresentationResult",
    "autoencode",
    "build_mlp_autoencoder",
    "load_csv_features",
    "pca",
    "run",
    "run_pca",
    "train_mlp_autoencoder",
]
