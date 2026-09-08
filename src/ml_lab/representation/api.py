from __future__ import annotations

from typing import Any, Iterable, Literal

from .config import (
    AutoencoderTrainingConfig,
    MLPAutoencoderConfig,
    PCARepresentationConfig,
    TransformerAutoencoderConfig,
)
from .pca import run_pca
from .results import RepresentationResult

RepresentationMethod = Literal["pca", "mlp_autoencoder", "transformer_autoencoder"]


def pca(X: Any, *, config: PCARepresentationConfig | None = None) -> RepresentationResult:
    return run_pca(X, config=config)


def autoencode(
    X: Any,
    *,
    model_config: MLPAutoencoderConfig | None = None,
    training_config: AutoencoderTrainingConfig | None = None,
    callbacks: Iterable[Any] | None = None,
) -> RepresentationResult:
    from .training import train_mlp_autoencoder

    return train_mlp_autoencoder(
        X,
        model_config=model_config,
        training_config=training_config,
        callbacks=callbacks,
    )


def transformer_autoencode(
    X: Any,
    *,
    model_config: TransformerAutoencoderConfig | None = None,
    training_config: AutoencoderTrainingConfig | None = None,
    callbacks: Iterable[Any] | None = None,
) -> RepresentationResult:
    from .transformer_training import train_transformer_autoencoder

    return train_transformer_autoencoder(
        X,
        model_config=model_config,
        training_config=training_config,
        callbacks=callbacks,
    )


def run(
    X: Any,
    *,
    method: RepresentationMethod = "pca",
    pca_config: PCARepresentationConfig | None = None,
    model_config: MLPAutoencoderConfig | None = None,
    transformer_config: TransformerAutoencoderConfig | None = None,
    training_config: AutoencoderTrainingConfig | None = None,
    callbacks: Iterable[Any] | None = None,
) -> RepresentationResult:
    if method == "pca":
        return pca(X, config=pca_config)
    if method == "mlp_autoencoder":
        return autoencode(
            X,
            model_config=model_config,
            training_config=training_config,
            callbacks=callbacks,
        )
    if method == "transformer_autoencoder":
        return transformer_autoencode(
            X,
            model_config=transformer_config,
            training_config=training_config,
            callbacks=callbacks,
        )
    raise ValueError(f"unsupported representation method: {method}")
