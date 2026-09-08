from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class TransformerAutoencoderConfig:
    """Compatibility configuration for the incubating binary experiment.

    Generic Transformer autoencoding graduated to ``ml_lab.representation`` in
    ML_Lab 0.7.0. This wrapper preserves the experimental API while delegating
    model/training behavior to the stable implementation.
    """

    token_dim: int
    model_dim: int = 128
    latent_dim: int = 64
    nhead: int = 4
    encoder_layers: int = 2
    decoder_layers: int = 2
    feedforward_dim: int = 256
    dropout: float = 0.0
    max_tokens: int = 1024

    def validate(self) -> None:
        if self.token_dim <= 0:
            raise ValueError("token_dim must be positive")
        _stable_config(self)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def _stable_config(config: TransformerAutoencoderConfig):
    from ml_lab.representation import TransformerAutoencoderConfig as StableConfig

    return StableConfig(
        token_width=1,
        model_dim=config.model_dim,
        latent_dim=config.latent_dim,
        nhead=config.nhead,
        encoder_layers=config.encoder_layers,
        decoder_layers=config.decoder_layers,
        feedforward_dim=config.feedforward_dim,
        dropout=config.dropout,
        max_tokens=config.max_tokens,
    )


def build_transformer_autoencoder(config: TransformerAutoencoderConfig):
    config.validate()
    from ml_lab.representation import build_transformer_autoencoder as build_stable

    return build_stable(config.token_dim, _stable_config(config))


def train_array_autoencoder(
    X: np.ndarray,
    *,
    config: TransformerAutoencoderConfig | None = None,
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    seed: int = 42,
    device: str | None = None,
) -> dict[str, Any]:
    """Compatibility entrypoint backed by the stable Transformer trainer."""
    values = np.asarray(X, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("X must have shape (samples, tokens, token_dim)")
    if config is None:
        config = TransformerAutoencoderConfig(
            token_dim=int(values.shape[-1]),
            max_tokens=max(1, int(values.shape[1])),
        )
    config.validate()
    if int(values.shape[-1]) != config.token_dim:
        raise ValueError("X token dimension does not match config.token_dim")

    from ml_lab.representation import AutoencoderTrainingConfig, transformer_autoencode

    result = transformer_autoencode(
        values,
        model_config=_stable_config(config),
        training_config=AutoencoderTrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            validation_fraction=0.0,
            patience=None,
            scaling="none",
            random_state=seed,
            device=(device or "auto"),
        ),
    )
    return {
        "model": result.model,
        "latent": result.latent,
        "history": [float(row["train_mse"]) for row in result.history],
        "config": config.to_record(),
        "device": result.metadata["device"],
    }
