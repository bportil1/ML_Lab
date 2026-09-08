from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


@dataclass(slots=True)
class TransformerPreparedData:
    original: np.ndarray
    tokens: np.ndarray
    element_mask: np.ndarray
    feature_names: list[str]
    input_shape: tuple[int, ...]
    scaler: Any
    token_width: int

    @property
    def token_count(self) -> int:
        return int(self.tokens.shape[1])

    @property
    def token_dim(self) -> int:
        return int(self.tokens.shape[2])


def _scaler(mode: str):
    if mode == "none":
        return None
    try:
        return {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
        }[mode]
    except KeyError as exc:
        raise ValueError(f"unsupported scaling mode: {mode}") from exc


def _validate(values: np.ndarray) -> None:
    if values.ndim not in (2, 3):
        raise ValueError("Transformer autoencoder input must be 2-D or 3-D")
    if values.shape[0] < 2:
        raise ValueError("Transformer autoencoder input must contain at least two samples")
    if values.shape[-1] < 1:
        raise ValueError("Transformer autoencoder input must contain at least one feature")
    if not np.isfinite(values).all():
        raise ValueError("X must contain only finite values; imputation is not implicit")


def prepare_transformer_data(
    X: Any,
    *,
    scaling: str,
    token_width: int,
) -> TransformerPreparedData:
    """Normalize generic vector/sequence input into ``(sample, token, feature)`` form.

    Two-dimensional matrices are split into adjacent feature tokens of ``token_width``.
    The final token is zero-padded when necessary and an element mask ensures padding
    never contributes to training loss. Three-dimensional tensors are treated as
    pre-tokenized sequences and are never padded by this adapter.
    """
    if token_width < 1:
        raise ValueError("token_width must be positive")

    if isinstance(X, pd.DataFrame):
        values = X.to_numpy(dtype=np.float32)
        feature_names = [str(column) for column in X.columns]
    else:
        values = np.asarray(X, dtype=np.float32)
        feature_names = []
    _validate(values)
    original = np.asarray(values, dtype=np.float32).copy()
    scaler = _scaler(scaling)

    if values.ndim == 2:
        if not feature_names:
            feature_names = [f"feature_{index}" for index in range(values.shape[1])]
        scaled = values if scaler is None else scaler.fit_transform(values).astype(np.float32)
        feature_count = int(scaled.shape[1])
        token_count = int(np.ceil(feature_count / token_width))
        padded_count = token_count * token_width
        padded = np.zeros((len(scaled), padded_count), dtype=np.float32)
        padded[:, :feature_count] = scaled
        element_mask = np.zeros_like(padded, dtype=bool)
        element_mask[:, :feature_count] = True
        return TransformerPreparedData(
            original=original,
            tokens=padded.reshape(len(scaled), token_count, token_width),
            element_mask=element_mask.reshape(len(scaled), token_count, token_width),
            feature_names=feature_names,
            input_shape=tuple(original.shape),
            scaler=scaler,
            token_width=token_width,
        )

    # Pre-tokenized sequence input: scale each token feature across all samples/tokens.
    samples, token_count, token_dim = values.shape
    flat = values.reshape(samples * token_count, token_dim)
    scaled_flat = flat if scaler is None else scaler.fit_transform(flat).astype(np.float32)
    tokens = np.asarray(scaled_flat, dtype=np.float32).reshape(samples, token_count, token_dim)
    if not feature_names:
        feature_names = [
            f"token_{token_index}_feature_{feature_index}"
            for token_index in range(token_count)
            for feature_index in range(token_dim)
        ]
    return TransformerPreparedData(
        original=original,
        tokens=tokens,
        element_mask=np.ones_like(tokens, dtype=bool),
        feature_names=feature_names,
        input_shape=tuple(original.shape),
        scaler=scaler,
        token_width=token_dim,
    )


def restore_transformer_reconstruction(
    prepared: TransformerPreparedData,
    reconstructed_tokens: np.ndarray,
) -> np.ndarray:
    reconstructed_tokens = np.asarray(reconstructed_tokens, dtype=np.float32)
    if reconstructed_tokens.shape != prepared.tokens.shape:
        raise ValueError(
            f"reconstructed token shape {reconstructed_tokens.shape} does not match "
            f"prepared shape {prepared.tokens.shape}"
        )

    if len(prepared.input_shape) == 2:
        feature_count = prepared.input_shape[1]
        scaled = reconstructed_tokens.reshape(len(reconstructed_tokens), -1)[:, :feature_count]
        restored = scaled if prepared.scaler is None else prepared.scaler.inverse_transform(scaled)
        return np.asarray(restored, dtype=float)

    samples, token_count, token_dim = prepared.input_shape
    flat = reconstructed_tokens.reshape(samples * token_count, token_dim)
    restored_flat = flat if prepared.scaler is None else prepared.scaler.inverse_transform(flat)
    return np.asarray(restored_flat, dtype=float).reshape(prepared.input_shape)
