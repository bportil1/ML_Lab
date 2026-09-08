from __future__ import annotations

from typing import Any

import numpy as np

from .binary_data import bytes_to_patch_tokens, bytes_to_rgb_array, read_binary


def _resolve_binary(*, data: Any = None, path: str | None = None) -> bytes | list[int]:
    if data is not None and path is not None:
        raise ValueError("provide either data or path, not both")
    if path is not None:
        return read_binary(path)
    if data is None:
        raise ValueError("binary data or path is required")
    return data


def run(
    *,
    mode: str = "describe",
    data: Any = None,
    path: str | None = None,
    image_size: int = 256,
    patch_size: int = 32,
    X: Any = None,
    config: dict[str, Any] | None = None,
    training: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Experimental entrypoint used by the ML Lab experimental registry."""
    if mode == "describe":
        return {
            "experiment": "linux_binary_identification",
            "status": "experimental",
            "source_concepts": [
                "byte-stream to RGB representation",
                "patch tokenization",
                "Transformer autoencoder compression",
                "latent-space binary analysis",
            ],
            "available_modes": ["describe", "bytes_to_rgb", "patch_tokens", "train_autoencoder"],
            "warnings": [
                "The Transformer path is an incubating rewrite of an unfinished legacy prototype.",
                "Binary-specific visualization and legacy clustering were intentionally not imported.",
            ],
        }

    if mode == "bytes_to_rgb":
        raw = _resolve_binary(data=data, path=path)
        array = bytes_to_rgb_array(raw, size=image_size)
        return {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "min": int(array.min()),
            "max": int(array.max()),
            "array": array.tolist(),
        }

    if mode == "patch_tokens":
        raw = _resolve_binary(data=data, path=path)
        tokens = bytes_to_patch_tokens(raw, image_size=image_size, patch_size=patch_size)
        return {
            "shape": list(tokens.shape),
            "dtype": str(tokens.dtype),
            "tokens": tokens.tolist(),
        }

    if mode == "train_autoencoder":
        if X is None:
            raise ValueError("X is required for train_autoencoder mode")
        from .transformer_autoencoder import TransformerAutoencoderConfig, train_array_autoencoder

        values = np.asarray(X, dtype=np.float32)
        cfg = TransformerAutoencoderConfig(**(config or {"token_dim": int(values.shape[-1])}))
        result = train_array_autoencoder(values, config=cfg, **(training or {}))
        return {
            "latent": result["latent"].tolist(),
            "history": result["history"],
            "config": result["config"],
            "device": result["device"],
        }

    raise ValueError(f"unknown linux_binary_identification mode: {mode}")
