from __future__ import annotations

import numpy as np


def reconstruction_metrics(reference, reconstruction) -> dict[str, float]:
    """Elementwise reconstruction metrics for matrices or higher-rank tensors."""
    reference = np.asarray(reference, dtype=float)
    reconstruction = np.asarray(reconstruction, dtype=float)
    if reference.shape != reconstruction.shape:
        raise ValueError("reference and reconstruction must have identical shapes")
    if reference.size == 0:
        raise ValueError("reference and reconstruction cannot be empty")
    error = reconstruction - reference
    mse = float(np.mean(np.square(error)))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(error))),
    }
