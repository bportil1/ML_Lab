from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def reconstruction_metrics(reference, reconstruction) -> dict[str, float]:
    reference = np.asarray(reference, dtype=float)
    reconstruction = np.asarray(reconstruction, dtype=float)
    if reference.shape != reconstruction.shape:
        raise ValueError("reference and reconstruction must have identical shapes")
    mse = float(mean_squared_error(reference, reconstruction))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(reference, reconstruction)),
    }
