from __future__ import annotations

import numpy as np


def _wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if not len(a) or not len(b):
        raise ValueError("Wasserstein inputs cannot be empty")
    count = max(len(a), len(b))
    quantiles = (np.arange(count, dtype=float) + 0.5) / count
    qa = np.quantile(a, quantiles)
    qb = np.quantile(b, quantiles)
    return float(np.mean(np.abs(qa - qb)))


def distribution_metrics(real: np.ndarray, generated: np.ndarray) -> dict[str, float]:
    real = np.asarray(real, dtype=float)
    generated = np.asarray(generated, dtype=float)
    if real.ndim != 2 or generated.ndim != 2 or real.shape[1] != generated.shape[1]:
        raise ValueError("real and generated samples must be 2-D with the same feature count")
    wasserstein = [
        _wasserstein_1d(real[:, column], generated[:, column])
        for column in range(real.shape[1])
    ]
    return {
        "featurewise_wasserstein_mean": float(np.mean(wasserstein)),
        "featurewise_wasserstein_max": float(np.max(wasserstein)),
        "mean_absolute_mean_shift": float(np.mean(np.abs(real.mean(axis=0) - generated.mean(axis=0)))),
        "mean_absolute_std_shift": float(np.mean(np.abs(real.std(axis=0) - generated.std(axis=0)))),
    }
