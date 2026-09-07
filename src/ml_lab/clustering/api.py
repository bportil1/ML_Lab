from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .config import ClusteringSearchConfig
from .selection import ClusteringResult, run_cluster_suite


def run(
    X: pd.DataFrame | np.ndarray,
    *,
    estimators: Iterable[str] = ("kmeans", "agglomerative", "spectral", "birch", "gmm"),
    config: ClusteringSearchConfig | None = None,
    y_true: pd.Series | np.ndarray | None = None,
) -> list[ClusteringResult]:
    return run_cluster_suite(estimators, X, config, y_true=y_true)
