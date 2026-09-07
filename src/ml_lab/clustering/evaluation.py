from __future__ import annotations

import math
from collections import Counter
from itertools import combinations
from typing import Any

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)


def _non_noise_view(X: Any, labels: Any) -> tuple[np.ndarray, np.ndarray]:
    X_arr = np.asarray(X)
    labels_arr = np.asarray(labels)
    keep = labels_arr != -1
    return X_arr[keep], labels_arr[keep]


def cluster_size_entropy(labels: Any) -> float:
    values = [label for label in np.asarray(labels).tolist() if label != -1]
    if not values:
        return 0.0
    counts = np.asarray(list(Counter(values).values()), dtype=float)
    probs = counts / counts.sum()
    return float(-(probs * np.log(probs)).sum())


def internal_metrics(X: Any, labels: Any) -> dict[str, float | int | None]:
    labels_arr = np.asarray(labels)
    X_eval, y_eval = _non_noise_view(X, labels_arr)
    unique = np.unique(y_eval)
    cluster_count = int(len(unique))
    noise_fraction = float(np.mean(labels_arr == -1)) if labels_arr.size else 0.0
    counts = Counter(y_eval.tolist())

    metrics: dict[str, float | int | None] = {
        "silhouette": None,
        "calinski_harabasz": None,
        "davies_bouldin": None,
        "cluster_count": cluster_count,
        "noise_fraction": noise_fraction,
        "cluster_size_entropy": cluster_size_entropy(labels_arr),
        "smallest_cluster": min(counts.values()) if counts else 0,
        "largest_cluster": max(counts.values()) if counts else 0,
    }
    if cluster_count < 2 or len(y_eval) <= cluster_count:
        return metrics
    metrics["silhouette"] = float(silhouette_score(X_eval, y_eval))
    metrics["calinski_harabasz"] = float(calinski_harabasz_score(X_eval, y_eval))
    metrics["davies_bouldin"] = float(davies_bouldin_score(X_eval, y_eval))
    return metrics


def external_metrics(y_true: Any, labels: Any) -> dict[str, float]:
    return {
        "adjusted_rand": float(adjusted_rand_score(y_true, labels)),
        "normalized_mutual_info": float(normalized_mutual_info_score(y_true, labels)),
        "homogeneity": float(homogeneity_score(y_true, labels)),
        "completeness": float(completeness_score(y_true, labels)),
        "v_measure": float(v_measure_score(y_true, labels)),
    }


def stability_score(label_runs: list[np.ndarray]) -> float | None:
    if len(label_runs) < 2:
        return None
    scores = [adjusted_rand_score(a, b) for a, b in combinations(label_runs, 2)]
    if not scores:
        return None
    value = float(np.mean(scores))
    return None if math.isnan(value) else value
