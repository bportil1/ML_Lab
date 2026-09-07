from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.cluster import (
    AgglomerativeClustering,
    Birch,
    DBSCAN,
    KMeans,
    MiniBatchKMeans,
    OPTICS,
    SpectralClustering,
)
from sklearn.mixture import GaussianMixture

from ml_lab.core.specs import EstimatorSpec


def _kmeans(seed: int) -> BaseEstimator:
    return KMeans(random_state=seed, n_init="auto")


def _minibatch(seed: int) -> BaseEstimator:
    return MiniBatchKMeans(random_state=seed, n_init="auto")


def _agglomerative(_: int) -> BaseEstimator:
    return AgglomerativeClustering()


def _spectral(seed: int) -> BaseEstimator:
    return SpectralClustering(random_state=seed)


def _birch(_: int) -> BaseEstimator:
    return Birch()


def _dbscan(_: int) -> BaseEstimator:
    return DBSCAN()


def _optics(_: int) -> BaseEstimator:
    return OPTICS()


def _gmm(seed: int) -> BaseEstimator:
    return GaussianMixture(random_state=seed)


_REGISTRY: dict[str, EstimatorSpec] = {
    "kmeans": EstimatorSpec(
        "kmeans", "K-Means", "clustering", _kmeans,
        [{"n_clusters": [2, 3, 4, 5], "init": ["k-means++", "random"]}],
        preprocess="standard",
    ),
    "minibatch_kmeans": EstimatorSpec(
        "minibatch_kmeans", "MiniBatch K-Means", "clustering", _minibatch,
        [{"n_clusters": [2, 3, 4, 5], "batch_size": [32, 64, 128]}],
        preprocess="standard",
    ),
    "agglomerative": EstimatorSpec(
        "agglomerative", "Agglomerative Clustering", "clustering", _agglomerative,
        [
            {"n_clusters": [2, 3, 4, 5], "linkage": ["ward"]},
            {"n_clusters": [2, 3, 4, 5], "linkage": ["complete", "average", "single"]},
        ],
        preprocess="standard",
    ),
    "spectral": EstimatorSpec(
        "spectral", "Spectral Clustering", "clustering", _spectral,
        [{"n_clusters": [2, 3, 4, 5], "affinity": ["rbf", "nearest_neighbors"], "assign_labels": ["kmeans", "cluster_qr"]}],
        preprocess="standard",
    ),
    "birch": EstimatorSpec(
        "birch", "BIRCH", "clustering", _birch,
        [{"n_clusters": [2, 3, 4, 5], "threshold": [0.25, 0.5, 0.75]}],
        preprocess="standard",
    ),
    "dbscan": EstimatorSpec(
        "dbscan", "DBSCAN", "clustering", _dbscan,
        [{"eps": [0.25, 0.5, 1.0], "min_samples": [3, 5, 10]}],
        preprocess="standard",
        notes="Noise points are labeled -1 and excluded from internal geometry metrics.",
    ),
    "optics": EstimatorSpec(
        "optics", "OPTICS", "clustering", _optics,
        [{"min_samples": [3, 5, 10], "xi": [0.03, 0.05, 0.1], "min_cluster_size": [None, 0.05, 0.1]}],
        preprocess="standard",
        notes="Noise points are labeled -1 and excluded from internal geometry metrics.",
    ),
    "gmm": EstimatorSpec(
        "gmm", "Gaussian Mixture", "clustering", _gmm,
        [{"n_components": [2, 3, 4, 5], "covariance_type": ["full", "tied", "diag"]}],
        preprocess="standard",
    ),
}


def list_estimators() -> list[EstimatorSpec]:
    return [_REGISTRY[key] for key in sorted(_REGISTRY)]


def get_estimator_spec(estimator_id: str) -> EstimatorSpec:
    try:
        return _REGISTRY[estimator_id]
    except KeyError as exc:
        supported = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"unknown clustering estimator {estimator_id!r}; supported: {supported}") from exc
