from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from .config import PCARepresentationConfig
from .evaluation import reconstruction_metrics
from .results import RepresentationResult


def _feature_matrix(X: Any) -> tuple[np.ndarray, list[str]]:
    if isinstance(X, pd.DataFrame):
        names = [str(column) for column in X.columns]
        values = X.to_numpy(dtype=float)
    else:
        values = np.asarray(X, dtype=float)
        if values.ndim != 2:
            raise ValueError("X must be a two-dimensional feature matrix")
        names = [f"feature_{index}" for index in range(values.shape[1])]
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("X must contain at least two rows and one feature")
    if not np.isfinite(values).all():
        raise ValueError("X must contain only finite values; imputation is not implicit")
    return values, names


def run_pca(
    X: Any,
    *,
    config: PCARepresentationConfig | None = None,
) -> RepresentationResult:
    config = config or PCARepresentationConfig()
    values, feature_names = _feature_matrix(X)

    steps: list[tuple[str, Any]] = []
    if config.scaling != "none":
        scaler = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
        }[config.scaling]
        steps.append(("scale", scaler))

    reducer = PCA(n_components=config.n_components, random_state=config.random_state)
    steps.append(("pca", reducer))
    pipeline = Pipeline(steps)
    latent = np.asarray(pipeline.fit_transform(values), dtype=float)

    pca = pipeline.named_steps["pca"]
    transformed_reconstruction = pca.inverse_transform(latent)
    if "scale" in pipeline.named_steps:
        reconstruction = pipeline.named_steps["scale"].inverse_transform(transformed_reconstruction)
    else:
        reconstruction = transformed_reconstruction
    reconstruction = np.asarray(reconstruction, dtype=float)

    metrics = reconstruction_metrics(values, reconstruction)
    metrics["explained_variance_ratio_sum"] = float(np.sum(pca.explained_variance_ratio_))
    return RepresentationResult(
        method="pca",
        latent=latent,
        reconstruction=reconstruction,
        metrics=metrics,
        feature_names=feature_names,
        model=pca,
        transformer=pipeline,
        metadata={
            "n_components": int(latent.shape[1]),
            "requested_n_components": config.n_components,
            "scaling": config.scaling,
            "explained_variance_ratio": np.asarray(pca.explained_variance_ratio_, dtype=float).tolist(),
        },
    )
