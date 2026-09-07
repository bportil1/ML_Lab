from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

ScaleMethod = Literal["standard", "minmax", "robust"]


@dataclass(slots=True)
class TransformResult:
    data: np.ndarray
    transformer: Any


@dataclass(slots=True)
class PCAResult(TransformResult):
    explained_variance_ratio: np.ndarray


def scale(X: Any, method: ScaleMethod = "standard") -> TransformResult:
    transformer = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }[method]
    return TransformResult(np.asarray(transformer.fit_transform(X)), transformer)


def impute(
    X: Any,
    *,
    strategy: Literal["mean", "median", "most_frequent", "constant"] = "median",
    fill_value: Any = None,
) -> TransformResult:
    transformer = SimpleImputer(strategy=strategy, fill_value=fill_value)
    return TransformResult(np.asarray(transformer.fit_transform(X)), transformer)


def pca(
    X: Any,
    n_components: int | float | None = 2,
    *,
    scaling: Literal["none", "standard", "minmax", "robust"] = "standard",
    random_state: int = 42,
) -> PCAResult:
    steps: list[tuple[str, Any]] = []
    if scaling != "none":
        scaler = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
        }[scaling]
        steps.append(("scale", scaler))
    reducer = PCA(n_components=n_components, random_state=random_state)
    steps.append(("pca", reducer))
    transformer = Pipeline(steps)
    data = np.asarray(transformer.fit_transform(X))
    fitted_pca = transformer.named_steps["pca"]
    return PCAResult(data, transformer, np.asarray(fitted_pca.explained_variance_ratio_))
