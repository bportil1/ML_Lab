"""ML-UW-06: capability-declared, bounded sklearn representation explorer.

The existing PCA implementation remains the only PCA execution path. Other
estimators are ordinary sklearn methods, not alternate implementations. No
models are persisted and this module has no Flask, PAH, or SQLite dependency.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from .config import PCARepresentationConfig
from .pca import _feature_matrix, run_pca
from .evaluation import reconstruction_metrics


MAX_ROWS = 5000
MAX_FEATURES = 64
MAX_NONLINEAR_ROWS = 400
MAX_PLOT_POINTS = 240
SCALING = ("none", "standard", "minmax", "robust")


@dataclass(frozen=True)
class MethodSpec:
    id: str
    title: str
    description: str
    can_transform_new_data: bool
    can_inverse_transform: bool
    nonlinear: bool = False
    requires_nonnegative: bool = False
    max_rows: int = MAX_ROWS


METHODS = (
    MethodSpec("pca", "PCA", "Existing ML_Lab PCA implementation; linear variance-preserving projection.", True, True),
    MethodSpec("truncated_svd", "Truncated SVD", "Linear low-rank factorization without estimator-side centering.", True, True),
    MethodSpec("fast_ica", "FastICA", "Independent component analysis; convergence depends on the data.", True, True),
    MethodSpec("nmf", "NMF", "Non-negative matrix factorization; needs nonnegative scaled features.", True, True, requires_nonnegative=True),
    MethodSpec("kernel_pca", "Kernel PCA (RBF)", "Nonlinear kernel projection; fitted inverse is approximate.", True, True, nonlinear=True, max_rows=MAX_NONLINEAR_ROWS),
    MethodSpec("isomap", "Isomap", "Geodesic manifold embedding; no inverse transform in sklearn.", True, False, nonlinear=True, max_rows=MAX_NONLINEAR_ROWS),
    MethodSpec("tsne", "t-SNE", "Visualization-only fit embedding; no general transform or inverse.", False, False, nonlinear=True, max_rows=MAX_NONLINEAR_ROWS),
)
_METHOD_BY_ID = {spec.id: spec for spec in METHODS}


def method_catalog() -> list[dict[str, Any]]:
    """Return stable, public capability data for UI and downstream clients."""
    return [asdict(spec) for spec in METHODS]


def method_spec(method: str) -> MethodSpec:
    try:
        return _METHOD_BY_ID[method]
    except KeyError as exc:
        raise ValueError(f"Unknown representation method: {method}") from exc


def _scaler(mode: str):
    return {
        "none": None,
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }[mode]() if mode != "none" else None


def _estimator(spec: MethodSpec, components: int, rows: int):
    # Import optional sklearn submodules only when that method executes.
    if spec.id == "truncated_svd":
        from sklearn.decomposition import TruncatedSVD
        return TruncatedSVD(n_components=components, random_state=42)
    if spec.id == "fast_ica":
        from sklearn.decomposition import FastICA
        return FastICA(n_components=components, whiten="unit-variance", max_iter=500, random_state=42)
    if spec.id == "nmf":
        from sklearn.decomposition import NMF
        return NMF(n_components=components, init="nndsvda", random_state=42, max_iter=500)
    if spec.id == "kernel_pca":
        from sklearn.decomposition import KernelPCA
        return KernelPCA(n_components=components, kernel="rbf", fit_inverse_transform=True,
                         random_state=42)
    if spec.id == "isomap":
        from sklearn.manifold import Isomap
        return Isomap(n_components=components, n_neighbors=min(5, rows - 1))
    if spec.id == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(n_components=components, random_state=42, init="pca",
                    learning_rate="auto", perplexity=min(30.0, (rows - 1) / 3.0))
    raise ValueError(f"Unknown representation method: {spec.id}")


def _scatter(latent: np.ndarray, axis_x: int, axis_y: int) -> dict[str, Any] | None:
    if latent.shape[1] < 2:
        return None
    if not (0 <= axis_x < latent.shape[1] and 0 <= axis_y < latent.shape[1]) or axis_x == axis_y:
        raise ValueError("Plot axes must be different, valid latent dimensions")
    indexes = np.linspace(0, len(latent) - 1, min(len(latent), MAX_PLOT_POINTS), dtype=int)
    pairs = np.asarray(latent[indexes][:, (axis_x, axis_y)], dtype=float)
    if not np.isfinite(pairs).all():
        raise ValueError("Embedding contains nonfinite plotting coordinates")
    lo = pairs.min(axis=0)
    hi = pairs.max(axis=0)
    lengths = np.where(hi > lo, hi - lo, 1.0)
    points = [
        {"row": int(idx) + 1, "x": round(float(38 + 522 * (pair[0] - lo[0]) / lengths[0]), 2),
         "y": round(float(282 - 246 * (pair[1] - lo[1]) / lengths[1]), 2)}
        for idx, pair in zip(indexes, pairs)
    ]
    return {"points": points, "axes": [axis_x + 1, axis_y + 1],
            "x_range": [float(lo[0]), float(hi[0])],
            "y_range": [float(lo[1]), float(hi[1])],
            "total_rows": int(len(latent)), "sampled": len(indexes) < len(latent)}


def explore_representation(
    matrix: Any, *, method: str = "pca", components: int = 2,
    scaling: str = "standard", reconstruct: bool = False,
    axis_x: int = 0, axis_y: int = 1, include_geometry: bool = False,
) -> dict[str, Any]:
    """Fit a declared method and return a bounded display-only report.

    This is not a model-experiment run or a saved fitted transformer. The
    caller verifies the selected source before and after materialization.
    """
    spec = method_spec(method)
    if scaling not in SCALING:
        raise ValueError("Unknown feature scaling option")
    if type(components) is not int or components < 1:
        raise ValueError("Number of components must be a positive integer")
    if type(reconstruct) is not bool:
        raise ValueError("reconstruct must be a boolean")
    if reconstruct and not spec.can_inverse_transform:
        raise ValueError(f"{spec.title} does not support inverse transformation")
    values, names = _feature_matrix(matrix)
    rows, columns = values.shape
    if rows > spec.max_rows or columns > MAX_FEATURES:
        raise ValueError(f"{spec.title} is limited to {spec.max_rows} rows and {MAX_FEATURES} features; select a smaller dataset explicitly")
    if components > min(rows, columns) or (spec.id in {"isomap", "kernel_pca"} and components >= rows):
        raise ValueError("Components exceed the valid rank/dimension limit for the selected method")
    if spec.id == "tsne":
        if rows < 4:
            raise ValueError("t-SNE requires at least four rows")
        if components > 3:
            raise ValueError("t-SNE Barnes-Hut supports at most three latent dimensions")
    if spec.requires_nonnegative and scaling not in {"none", "minmax"}:
        raise ValueError("NMF supports only none or minmax scaling to preserve nonnegativity")
    if spec.id == "pca":
        result = run_pca(matrix, config=PCARepresentationConfig(n_components=components, scaling=scaling))
        latent = result.latent
        metrics = result.metrics if reconstruct else {}
        metadata = {"explained_variance_ratio_sum": result.metrics["explained_variance_ratio_sum"]}
    else:
        scaler = _scaler(scaling)
        scaled = scaler.fit_transform(values) if scaler is not None else values
        if spec.requires_nonnegative and np.min(scaled) < 0:
            raise ValueError("NMF requires nonnegative features; choose minmax scaling")
        estimator = _estimator(spec, components, rows)
        latent = np.asarray(estimator.fit_transform(scaled), dtype=float)
        if not np.isfinite(latent).all():
            raise ValueError("Fitted embedding contains nonfinite coordinates")
        metadata = {}
        if hasattr(estimator, "explained_variance_ratio_"):
            metadata["explained_variance_ratio_sum"] = float(np.sum(estimator.explained_variance_ratio_))
        metrics = {}
        if reconstruct:
            # Capabilities are enforced by metadata AND implementation; never
            # pretend an embedding has a meaningful inverse when it does not.
            inverse = getattr(estimator, "inverse_transform", None)
            if not callable(inverse):
                raise ValueError(f"{spec.title} does not expose an inverse transformer")
            reconstructed_scaled = np.asarray(inverse(latent), dtype=float)
            reconstructed = scaler.inverse_transform(reconstructed_scaled) if scaler is not None else reconstructed_scaled
            metrics = reconstruction_metrics(values, np.asarray(reconstructed, dtype=float))
    if not np.isfinite(latent).all():
        raise ValueError("Fitted embedding contains nonfinite coordinates")
    geometry = None
    if include_geometry:
        # Import only on demand. The comparison reuses the fitted embedding,
        # not a second estimator or projection pipeline.
        from .comparison_workbench import _scaled, geometry_metrics
        geometry = geometry_metrics(_scaled(values, scaling), latent)
    return {
        "schema": "ml-lab.representation-explorer@1",
        "method": spec.id, "method_title": spec.title,
        "capabilities": {"transform_new_data": spec.can_transform_new_data,
                         "inverse_transform": spec.can_inverse_transform},
        "shape": [rows, columns], "latent_shape": list(latent.shape),
        "feature_names": names, "scaling": scaling, "requested_components": components,
        "reconstruction_requested": reconstruct, "metrics": metrics, "metadata": metadata,
        "preview": [{"row": int(idx) + 1, "values": [round(float(v), 7) for v in latent[idx]]}
                    for idx in range(min(12, rows))],
        "scatter": _scatter(latent, axis_x, axis_y),
        "saved_model": False, "geometry": geometry,
    }
