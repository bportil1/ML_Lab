from __future__ import annotations

from typing import Any, Callable

from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    OrthogonalMatchingPursuit,
    QuantileRegressor,
    Ridge,
    SGDRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor

from ml_lab.core.specs import EstimatorSpec


def _const(factory: Callable[[], BaseEstimator]) -> Callable[[int], BaseEstimator]:
    return lambda _seed: factory()


def _ridge(_: int) -> BaseEstimator:
    return Ridge(max_iter=5000)


def _sgd(seed: int) -> BaseEstimator:
    return SGDRegressor(random_state=seed, max_iter=5000, tol=1e-3)


def _dt(seed: int) -> BaseEstimator:
    return DecisionTreeRegressor(random_state=seed)


def _rf(seed: int) -> BaseEstimator:
    return RandomForestRegressor(random_state=seed, n_jobs=1)


def _etc(seed: int) -> BaseEstimator:
    return ExtraTreesRegressor(random_state=seed, n_jobs=1)


def _lasso(seed: int) -> BaseEstimator:
    return Lasso(random_state=seed, max_iter=5000)


def _elastic(seed: int) -> BaseEstimator:
    return ElasticNet(random_state=seed, max_iter=5000)


def _linear_svr(seed: int) -> BaseEstimator:
    return LinearSVR(random_state=seed, max_iter=5000)


# The initial registry is adapted from the supplied dimensional-mapping experiments,
# but search spaces are deliberately bounded so ML Lab remains a practical callable
# service instead of reproducing the legacy repository's very large hand-written grids.
_REGISTRY: dict[str, EstimatorSpec] = {
    "linear": EstimatorSpec(
        "linear", "Linear Regression", "regression", _const(LinearRegression), [{}]
    ),
    "ridge": EstimatorSpec(
        "ridge", "Ridge Regression", "regression", _ridge,
        [{"alpha": [0.1, 1.0, 10.0, 100.0], "tol": [1e-4, 1e-3, 1e-2]}],
        preprocess="standard",
    ),
    "sgd": EstimatorSpec(
        "sgd", "SGD Regressor", "regression", _sgd,
        [{
            "alpha": [1e-4, 1e-3, 1e-2],
            "penalty": ["l2", "l1", "elasticnet"],
            "loss": ["squared_error", "huber"],
            "tol": [1e-4, 1e-3],
        }],
        preprocess="standard",
    ),
    "decision_tree": EstimatorSpec(
        "decision_tree", "Decision Tree Regressor", "regression", _dt,
        [{
            "criterion": ["squared_error", "friedman_mse", "absolute_error"],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }],
    ),
    "lasso": EstimatorSpec(
        "lasso", "Lasso Regression", "regression", _lasso,
        [{"alpha": [0.001, 0.01, 0.1, 1.0, 10.0], "tol": [1e-4, 1e-3]}],
        preprocess="standard",
    ),
    "elasticnet": EstimatorSpec(
        "elasticnet", "Elastic Net", "regression", _elastic,
        [{
            "alpha": [0.001, 0.01, 0.1, 1.0],
            "l1_ratio": [0.1, 0.5, 0.9],
            "tol": [1e-4, 1e-3],
        }],
        preprocess="standard",
    ),
    "random_forest": EstimatorSpec(
        "random_forest", "Random Forest Regressor", "regression", _rf,
        [{
            "n_estimators": [100, 200],
            "max_features": ["sqrt", "log2", 1.0],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        }],
    ),
    "extra_trees": EstimatorSpec(
        "extra_trees", "Extra Trees Regressor", "regression", _etc,
        [{
            "n_estimators": [100, 200],
            "max_features": ["sqrt", "log2", 1.0],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        }],
    ),
    "kernel_ridge": EstimatorSpec(
        "kernel_ridge", "Kernel Ridge", "regression", _const(KernelRidge),
        [{"alpha": [0.1, 1.0, 10.0], "kernel": ["linear", "rbf", "poly"]}],
        preprocess="standard",
    ),
    "omp": EstimatorSpec(
        "omp", "Orthogonal Matching Pursuit", "regression", _const(OrthogonalMatchingPursuit),
        [{"n_nonzero_coefs": [None, 1, 3, 5]}],
        preprocess="standard",
    ),
    "knn": EstimatorSpec(
        "knn", "K-Nearest Neighbors Regressor", "regression",
        lambda _seed: KNeighborsRegressor(n_jobs=1),
        [{
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        }],
        preprocess="standard",
    ),
    "svr": EstimatorSpec(
        "svr", "Support Vector Regressor", "regression", _const(SVR),
        [
            {"kernel": ["linear"], "C": [0.1, 1.0, 10.0], "epsilon": [0.01, 0.1, 0.2]},
            {"kernel": ["rbf"], "C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"], "epsilon": [0.01, 0.1, 0.2]},
        ],
        preprocess="standard",
    ),
    "linear_svr": EstimatorSpec(
        "linear_svr", "Linear SVR", "regression", _linear_svr,
        [{"C": [0.1, 1.0, 10.0], "epsilon": [0.0, 0.1, 0.2], "tol": [1e-4, 1e-3]}],
        preprocess="standard",
    ),
    "huber": EstimatorSpec(
        "huber", "Huber Regressor", "regression", _const(HuberRegressor),
        [{"epsilon": [1.1, 1.35, 1.5, 2.0], "alpha": [1e-4, 1e-3, 1e-2]}],
        preprocess="standard",
    ),
    "quantile": EstimatorSpec(
        "quantile", "Quantile Regressor", "regression", _const(QuantileRegressor),
        [{"quantile": [0.25, 0.5, 0.75], "alpha": [0.0, 0.1, 1.0]}],
        preprocess="standard",
        notes="Quantile loss is evaluated by the estimator itself; ML Lab does not fit a second model inside a metric function.",
    ),
}


def list_estimators() -> list[EstimatorSpec]:
    return [_REGISTRY[key] for key in sorted(_REGISTRY)]


def get_estimator_spec(estimator_id: str) -> EstimatorSpec:
    try:
        return _REGISTRY[estimator_id]
    except KeyError as exc:
        supported = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"unknown estimator {estimator_id!r}; supported: {supported}") from exc


def prefixed_param_grid(spec: EstimatorSpec, *, multi_output: bool = False) -> list[dict[str, list[Any]]]:
    prefix = "estimator__estimator__" if multi_output else "estimator__"
    return [{f"{prefix}{key}": value for key, value in grid.items()} for grid in spec.param_grid]
