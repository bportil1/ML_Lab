from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any, Iterable
import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from ml_lab.core.preprocessing import make_preprocessor
from ml_lab.core.specs import EstimatorSpec

from .config import RegressionSearchConfig
from .data import RegressionDataset
from .evaluation import regression_metrics
from .registry import get_estimator_spec, prefixed_param_grid


@dataclass(slots=True)
class RegressionResult:
    task: str
    estimator_id: str
    estimator_name: str
    best_cv_score: float
    best_params: dict[str, Any]
    metrics: dict[str, float]
    search_seconds: float
    cv_folds_used: int
    preprocessing: str
    target_names: list[str]
    multi_output: bool
    fitted_model: BaseEstimator
    candidates: list[dict[str, Any]]
    y_true: np.ndarray
    y_pred: np.ndarray
    failed_candidates: int = 0

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record.pop("fitted_model")
        record.pop("y_true")
        record.pop("y_pred")
        return record


def _effective_cv_folds(n_rows: int, requested: int) -> int:
    folds = min(requested, n_rows)
    if folds < 2:
        raise ValueError("not enough training rows for cross-validation")
    return folds


def build_pipeline(
    spec: EstimatorSpec,
    config: RegressionSearchConfig,
    *,
    multi_output: bool = False,
) -> Pipeline:
    estimator: BaseEstimator = spec.factory(config.random_state)
    if multi_output:
        estimator = MultiOutputRegressor(estimator, n_jobs=config.n_jobs)
    return Pipeline([
        ("preprocess", make_preprocessor(spec, config.scaling)),
        ("estimator", estimator),
    ])


def _strip_estimator_prefix(key: str) -> str:
    key = key.removeprefix("estimator__")
    return key.removeprefix("estimator__")


def run_model_selection(
    estimator: str | EstimatorSpec,
    dataset: RegressionDataset,
    config: RegressionSearchConfig | None = None,
) -> RegressionResult:
    config = config or RegressionSearchConfig()
    config.validate()
    spec = get_estimator_spec(estimator) if isinstance(estimator, str) else estimator
    if spec.task != "regression":
        raise ValueError(f"estimator {spec.id!r} is registered for {spec.task}, not regression")

    folds = _effective_cv_folds(len(dataset.X_train), config.cv_folds)
    cv = KFold(n_splits=folds, shuffle=True, random_state=config.random_state)
    pipeline = build_pipeline(spec, config, multi_output=dataset.multi_output)
    search = GridSearchCV(
        pipeline,
        prefixed_param_grid(spec, multi_output=dataset.multi_output),
        scoring=config.scoring,
        cv=cv,
        n_jobs=config.n_jobs,
        refit=config.refit,
        error_score=config.error_score,
        return_train_score=False,
    )

    started = time.perf_counter()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        search.fit(dataset.X_train, dataset.y_train)
    elapsed = time.perf_counter() - started
    if not hasattr(search, "best_estimator_"):
        raise RuntimeError(f"no valid candidate was found for {spec.id}")

    scores = np.asarray(search.cv_results_["mean_test_score"], dtype=float)
    candidates: list[dict[str, Any]] = []
    for index, params in enumerate(search.cv_results_["params"]):
        candidates.append({
            "params": {_strip_estimator_prefix(key): value for key, value in params.items()},
            "mean_test_score": float(search.cv_results_["mean_test_score"][index]),
            "std_test_score": float(search.cv_results_["std_test_score"][index]),
            "rank_test_score": int(search.cv_results_["rank_test_score"][index]),
        })

    best_params = {_strip_estimator_prefix(key): value for key, value in search.best_params_.items()}
    metrics = regression_metrics(search.best_estimator_, dataset.X_test, dataset.y_test)
    y_pred = np.asarray(search.best_estimator_.predict(dataset.X_test))
    y_true = np.asarray(dataset.y_test)
    preprocessing = type(search.best_estimator_.named_steps["preprocess"]).__name__
    if search.best_estimator_.named_steps["preprocess"] == "passthrough":
        preprocessing = "passthrough"

    return RegressionResult(
        task="regression",
        estimator_id=spec.id,
        estimator_name=spec.name,
        best_cv_score=float(search.best_score_),
        best_params=best_params,
        metrics=metrics,
        search_seconds=float(elapsed),
        cv_folds_used=folds,
        preprocessing=preprocessing,
        target_names=list(dataset.target_names),
        multi_output=dataset.multi_output,
        fitted_model=search.best_estimator_,
        candidates=candidates,
        y_true=y_true,
        y_pred=y_pred,
        failed_candidates=int(np.isnan(scores).sum()),
    )


def run_model_suite(
    estimators: Iterable[str],
    dataset: RegressionDataset,
    config: RegressionSearchConfig | None = None,
) -> list[RegressionResult]:
    results = [run_model_selection(estimator, dataset, config) for estimator in estimators]
    return sorted(results, key=lambda item: item.best_cv_score, reverse=True)
