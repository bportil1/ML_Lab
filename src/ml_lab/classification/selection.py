from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any, Iterable
import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from ml_lab.core.preprocessing import make_preprocessor

from .config import ClassificationSearchConfig
from .data import ClassificationDataset
from .evaluation import classification_metrics
from .registry import get_estimator_spec, prefixed_param_grid
from ml_lab.core.specs import EstimatorSpec


@dataclass(slots=True)
class ClassificationResult:
    task: str
    estimator_id: str
    estimator_name: str
    best_cv_score: float
    best_params: dict[str, Any]
    metrics: dict[str, float | None]
    search_seconds: float
    cv_folds_used: int
    preprocessing: str
    fitted_model: BaseEstimator
    candidates: list[dict[str, Any]]
    failed_candidates: int = 0

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record.pop("fitted_model")
        return record


def _effective_cv_folds(y_train: Any, requested: int) -> int:
    _, counts = np.unique(np.asarray(y_train), return_counts=True)
    folds = min(requested, int(counts.min()))
    if folds < 2:
        raise ValueError("not enough samples in the smallest class for stratified cross-validation")
    return folds


def build_pipeline(spec: EstimatorSpec, config: ClassificationSearchConfig) -> Pipeline:
    return Pipeline([
        ("preprocess", make_preprocessor(spec, config.scaling)),
        ("estimator", spec.factory(config.random_state)),
    ])


def run_model_selection(
    estimator: str | EstimatorSpec,
    dataset: ClassificationDataset,
    config: ClassificationSearchConfig | None = None,
) -> ClassificationResult:
    config = config or ClassificationSearchConfig()
    config.validate()
    spec = get_estimator_spec(estimator) if isinstance(estimator, str) else estimator
    if spec.task != "classification":
        raise ValueError(f"estimator {spec.id!r} is registered for {spec.task}, not classification")

    folds = _effective_cv_folds(dataset.y_train, config.cv_folds)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.random_state)
    pipeline = build_pipeline(spec, config)
    search = GridSearchCV(
        pipeline,
        prefixed_param_grid(spec),
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
            "params": {key.removeprefix("estimator__"): value for key, value in params.items()},
            "mean_test_score": float(search.cv_results_["mean_test_score"][index]),
            "std_test_score": float(search.cv_results_["std_test_score"][index]),
            "rank_test_score": int(search.cv_results_["rank_test_score"][index]),
        })
    best_params = {
        key.removeprefix("estimator__"): value
        for key, value in search.best_params_.items()
    }
    metrics = classification_metrics(search.best_estimator_, dataset.X_test, dataset.y_test)
    preprocessing = type(search.best_estimator_.named_steps["preprocess"]).__name__
    if search.best_estimator_.named_steps["preprocess"] == "passthrough":
        preprocessing = "passthrough"

    return ClassificationResult(
        task="classification",
        estimator_id=spec.id,
        estimator_name=spec.name,
        best_cv_score=float(search.best_score_),
        best_params=best_params,
        metrics=metrics,
        search_seconds=float(elapsed),
        cv_folds_used=folds,
        preprocessing=preprocessing,
        fitted_model=search.best_estimator_,
        candidates=candidates,
        failed_candidates=int(np.isnan(scores).sum()),
    )


def run_model_suite(
    estimators: Iterable[str],
    dataset: ClassificationDataset,
    config: ClassificationSearchConfig | None = None,
) -> list[ClassificationResult]:
    results = [run_model_selection(estimator, dataset, config) for estimator in estimators]
    return sorted(results, key=lambda item: item.best_cv_score, reverse=True)
