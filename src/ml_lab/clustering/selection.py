from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

from ml_lab.core.preprocessing import make_preprocessor
from ml_lab.core.specs import EstimatorSpec
from .config import ClusteringSearchConfig
from .evaluation import external_metrics, internal_metrics
from .stability import RepeatStabilityReport, repeat_stability_report
from .registry import get_estimator_spec


@dataclass(slots=True)
class ClusteringCandidateResult:
    estimator_id: str
    estimator_name: str
    params: dict[str, Any]
    metrics: dict[str, float | int | None]
    stability: float | None
    failed_repeats: int
    elapsed_seconds: float
    warning: str | None = None
    stability_std: float | None = None

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ClusteringResult:
    task: str
    estimator_id: str
    estimator_name: str
    best_params: dict[str, Any]
    selection_metric: str
    selection_score: float | None
    metrics: dict[str, float | int | None]
    external_metrics: dict[str, float] | None
    stability: float | None
    preprocessing: str
    labels: np.ndarray
    fitted_model: BaseEstimator
    candidates: list[ClusteringCandidateResult]
    stability_analysis: RepeatStabilityReport | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "estimator_id": self.estimator_id,
            "estimator_name": self.estimator_name,
            "best_params": dict(self.best_params),
            "selection_metric": self.selection_metric,
            "selection_score": self.selection_score,
            "metrics": dict(self.metrics),
            "external_metrics": None if self.external_metrics is None else dict(self.external_metrics),
            "stability": self.stability,
            "preprocessing": self.preprocessing,
            "labels": self.labels.tolist(),
            "candidates": [candidate.to_record() for candidate in self.candidates],
            "stability_analysis": (
                None if self.stability_analysis is None else self.stability_analysis.summary_record()
            ),
        }


def _pipeline(spec: EstimatorSpec, config: ClusteringSearchConfig, seed: int) -> Pipeline:
    return Pipeline([
        ("preprocess", make_preprocessor(spec, config.scaling)),
        ("estimator", spec.factory(seed)),
    ])


def _fit_predict(pipeline: Pipeline, X: pd.DataFrame | np.ndarray, params: dict[str, Any]) -> np.ndarray:
    configured = clone(pipeline).set_params(**{f"estimator__{k}": v for k, v in params.items()})
    if not hasattr(configured, "fit_predict"):
        raise TypeError(f"{configured.named_steps['estimator'].__class__.__name__} does not support fit_predict")
    return np.asarray(configured.fit_predict(X))


def _raw_selection_score(candidate: ClusteringCandidateResult, metric: str) -> float | None:
    if metric == "stability":
        return None if candidate.stability is None else float(candidate.stability)
    value = candidate.metrics.get(metric)
    return None if value is None else float(value)


def _metric_sort_value(candidate: ClusteringCandidateResult, metric: str) -> float:
    if metric == "stability":
        return float("-inf") if candidate.stability is None else float(candidate.stability)
    value = candidate.metrics.get(metric)
    if value is None:
        return float("-inf")
    value = float(value)
    if np.isnan(value):
        return float("-inf")
    return -value if metric == "davies_bouldin" else value


def run_cluster_selection(
    estimator: str | EstimatorSpec,
    X: pd.DataFrame | np.ndarray,
    config: ClusteringSearchConfig | None = None,
    *,
    y_true: pd.Series | np.ndarray | None = None,
) -> ClusteringResult:
    config = config or ClusteringSearchConfig()
    config.validate()
    spec = get_estimator_spec(estimator) if isinstance(estimator, str) else estimator
    if spec.task != "clustering":
        raise ValueError(f"estimator {spec.id!r} is registered for {spec.task}, not clustering")

    candidates: list[ClusteringCandidateResult] = []
    seed_sequence = [config.random_state + i for i in range(config.repeats)]
    for params in ParameterGrid(spec.param_grid):
        started = time.perf_counter()
        label_runs: list[np.ndarray] = []
        failed = 0
        warning: str | None = None
        repeat_metrics: list[dict[str, float | int | None]] = []
        for seed in seed_sequence:
            try:
                pipeline = _pipeline(spec, config, seed)
                labels = _fit_predict(pipeline, X, params)
                label_runs.append(labels)
                transformed = pipeline.named_steps["preprocess"].fit_transform(X) if pipeline.named_steps["preprocess"] != "passthrough" else np.asarray(X)
                repeat_metrics.append(internal_metrics(transformed, labels))
            except Exception as exc:  # candidate failures are isolated by design
                failed += 1
                warning = f"{type(exc).__name__}: {exc}"

        elapsed = time.perf_counter() - started
        if not label_runs:
            candidates.append(
                ClusteringCandidateResult(spec.id, spec.name, dict(params), {}, None, failed, elapsed, warning)
            )
            continue

        metric_names = set().union(*(m.keys() for m in repeat_metrics))
        averaged: dict[str, float | int | None] = {}
        for name in metric_names:
            values = [m[name] for m in repeat_metrics if m.get(name) is not None]
            averaged[name] = float(np.mean(values)) if values else None
        repeat_report = repeat_stability_report(
            label_runs,
            repeats_requested=config.repeats,
            failed_repeats=failed,
            ignore_noise=config.stability_ignore_noise,
        )
        candidates.append(
            ClusteringCandidateResult(
                estimator_id=spec.id,
                estimator_name=spec.name,
                params=dict(params),
                metrics=averaged,
                stability=repeat_report.adjusted_rand_mean,
                failed_repeats=failed,
                elapsed_seconds=float(elapsed),
                warning=warning,
                stability_std=repeat_report.adjusted_rand_std,
            )
        )

    valid = [c for c in candidates if c.metrics]
    if not valid:
        first = candidates[0].warning if candidates else "no candidates"
        raise RuntimeError(f"all clustering candidates failed for {spec.id}; first error: {first}")

    best = max(valid, key=lambda c: _metric_sort_value(c, config.selection_metric))
    final_pipeline = _pipeline(spec, config, config.random_state)
    final_pipeline.set_params(**{f"estimator__{k}": v for k, v in best.params.items()})
    labels = np.asarray(final_pipeline.fit_predict(X))
    transformed = final_pipeline.named_steps["preprocess"].transform(X) if final_pipeline.named_steps["preprocess"] != "passthrough" else np.asarray(X)
    metrics = internal_metrics(transformed, labels)
    ext = external_metrics(y_true, labels) if y_true is not None else None
    preprocessing = type(final_pipeline.named_steps["preprocess"]).__name__
    if final_pipeline.named_steps["preprocess"] == "passthrough":
        preprocessing = "passthrough"

    stability_analysis: RepeatStabilityReport | None = None
    if config.stability_analysis:
        selected_runs: list[np.ndarray] = []
        selected_seeds: list[int] = []
        selected_failed = 0
        for seed in seed_sequence:
            try:
                pipeline = _pipeline(spec, config, seed)
                selected_runs.append(_fit_predict(pipeline, X, best.params))
                selected_seeds.append(seed)
            except Exception:
                selected_failed += 1
        stability_analysis = repeat_stability_report(
            selected_runs,
            repeats_requested=config.repeats,
            seeds=selected_seeds,
            failed_repeats=selected_failed,
            ignore_noise=config.stability_ignore_noise,
        )

    return ClusteringResult(
        task="clustering",
        estimator_id=spec.id,
        estimator_name=spec.name,
        best_params=dict(best.params),
        selection_metric=config.selection_metric,
        selection_score=_raw_selection_score(best, config.selection_metric),
        metrics=metrics,
        external_metrics=ext,
        stability=best.stability,
        preprocessing=preprocessing,
        labels=labels,
        fitted_model=final_pipeline,
        candidates=candidates,
        stability_analysis=stability_analysis,
    )


def run_cluster_suite(
    estimators: Iterable[str],
    X: pd.DataFrame | np.ndarray,
    config: ClusteringSearchConfig | None = None,
    *,
    y_true: pd.Series | np.ndarray | None = None,
) -> list[ClusteringResult]:
    config = config or ClusteringSearchConfig()
    results = [run_cluster_selection(estimator, X, config, y_true=y_true) for estimator in estimators]

    def score(result: ClusteringResult) -> float:
        if result.selection_metric == "davies_bouldin":
            value = result.metrics.get("davies_bouldin")
            return float("-inf") if value is None else -float(value)
        if result.selection_metric == "stability":
            return float("-inf") if result.stability is None else float(result.stability)
        value = result.metrics.get(result.selection_metric)
        return float("-inf") if value is None else float(value)

    return sorted(results, key=score, reverse=True)
