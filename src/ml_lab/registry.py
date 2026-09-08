from __future__ import annotations

from ml_lab.classification.registry import list_estimators as list_classifiers
from ml_lab.clustering.registry import list_estimators as list_clusterers
from ml_lab.core.specs import EstimatorSpec
from ml_lab.regression.registry import list_estimators as list_regressors


def list_estimators(task: str | None = None) -> list[EstimatorSpec]:
    if task is None or task == "all":
        return sorted([*list_classifiers(), *list_clusterers(), *list_regressors()], key=lambda spec: (spec.task, spec.id))
    if task == "classification":
        return list_classifiers()
    if task == "clustering":
        return list_clusterers()
    if task == "regression":
        return list_regressors()
    raise ValueError("task must be classification, clustering, regression, all, or None")
