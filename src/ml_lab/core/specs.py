from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from sklearn.base import BaseEstimator

TaskType = Literal["classification", "clustering"]
PreprocessHint = Literal["none", "standard", "minmax", "robust"]


@dataclass(frozen=True, slots=True)
class EstimatorSpec:
    """Declarative estimator registration shared by ML Lab task families."""

    id: str
    name: str
    task: TaskType
    factory: Callable[[int], BaseEstimator]
    param_grid: list[dict[str, list[Any]]]
    preprocess: PreprocessHint = "none"
    notes: str = ""
