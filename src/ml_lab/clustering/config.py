from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ml_lab.core.preprocessing import ScalingMode

SelectionMetric = Literal["silhouette", "calinski_harabasz", "davies_bouldin", "stability"]


@dataclass(slots=True)
class ClusteringSearchConfig:
    selection_metric: SelectionMetric = "silhouette"
    repeats: int = 3
    random_state: int = 42
    scaling: ScalingMode = "auto"
    stability_analysis: bool = True
    stability_ignore_noise: bool = True

    def validate(self) -> None:
        if self.repeats < 1:
            raise ValueError("repeats must be >= 1")
