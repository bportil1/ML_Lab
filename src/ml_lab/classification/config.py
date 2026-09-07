from __future__ import annotations

from dataclasses import dataclass

from ml_lab.core.preprocessing import ScalingMode


@dataclass(slots=True)
class ClassificationSearchConfig:
    cv_folds: int = 5
    scoring: str = "accuracy"
    n_jobs: int = -1
    random_state: int = 42
    scaling: ScalingMode = "auto"
    refit: bool = True
    error_score: float = float("nan")

    def validate(self) -> None:
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be >= 2")
