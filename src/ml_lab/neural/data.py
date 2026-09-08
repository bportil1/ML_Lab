from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ValidationSplit:
    train_indices: np.ndarray
    validation_indices: np.ndarray

    @property
    def train_count(self) -> int:
        return int(len(self.train_indices))

    @property
    def validation_count(self) -> int:
        return int(len(self.validation_indices))


def split_validation_indices(
    sample_count: int,
    validation_fraction: float,
    *,
    random_state: int,
) -> ValidationSplit:
    if sample_count < 2:
        raise ValueError("at least two samples are required")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")

    validation_count = 0
    if validation_fraction > 0 and sample_count >= 3:
        validation_count = max(1, int(round(sample_count * validation_fraction)))
        validation_count = min(validation_count, sample_count - 1)

    indices = np.random.default_rng(random_state).permutation(sample_count)
    return ValidationSplit(
        train_indices=indices[validation_count:],
        validation_indices=indices[:validation_count],
    )
