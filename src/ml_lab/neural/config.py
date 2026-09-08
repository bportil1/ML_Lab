from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DeviceName = Literal["auto", "cpu", "cuda", "mps"]


@dataclass(frozen=True, slots=True)
class NeuralTrainingConfig:
    """Common controls shared by stable neural trainers.

    Task-specific trainers should subclass this configuration rather than duplicate
    runtime, optimization, validation, model-selection, and checkpoint controls.
    """

    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    validation_fraction: float = 0.1
    patience: int | None = 15
    min_delta: float = 1e-6
    random_state: int = 42
    device: DeviceName = "auto"
    deterministic: bool = True
    checkpoint_path: str | None = None
    restore_best: bool = True

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        if self.patience is not None and self.patience < 1:
            raise ValueError("patience must be positive or None")
        if self.min_delta < 0:
            raise ValueError("min_delta cannot be negative")
