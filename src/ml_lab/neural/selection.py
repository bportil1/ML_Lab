from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SelectionMode = Literal["min", "max"]


@dataclass(slots=True)
class EarlyStopping:
    patience: int | None = 15
    min_delta: float = 1e-6
    mode: SelectionMode = "min"
    best_value: float | None = None
    best_epoch: int | None = None
    stale_epochs: int = 0

    def _improved(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "min":
            return value < self.best_value - self.min_delta
        return value > self.best_value + self.min_delta

    def update(self, value: float, epoch: int) -> tuple[bool, bool]:
        improved = self._improved(float(value))
        if improved:
            self.best_value = float(value)
            self.best_epoch = int(epoch)
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        should_stop = self.patience is not None and self.stale_epochs >= self.patience
        return improved, should_stop
