from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .config import BestModelSelectionConfig


@dataclass(frozen=True)
class BestModelUpdate:
    metric: float
    best_metric: float | None
    best_epoch: int | None
    new_best: bool
    significant_improvement: bool
    epochs_since_improvement: int
    should_stop: bool


class BestModelTracker:
    def __init__(self, config: BestModelSelectionConfig):
        self.config = config
        self.best_metric = math.inf
        self.best_epoch: int | None = None
        self.best_state: dict[str, Any] | None = None
        self.epochs_since_improvement = 0
        self.restored = False

    @staticmethod
    def _snapshot(model) -> dict[str, Any]:
        return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    def update(self, model, *, metric: float, epoch: int) -> BestModelUpdate:
        metric = float(metric)
        if not self.config.enabled:
            return BestModelUpdate(metric, None, None, False, False, 0, False)

        if not math.isfinite(metric):
            if epoch >= self.config.warmup_epochs:
                self.epochs_since_improvement += 1
            return BestModelUpdate(
                metric=metric,
                best_metric=None if self.best_epoch is None else self.best_metric,
                best_epoch=self.best_epoch,
                new_best=False,
                significant_improvement=False,
                epochs_since_improvement=self.epochs_since_improvement,
                should_stop=(
                    epoch >= self.config.warmup_epochs
                    and self.epochs_since_improvement >= self.config.patience
                ),
            )

        previous = self.best_metric
        new_best = metric < previous
        significant = False
        if new_best:
            if math.isinf(previous):
                significant = True
            else:
                relative = (previous - metric) / max(abs(previous), 1e-12)
                significant = relative >= self.config.min_relative_improvement
            self.best_metric = metric
            self.best_epoch = int(epoch)
            self.best_state = self._snapshot(model)

        if epoch < self.config.warmup_epochs:
            self.epochs_since_improvement = 0
        elif significant:
            self.epochs_since_improvement = 0
        else:
            self.epochs_since_improvement += 1

        return BestModelUpdate(
            metric=metric,
            best_metric=self.best_metric,
            best_epoch=self.best_epoch,
            new_best=new_best,
            significant_improvement=significant,
            epochs_since_improvement=self.epochs_since_improvement,
            should_stop=(
                epoch >= self.config.warmup_epochs
                and self.epochs_since_improvement >= self.config.patience
            ),
        )

    def restore(self, model) -> bool:
        if not self.config.enabled or not self.config.restore_best or self.best_state is None:
            return False
        model.load_state_dict(self.best_state, strict=True)
        self.restored = True
        return True

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.config.enabled),
            "metric": self.config.metric,
            "best_metric": None if self.best_epoch is None else float(self.best_metric),
            "best_epoch": self.best_epoch,
            "epochs_since_improvement": int(self.epochs_since_improvement),
            "restore_best": bool(self.config.restore_best),
            "restored_best": bool(self.restored),
            "patience": int(self.config.patience),
            "min_relative_improvement": float(self.config.min_relative_improvement),
            "warmup_epochs": int(self.config.warmup_epochs),
        }


def selection_metric_value(name: str, *, energy_gap: float, reconstruction_error: float, cd_objective: float) -> float:
    if name == "energy_gap_abs":
        return abs(float(energy_gap))
    if name == "reconstruction_error":
        return float(reconstruction_error)
    if name == "cd_objective":
        return float(cd_objective)
    raise ValueError(f"unsupported model-selection metric: {name!r}")
