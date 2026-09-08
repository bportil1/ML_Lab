from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class NeuralTrainingResult:
    model: Any
    history: list[dict[str, Any]]
    monitor: str
    best_metric: float | None
    best_epoch: int | None
    stopped_epoch: int
    device: str
    checkpoint_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "monitor": self.monitor,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "stopped_epoch": self.stopped_epoch,
            "device": self.device,
            "checkpoint_path": self.checkpoint_path,
            "history": self.history,
            "metadata": self.metadata,
        }


class NeuralTrainer(ABC):
    """Small common trainer contract for stable neural task implementations."""

    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> NeuralTrainingResult:
        raise NotImplementedError
