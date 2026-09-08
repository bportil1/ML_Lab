from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .backend import require_torch
from .selection import SelectionMode


def _cpu_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in state_dict.items():
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            copied[key] = value.detach().cpu().clone()
        else:
            copied[key] = deepcopy(value)
    return copied


@dataclass(slots=True)
class BestModelCheckpoint:
    monitor: str
    mode: SelectionMode = "min"
    min_delta: float = 0.0
    path: str | Path | None = None
    best_value: float | None = None
    best_epoch: int | None = None
    _model_state: dict[str, Any] | None = field(default=None, repr=False)
    _optimizer_state: dict[str, Any] | None = field(default=None, repr=False)

    def _improved(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "min":
            return value < self.best_value - self.min_delta
        return value > self.best_value + self.min_delta

    def update(
        self,
        model: Any,
        value: float,
        epoch: int,
        *,
        optimizer: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        value = float(value)
        if not self._improved(value):
            return False
        self.best_value = value
        self.best_epoch = int(epoch)
        self._model_state = _cpu_state_dict(model.state_dict())
        if optimizer is not None:
            self._optimizer_state = deepcopy(optimizer.state_dict())

        if self.path is not None:
            torch = require_torch(purpose="neural checkpoint persistence")
            path = Path(self.path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": self._model_state,
                    "optimizer_state_dict": self._optimizer_state,
                    "monitor": self.monitor,
                    "mode": self.mode,
                    "best_value": self.best_value,
                    "best_epoch": self.best_epoch,
                    "metadata": metadata or {},
                },
                path,
            )
        return True

    def restore(self, model: Any, *, optimizer: Any | None = None) -> None:
        if self._model_state is None:
            return
        model.load_state_dict(self._model_state)
        if optimizer is not None and self._optimizer_state is not None:
            optimizer.load_state_dict(self._optimizer_state)


def load_checkpoint(
    path: str | Path,
    model: Any,
    *,
    optimizer: Any | None = None,
    map_location: str | Any = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint produced by :class:`BestModelCheckpoint`."""
    torch = require_torch(purpose="neural checkpoint loading")
    payload = torch.load(Path(path), map_location=map_location)
    model.load_state_dict(payload["model_state_dict"])
    optimizer_state = payload.get("optimizer_state_dict")
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    return payload
