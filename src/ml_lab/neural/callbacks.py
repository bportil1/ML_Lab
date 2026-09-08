from __future__ import annotations

from typing import Any, Iterable, Protocol


class TrainingCallback(Protocol):
    def on_train_begin(self, context: dict[str, Any]) -> None: ...
    def on_epoch_end(self, epoch: int, metrics: dict[str, Any], context: dict[str, Any]) -> None: ...
    def on_train_end(self, context: dict[str, Any]) -> None: ...


class CallbackList:
    def __init__(self, callbacks: Iterable[Any] | None = None) -> None:
        self.callbacks = list(callbacks or [])

    def on_train_begin(self, context: dict[str, Any]) -> None:
        for callback in self.callbacks:
            hook = getattr(callback, "on_train_begin", None)
            if hook is not None:
                hook(context)

    def on_epoch_end(self, epoch: int, metrics: dict[str, Any], context: dict[str, Any]) -> None:
        for callback in self.callbacks:
            hook = getattr(callback, "on_epoch_end", None)
            if hook is not None:
                hook(epoch, metrics, context)

    def on_train_end(self, context: dict[str, Any]) -> None:
        for callback in self.callbacks:
            hook = getattr(callback, "on_train_end", None)
            if hook is not None:
                hook(context)
