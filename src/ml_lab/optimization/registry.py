"""Optimization algorithm registry."""
from __future__ import annotations

from typing import Callable

from .firefly import Firefly


_ALGORITHMS: dict[str, Callable] = {"firefly": Firefly}


def normalize_optimizer_name(name: str) -> str:
    return str(name).strip().lower()


def register_optimizer(name: str, factory: Callable, *, replace: bool = False) -> None:
    key = normalize_optimizer_name(name)
    if key in _ALGORITHMS and not replace:
        raise ValueError(f"optimizer {key!r} is already registered")
    _ALGORITHMS[key] = factory


def create_optimizer(name: str, **kwargs):
    key = normalize_optimizer_name(name)
    try:
        factory = _ALGORITHMS[key]
    except KeyError as exc:
        raise ValueError(f"Unknown optimizer {name!r}; available={available_optimizers()}") from exc
    return factory(**kwargs)


def available_optimizers() -> tuple[str, ...]:
    return tuple(sorted(_ALGORITHMS))
