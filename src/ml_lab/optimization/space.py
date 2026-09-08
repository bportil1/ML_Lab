"""Typed numeric search-space primitives shared by optimization algorithms."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


_PARAMETER_KINDS = {"continuous", "continuous_log", "integer", "categorical"}


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    kind: str
    lower: float | None = None
    upper: float | None = None
    choices: tuple[Any, ...] = ()

    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("parameter name must not be empty")
        if self.kind not in _PARAMETER_KINDS:
            raise ValueError(f"Unsupported parameter kind {self.kind!r}")
        if self.kind == "categorical":
            if not self.choices:
                raise ValueError(f"categorical parameter {self.name!r} requires choices")
        else:
            if self.lower is None or self.upper is None:
                raise ValueError(f"parameter {self.name!r} requires lower/upper bounds")
            if self.lower > self.upper:
                raise ValueError(f"parameter {self.name!r} bounds must be ordered")
            if self.kind == "continuous_log" and self.lower <= 0:
                raise ValueError(f"log parameter {self.name!r} requires lower > 0")

    @property
    def numeric_bounds(self) -> tuple[float, float]:
        if self.kind == "categorical":
            return (0.0, float(len(self.choices) - 1))
        return (float(self.lower), float(self.upper))

    def sample(self, rng: np.random.Generator) -> float:
        low, high = self.numeric_bounds
        if low == high:
            return low
        if self.kind == "continuous":
            return float(rng.uniform(low, high))
        if self.kind == "continuous_log":
            return float(10 ** rng.uniform(np.log10(low), np.log10(high)))
        if self.kind in {"integer", "categorical"}:
            return float(rng.integers(int(np.ceil(low)), int(np.floor(high)) + 1))
        raise RuntimeError(self.kind)

    def coerce(self, value: float) -> float:
        low, high = self.numeric_bounds
        value = float(np.clip(float(value), low, high))
        if self.kind in {"integer", "categorical"}:
            value = float(int(round(value)))
        return value

    def decode(self, value: float) -> Any:
        value = self.coerce(value)
        if self.kind == "categorical":
            return self.choices[int(value)]
        if self.kind == "integer":
            return int(value)
        return float(value)


class SearchSpace:
    """A generic numeric position space with typed decoded parameters."""

    def __init__(self, specs: Sequence[ParameterSpec], *, fixed_parameters: Mapping[str, Any] | None = None):
        self.specs = tuple(specs)
        names = [spec.name for spec in self.specs]
        if len(names) != len(set(names)):
            raise ValueError("search parameter names cannot contain duplicates")
        fixed = dict(fixed_parameters or {})
        overlap = sorted(set(names) & set(fixed))
        if overlap:
            raise ValueError(f"fixed parameters overlap searched parameters: {overlap}")
        self.fixed_parameters = fixed

    @property
    def dimension(self) -> int:
        return len(self.specs)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    @property
    def bound_pairs(self) -> list[tuple[float, float]]:
        return [spec.numeric_bounds for spec in self.specs]

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return np.asarray([spec.sample(rng) for spec in self.specs], dtype=float)

    def coerce(self, position: Sequence[Any]) -> np.ndarray:
        if len(position) != self.dimension:
            raise ValueError(f"Position has {len(position)} values but space has dimension {self.dimension}")
        return np.asarray([spec.coerce(value) for spec, value in zip(self.specs, position)], dtype=float)

    def decode_mapping(self, position: Sequence[Any]) -> dict[str, Any]:
        position = self.coerce(position)
        values = dict(self.fixed_parameters)
        values.update({spec.name: spec.decode(value) for spec, value in zip(self.specs, position)})
        return values

    def describe(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "search_parameters": list(self.parameter_names),
            "parameters": [
                {
                    "name": spec.name,
                    "kind": spec.kind,
                    "bounds": list(spec.numeric_bounds),
                    **({"choices": list(spec.choices)} if spec.kind == "categorical" else {}),
                }
                for spec in self.specs
            ],
            "fixed_parameters": dict(self.fixed_parameters),
        }
