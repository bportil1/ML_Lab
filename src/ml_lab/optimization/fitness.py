"""Named fitness-term composition for generic candidate optimization."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


@dataclass(frozen=True)
class FitnessTermSpec:
    name: str
    weight: float = 1.0
    goal: str = "minimize"

    @classmethod
    def from_value(cls, value):
        if isinstance(value, cls):
            return value
        raw = dict(value)
        return cls(name=str(raw["name"]), weight=float(raw.get("weight", 1.0)), goal=str(raw.get("goal", "minimize")))

    def __post_init__(self):
        if self.goal not in {"minimize", "maximize"}:
            raise ValueError(f"fitness term {self.name!r} goal must be minimize or maximize")
        if self.weight < 0:
            raise ValueError(f"fitness term {self.name!r} weight must be >= 0")


class FitnessEvaluator:
    """Reduce named candidate measurements to the scalar minimized by optimizers."""

    def __init__(self, terms: Sequence[FitnessTermSpec | Mapping]):
        self.terms = tuple(FitnessTermSpec.from_value(item) for item in terms)
        if not self.terms:
            raise ValueError("fitness requires at least one term")
        names = [item.name for item in self.terms]
        if len(names) != len(set(names)):
            raise ValueError("fitness terms cannot contain duplicate names")

    def evaluate(self, objective_terms: Mapping[str, float]) -> float:
        total = 0.0
        for spec in self.terms:
            if spec.name not in objective_terms:
                raise ValueError(
                    f"Candidate did not produce configured fitness term {spec.name!r}; available={sorted(objective_terms)}"
                )
            value = float(objective_terms[spec.name])
            if not math.isfinite(value):
                raise ValueError(f"fitness term {spec.name!r} is non-finite: {value}")
            direction = 1.0 if spec.goal == "minimize" else -1.0
            total += direction * spec.weight * value
        return float(total)

    def describe(self) -> list[dict]:
        return [{"name": item.name, "weight": item.weight, "goal": item.goal} for item in self.terms]
