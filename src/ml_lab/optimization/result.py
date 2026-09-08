"""Common optimization result and candidate-evaluation records."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CandidateEvaluation:
    fitness: float
    objective_terms: dict[str, float]
    parameters: dict[str, Any]
    training_history: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fitness": float(self.fitness),
            "objective_terms": dict(self.objective_terms),
            "parameters": dict(self.parameters),
            "training_history": self.training_history,
            "metadata": dict(self.metadata),
            "status": self.status,
            "error": self.error,
        }


@dataclass
class OptimizationResult:
    algorithm: str
    best_position: list[float]
    best_parameters: dict[str, Any]
    best_fitness: float
    best_objective_terms: dict[str, float]
    evaluations: int
    search_parameters: list[str]
    scope: str = "model"
    metadata: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "scope": self.scope,
            "search_parameters": list(self.search_parameters),
            "best_position": list(self.best_position),
            "best_parameters": dict(self.best_parameters),
            "best_fitness": float(self.best_fitness),
            "best_objective_terms": dict(self.best_objective_terms),
            "evaluations": int(self.evaluations),
            "metadata": dict(self.metadata),
            "artifacts": dict(self.artifacts),
        }
