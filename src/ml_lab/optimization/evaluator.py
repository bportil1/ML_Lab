"""Candidate evaluator contracts and function adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Real
from typing import Any, Callable, Mapping

from .fitness import FitnessEvaluator
from .result import CandidateEvaluation


class CandidateEvaluator(ABC):
    @abstractmethod
    def evaluate(self, position, *, iteration: int | None = None, candidate_id: int | None = None):
        raise NotImplementedError

    def estimate_memory(self, position) -> int:
        return 0


class FunctionEvaluator(CandidateEvaluator):
    """Adapt a Python function over decoded parameters to CandidateEvaluator.

    The callable may return a scalar fitness, a CandidateEvaluation, or a mapping.
    A mapping may contain ``fitness`` and/or ``objective_terms`` plus optional
    ``metadata`` and ``training_history``. When only objective terms are returned,
    a FitnessEvaluator must be supplied.
    """

    def __init__(
        self,
        search_space,
        function: Callable[..., Any],
        *,
        fitness: FitnessEvaluator | None = None,
        memory_estimator: Callable[[Mapping[str, Any]], int] | None = None,
    ):
        self.search_space = search_space
        self.function = function
        self.fitness_evaluator = fitness
        self.memory_estimator = memory_estimator

    def estimate_memory(self, position) -> int:
        if self.memory_estimator is None:
            return 0
        return int(self.memory_estimator(self.search_space.decode_mapping(position)))

    def evaluate(self, position, *, iteration=None, candidate_id=None):
        parameters = self.search_space.decode_mapping(position)
        raw = self.function(parameters, iteration=iteration, candidate_id=candidate_id)
        if isinstance(raw, CandidateEvaluation):
            if not raw.parameters:
                raw.parameters = dict(parameters)
            return raw
        if isinstance(raw, Real):
            value = float(raw)
            return CandidateEvaluation(value, {"objective": value}, parameters)
        if not isinstance(raw, Mapping):
            raise TypeError("optimization function must return a scalar, mapping, or CandidateEvaluation")
        payload = dict(raw)
        objective_terms = {str(k): float(v) for k, v in dict(payload.get("objective_terms", {})).items()}
        if "fitness" in payload:
            fitness_value = float(payload["fitness"])
        elif self.fitness_evaluator is not None:
            fitness_value = self.fitness_evaluator.evaluate(objective_terms)
        else:
            raise ValueError("mapping result must contain fitness unless a FitnessEvaluator is configured")
        return CandidateEvaluation(
            fitness=fitness_value,
            objective_terms=objective_terms,
            parameters=dict(payload.get("parameters", parameters)),
            training_history=payload.get("training_history"),
            metadata=dict(payload.get("metadata", {})),
            status=str(payload.get("status", "ok")),
            error=payload.get("error"),
        )
