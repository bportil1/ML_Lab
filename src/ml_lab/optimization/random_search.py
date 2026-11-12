"""Independent, seeded Random Search over the shared typed SearchSpace.

This is a serial baseline: it samples independent candidates rather than using
Firefly movement, and writes the same common CSV/result artifact contract.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from .base import OptimizationAlgorithm
from .firefly import OptimizationStopped, _safe_json
from .result import CandidateEvaluation, OptimizationResult
from .runtime import OptimizationRuntime


class RandomSearch(OptimizationAlgorithm):
    name = "random_search"

    def __init__(self, *, search_space, evaluator, evaluations: int = 10,
                 failure_fitness: float = 1e9, runtime: OptimizationRuntime | None = None,
                 output_path: str | None = None, seed: int | None = 42,
                 scope: str = "model", metadata: dict | None = None, should_stop=None):
        if type(evaluations) is not int or not 1 <= evaluations <= 100_000:
            raise ValueError("Random Search evaluations must be in [1,100000]")
        if not math.isfinite(failure_fitness) or failure_fitness <= 0:
            raise ValueError("failure_fitness must be finite and positive")
        self.search_space = search_space
        self.evaluator = evaluator
        self.evaluations = evaluations
        self.failure_fitness = float(failure_fitness)
        self.runtime = runtime or OptimizationRuntime()
        if self.runtime.effective_candidate_parallelism != "serial":
            raise ValueError("Random Search currently requires serial candidate evaluation")
        self.output_path = Path(output_path) if output_path is not None else None
        self.seed = seed
        self.scope = scope
        self.metadata = dict(metadata or {})
        self.should_stop = should_stop
        self.rng = np.random.default_rng(seed)
        self.records: list[dict] = []
        self.history_rows: list[dict] = []

    def _check_stop(self):
        if self.should_stop is not None and self.should_stop():
            raise OptimizationStopped("Optimizer stop requested")

    def _save(self, result: OptimizationResult):
        if self.output_path is None:
            return
        self.output_path.mkdir(parents=True, exist_ok=True)
        (self.output_path / "optimization_search_space.json").write_text(
            json.dumps(_safe_json(self.search_space.describe()), indent=2), encoding="utf-8")
        (self.output_path / "random_search_history.json").write_text(
            json.dumps(_safe_json({"algorithm": self.name, "scope": self.scope,
                                   "seed": self.seed, "evaluations_requested": self.evaluations,
                                   "trials": self.records}), indent=2), encoding="utf-8")
        (self.output_path / "random_search_evaluations.json").write_text(
            json.dumps(_safe_json(self.records), indent=2), encoding="utf-8")
        columns = ["iteration", "candidate_id", "fitness", "status", "error"]
        extra = sorted({key for row in self.history_rows for key in row} - set(columns))
        with (self.output_path / "optimization_history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns + extra)
            writer.writeheader()
            writer.writerows(self.history_rows)
        result.artifacts.update({"search_space": "optimization_search_space.json",
                                 "history": "optimization_history.csv",
                                 "random_search_history": "random_search_history.json",
                                 "evaluations": "random_search_evaluations.json",
                                 "result": "optimization_result.json"})
        (self.output_path / "optimization_result.json").write_text(
            json.dumps(_safe_json(result.to_dict()), indent=2), encoding="utf-8")

    def optimize(self) -> OptimizationResult:
        self.runtime.configure_torch()
        best_position = None
        best_evaluation = None
        for index in range(self.evaluations):
            self._check_stop()
            position = self.search_space.sample(self.rng)
            parameters = self.search_space.decode_mapping(position)
            try:
                evaluation = self.evaluator.evaluate(position, iteration=index, candidate_id=index)
                if not isinstance(evaluation, CandidateEvaluation):
                    raise TypeError("Candidate evaluator must return CandidateEvaluation")
                if not math.isfinite(evaluation.fitness):
                    raise ValueError("Candidate fitness must be finite")
            except Exception as exc:
                evaluation = CandidateEvaluation(self.failure_fitness, {}, parameters,
                                                 status="error", error=repr(exc))
            self.records.append({"iteration": index, "candidate_id": index,
                                 "position": position.tolist(), "evaluation": evaluation.to_dict()})
            self.history_rows.append({
                "iteration": index, "candidate_id": index, "fitness": float(evaluation.fitness),
                "status": evaluation.status, "error": evaluation.error or "",
                **{f"parameter.{key}": value for key, value in evaluation.parameters.items()
                   if not isinstance(value, dict)},
                **{f"objective.{key}": value for key, value in evaluation.objective_terms.items()},
            })
            if (evaluation.status == "ok" and evaluation.fitness < self.failure_fitness
                    and (best_evaluation is None or evaluation.fitness < best_evaluation.fitness)):
                best_position = position.copy()
                best_evaluation = evaluation
            self.runtime.clear_cache()
        failed = best_evaluation is None
        result = OptimizationResult(
            algorithm=self.name, best_position=best_position.tolist() if best_position is not None else [],
            best_parameters=dict(best_evaluation.parameters) if best_evaluation else {},
            best_fitness=float(best_evaluation.fitness) if best_evaluation else self.failure_fitness,
            best_objective_terms=dict(best_evaluation.objective_terms) if best_evaluation else {},
            evaluations=len(self.records), search_parameters=list(self.search_space.parameter_names),
            scope=self.scope, metadata={**self.metadata, "runtime": self.runtime.metadata(),
                                        **({"optimization_failed": True} if failed else {})},
        )
        self._check_stop()
        self._save(result)
        if failed:
            first_error = next((r["evaluation"].get("error") for r in self.records
                                if r["evaluation"].get("error")), "no successful candidate")
            raise RuntimeError("Random Search optimization failed: every candidate evaluation returned an error. "
                               f"First error: {first_error}.")
        return result
