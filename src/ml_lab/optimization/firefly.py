"""Generic Firefly population optimizer ported from HSQA_DBN.

The movement policy intentionally preserves the current HSQA_DBN algorithm:
normalized search-space distance, sigmoid fitness attraction, log-normal
randomization, and independent per-dimension mutation. The source HSQA_DBN
repository is not modified.
"""
from __future__ import annotations

import csv
import json
import math
import os
import time
from queue import Empty
from typing import Any

import numpy as np

from .base import OptimizationAlgorithm
from .result import CandidateEvaluation, OptimizationResult
from .runtime import OptimizationRuntime


def _safe_json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "nan"
        return "inf" if value > 0 else "-inf"
    return value


def _worker(evaluator, position, iteration, candidate_id, queue, failure_fitness, runtime):
    try:
        runtime.configure_torch()
        evaluation = evaluator.evaluate(position, iteration=iteration, candidate_id=candidate_id)
        queue.put((candidate_id, evaluation))
    except Exception as exc:  # pragma: no cover - process hard-failure safety net
        queue.put((candidate_id, CandidateEvaluation(float(failure_fitness), {}, {}, status="error", error=repr(exc))))


class Firefly(OptimizationAlgorithm):
    name = "firefly"

    def __init__(
        self,
        *,
        search_space,
        evaluator,
        population_size: int,
        iterations: int,
        gamma: float,
        alpha: float,
        mutation_probability: float,
        mutation_scale_fraction: float,
        alpha_lognormal_sigma: float,
        fitness_difference_clip: float,
        failure_fitness: float,
        runtime: OptimizationRuntime | None = None,
        output_path: str | None = None,
        seed: int | None = None,
        scope: str = "model",
        metadata: dict[str, Any] | None = None,
    ):
        if population_size < 1:
            raise ValueError("Firefly population_size must be >= 1")
        if iterations < 0:
            raise ValueError("Firefly iterations must be >= 0")
        self.search_space = search_space
        self.evaluator = evaluator
        self.population_size = int(population_size)
        self.iterations = int(iterations)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.mutation_probability = float(mutation_probability)
        self.mutation_scale_fraction = float(mutation_scale_fraction)
        self.alpha_lognormal_sigma = float(alpha_lognormal_sigma)
        self.fitness_difference_clip = float(fitness_difference_clip)
        self.failure_fitness = float(failure_fitness)
        self.runtime = runtime or OptimizationRuntime()
        self.output_path = None if output_path is None else str(output_path)
        self.seed = seed
        self.scope = scope
        self.metadata = dict(metadata or {})
        self.bounds = list(search_space.bound_pairs)
        self.runtime.configure_torch()
        if seed is not None:
            self.runtime.seed_all(seed)
        # HSQA_DBN samples before seeding. ML_Lab intentionally corrects that
        # ordering so the optimizer's seed controls the initial population too.
        self.rng = np.random.default_rng(seed)
        self.positions = [search_space.sample(self.rng) for _ in range(self.population_size)]
        self.fitness = np.full(self.population_size, np.inf, dtype=float)
        self.evaluations: list[CandidateEvaluation | None] = [None] * self.population_size
        self.history_rows: list[dict[str, Any]] = []
        self.position_history: dict[int, list[list[float]]] = {}
        self.fitness_history: dict[int, list[float]] = {}
        self.objective_history: dict[int, list[dict[str, float]]] = {}
        self.evaluation_count = 0
        self._detailed_evaluations: list[dict[str, Any]] = []
        self._initial_evaluated = False

    @property
    def parallel(self) -> bool:
        return self.runtime.effective_candidate_parallelism == "process" and self.runtime.candidate_workers > 1

    def _error_evaluation(self, idx: int, exc: Exception) -> CandidateEvaluation:
        return CandidateEvaluation(
            self.failure_fitness,
            {},
            self.search_space.decode_mapping(self.positions[idx]),
            status="error",
            error=repr(exc),
        )

    def _evaluate_serial(self, iteration: int):
        for idx, position in enumerate(self.positions):
            try:
                evaluation = self.evaluator.evaluate(position, iteration=iteration, candidate_id=idx)
            except Exception as exc:
                evaluation = self._error_evaluation(idx, exc)
            self._record(idx, iteration, evaluation)
            self.runtime.clear_cache()

    def _evaluate_parallel(self, iteration: int):
        ctx = self.runtime.multiprocessing_context()
        queue = ctx.Queue()
        active: list[tuple[Any, int, int]] = []
        active_memory = 0
        received: set[int] = set()
        gpu_budget = self.runtime.gpu_budget_bytes()
        indices = sorted(range(self.population_size), key=lambda i: self.evaluator.estimate_memory(self.positions[i]), reverse=True)

        def collect():
            while True:
                try:
                    idx, evaluation = queue.get_nowait()
                except Empty:
                    break
                self._record(idx, iteration, evaluation)
                received.add(idx)

        for idx in indices:
            mem = int(self.evaluator.estimate_memory(self.positions[idx]))
            while active and (
                len(active) >= self.runtime.candidate_workers
                or (self.runtime.is_cuda and gpu_budget > 0 and active_memory + mem > gpu_budget)
            ):
                collect()
                proc, proc_mem, _ = active[0]
                if proc.is_alive():
                    time.sleep(0.05)
                    continue
                proc.join(timeout=1)
                active.pop(0)
                active_memory -= proc_mem
            proc = ctx.Process(
                target=_worker,
                args=(self.evaluator, self.positions[idx], iteration, idx, queue, self.failure_fitness, self.runtime),
            )
            proc.start()
            active.append((proc, mem, idx))
            active_memory += mem

        while active:
            collect()
            proc, proc_mem, _ = active[0]
            if proc.is_alive():
                time.sleep(0.05)
                continue
            proc.join(timeout=1)
            active.pop(0)
            active_memory -= proc_mem
        collect()

        for idx in range(self.population_size):
            if idx not in received:
                self._record(
                    idx,
                    iteration,
                    CandidateEvaluation(
                        self.failure_fitness,
                        {},
                        self.search_space.decode_mapping(self.positions[idx]),
                        status="missing_result_or_worker_crash",
                        error="candidate worker did not return a result",
                    ),
                )

    def evaluate_population(self, iteration: int):
        if self.parallel:
            self._evaluate_parallel(iteration)
        else:
            self._evaluate_serial(iteration)
        if iteration == -1:
            self._initial_evaluated = True
        self.position_history[iteration] = [np.asarray(p, dtype=float).tolist() for p in self.positions]
        self.fitness_history[iteration] = self.fitness.tolist()
        self.objective_history[iteration] = [dict(item.objective_terms) if item is not None else {} for item in self.evaluations]

    def _move_position(self, pos1, pos2, fitness1, fitness2):
        pos1 = np.asarray(pos1, dtype=float)
        pos2 = np.asarray(pos2, dtype=float)
        lower = np.asarray([pair[0] for pair in self.bounds], dtype=float)
        upper = np.asarray([pair[1] for pair in self.bounds], dtype=float)
        scale = upper - lower
        safe_scale = np.where(scale == 0, 1.0, scale)
        p1 = (pos1 - lower) / safe_scale
        p2 = (pos2 - lower) / safe_scale
        distance = np.linalg.norm(p1 - p2)
        delta = np.clip(float(fitness1) - float(fitness2), -self.fitness_difference_clip, self.fitness_difference_clip)
        beta = 1.0 / (1.0 + np.exp(-delta))
        attraction = beta * np.exp(-self.gamma * distance**2) * (pos2 - pos1)
        local_alpha = self.alpha * np.exp(self.alpha_lognormal_sigma * self.rng.normal(size=pos1.shape))
        random_term = local_alpha * (self.rng.random(pos1.shape) - 0.5) * safe_scale
        new_pos = pos1 + attraction + random_term
        for index in range(len(new_pos)):
            if self.rng.random() < self.mutation_probability and scale[index] > 0:
                new_pos[index] += self.rng.uniform(-scale[index] * self.mutation_scale_fraction, scale[index] * self.mutation_scale_fraction)
        return self.search_space.coerce(new_pos)

    def move_population(self):
        new_positions = [np.asarray(position, dtype=float).copy() for position in self.positions]
        for i in range(self.population_size):
            for j in range(self.population_size):
                if i == j:
                    continue
                new_positions[i] = self._move_position(new_positions[i], self.positions[j], self.fitness[i], self.fitness[j])
        self.positions = new_positions

    def _save(self, result: OptimizationResult):
        if self.output_path is None:
            return
        os.makedirs(self.output_path, exist_ok=True)
        search_path = os.path.join(self.output_path, "optimization_search_space.json")
        with open(search_path, "w", encoding="utf-8") as handle:
            json.dump(_safe_json(self.search_space.describe()), handle, indent=2)
        history_json = os.path.join(self.output_path, "firefly_history.json")
        with open(history_json, "w", encoding="utf-8") as handle:
            json.dump(_safe_json({
                "algorithm": self.name,
                "scope": self.scope,
                "movement_policy": {
                    "gamma": self.gamma,
                    "alpha": self.alpha,
                    "mutation_probability": self.mutation_probability,
                    "mutation_scale_fraction": self.mutation_scale_fraction,
                    "alpha_lognormal_sigma": self.alpha_lognormal_sigma,
                    "fitness_difference_clip": self.fitness_difference_clip,
                    "failure_fitness": self.failure_fitness,
                },
                "positions": self.position_history,
                "fitness": self.fitness_history,
                "objectives": self.objective_history,
            }), handle, indent=2)
        evaluation_log_path = os.path.join(self.output_path, "firefly_evaluations.json")
        with open(evaluation_log_path, "w", encoding="utf-8") as handle:
            json.dump(_safe_json(self._detailed_evaluations), handle, indent=2)
        csv_path = os.path.join(self.output_path, "optimization_history.csv")
        fields = sorted({key for row in self.history_rows for key in row})
        preferred = ["iteration", "candidate_id", "fitness", "status", "error"]
        fields = preferred + [field for field in fields if field not in preferred]
        with open(csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.history_rows)
        result.artifacts.update({
            "search_space": "optimization_search_space.json",
            "history": "optimization_history.csv",
            "firefly_history": "firefly_history.json",
            "evaluations": "firefly_evaluations.json",
            "result": "optimization_result.json",
        })
        with open(os.path.join(self.output_path, "optimization_result.json"), "w", encoding="utf-8") as handle:
            json.dump(_safe_json(result.to_dict()), handle, indent=2)

    def _record(self, candidate_id: int, iteration: int, evaluation: CandidateEvaluation):
        self.evaluations[candidate_id] = evaluation
        self.fitness[candidate_id] = float(evaluation.fitness)
        self.evaluation_count += 1
        self._detailed_evaluations.append({
            "iteration": int(iteration),
            "candidate_id": int(candidate_id),
            "position": np.asarray(self.positions[candidate_id], dtype=float).tolist(),
            "evaluation": evaluation.to_dict(),
        })
        row = {
            "iteration": int(iteration),
            "candidate_id": int(candidate_id),
            "fitness": float(evaluation.fitness),
            "status": evaluation.status,
            "error": evaluation.error or "",
            **{f"parameter.{k}": v for k, v in evaluation.parameters.items() if not isinstance(v, dict)},
            **{f"objective.{k}": v for k, v in evaluation.objective_terms.items()},
        }
        objective_params = evaluation.parameters.get("objective") if isinstance(evaluation.parameters, dict) else None
        if isinstance(objective_params, dict):
            for key, value in objective_params.get("parameters", {}).items():
                row[f"parameter.objective.{key}"] = value
        batching = evaluation.metadata.get("batching") if isinstance(evaluation.metadata, dict) else None
        if isinstance(batching, dict):
            for key, value in batching.items():
                row[f"batching.{key}"] = value
        self.history_rows.append(row)

    def optimize(self) -> OptimizationResult:
        if not self._initial_evaluated:
            self.evaluate_population(iteration=-1)
        best_index = int(np.argmin(self.fitness))
        best_position = np.asarray(self.positions[best_index], dtype=float).copy()
        best_evaluation = self.evaluations[best_index]

        for iteration in range(self.iterations):
            self.move_population()
            self.evaluate_population(iteration=iteration)
            finite = np.isfinite(self.fitness) & (self.fitness < self.failure_fitness)
            if not np.any(finite):
                self.positions = [self.search_space.sample(self.rng) for _ in range(self.population_size)]
                continue
            current_index = int(np.argmin(self.fitness))
            current_eval = self.evaluations[current_index]
            if current_eval is not None and (best_evaluation is None or current_eval.fitness < best_evaluation.fitness):
                best_position = np.asarray(self.positions[current_index], dtype=float).copy()
                best_evaluation = current_eval

        successful = [
            item for item in self._detailed_evaluations
            if item.get("evaluation", {}).get("status") == "ok"
            and float(item.get("evaluation", {}).get("fitness", self.failure_fitness)) < self.failure_fitness
        ]
        if best_evaluation is None:  # pragma: no cover
            raise RuntimeError("Firefly completed without any candidate evaluation")
        if not successful:
            result = OptimizationResult(
                algorithm=self.name,
                best_position=best_position.tolist(),
                best_parameters=dict(best_evaluation.parameters),
                best_fitness=float(best_evaluation.fitness),
                best_objective_terms=dict(best_evaluation.objective_terms),
                evaluations=self.evaluation_count,
                search_parameters=list(self.search_space.parameter_names),
                scope=self.scope,
                metadata={**dict(self.metadata), "optimization_failed": True, "runtime": self.runtime.metadata()},
            )
            self._save(result)
            first_error = next((item.get("evaluation", {}).get("error") for item in self._detailed_evaluations if item.get("evaluation", {}).get("error")), "unknown candidate evaluation error")
            raise RuntimeError(
                "Firefly optimization failed: every candidate evaluation returned an error. "
                f"First error: {first_error}."
                + (" See optimization_history.csv and firefly_evaluations.json for details." if self.output_path else "")
            )
        result = OptimizationResult(
            algorithm=self.name,
            best_position=best_position.tolist(),
            best_parameters=dict(best_evaluation.parameters),
            best_fitness=float(best_evaluation.fitness),
            best_objective_terms=dict(best_evaluation.objective_terms),
            evaluations=self.evaluation_count,
            search_parameters=list(self.search_space.parameter_names),
            scope=self.scope,
            metadata={**dict(self.metadata), "runtime": self.runtime.metadata()},
        )
        self._save(result)
        return result
