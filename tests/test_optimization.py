from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from ml_lab import optimization


def test_search_space_supports_typed_and_fixed_parameters():
    space = optimization.SearchSpace(
        [
            optimization.ParameterSpec("lr", "continuous_log", 1e-5, 1e-2),
            optimization.ParameterSpec("width", "integer", 2, 8),
            optimization.ParameterSpec("activation", "categorical", choices=("relu", "gelu")),
        ],
        fixed_parameters={"epochs": 10},
    )
    position = space.coerce([3e-4, 5.7, 1.2])
    params = space.decode_mapping(position)
    assert params["epochs"] == 10
    assert params["width"] == 6
    assert params["activation"] == "gelu"
    assert 1e-5 <= params["lr"] <= 1e-2


def _sphere(params, **_):
    x = params["x"]
    y = params["y"]
    value = x * x + y * y
    return {"fitness": value, "objective_terms": {"sphere": value}}


def _run_seeded(tmp_path, seed):
    space = optimization.SearchSpace([
        optimization.ParameterSpec("x", "continuous", -2.0, 2.0),
        optimization.ParameterSpec("y", "continuous", -2.0, 2.0),
    ])
    evaluator = optimization.FunctionEvaluator(space, _sphere)
    return optimization.optimize(
        space,
        evaluator,
        config=optimization.FireflyConfig(population_size=4, iterations=2, seed=seed),
        runtime=optimization.OptimizationRuntime(candidate_parallelism="serial"),
        output_path=str(tmp_path),
    )


def test_firefly_is_reproducible_from_seed(tmp_path):
    first = _run_seeded(tmp_path / "first", 17)
    second = _run_seeded(tmp_path / "second", 17)
    assert first.best_position == second.best_position
    assert first.best_fitness == pytest.approx(second.best_fitness)
    assert first.best_parameters == second.best_parameters


def test_firefly_writes_hsqa_style_trace_artifacts(tmp_path):
    result = _run_seeded(tmp_path, 11)
    expected = {
        "optimization_search_space.json",
        "optimization_history.csv",
        "firefly_history.json",
        "firefly_evaluations.json",
        "optimization_result.json",
    }
    assert expected == {path.name for path in tmp_path.iterdir()}
    payload = json.loads((tmp_path / "optimization_result.json").read_text())
    assert payload["algorithm"] == "firefly"
    assert payload["evaluations"] == result.evaluations
    assert payload["search_parameters"] == ["x", "y"]


def test_weighted_fitness_supports_minimize_and_maximize():
    fitness = optimization.FitnessEvaluator([
        {"name": "loss", "weight": 2.0, "goal": "minimize"},
        {"name": "accuracy", "weight": 0.5, "goal": "maximize"},
    ])
    assert fitness.evaluate({"loss": 3.0, "accuracy": 0.8}) == pytest.approx(5.6)


def test_function_evaluator_can_compute_fitness_from_terms():
    space = optimization.SearchSpace([optimization.ParameterSpec("x", "continuous", -1.0, 1.0)])
    fitness = optimization.FitnessEvaluator([{"name": "abs_x", "weight": 1.0, "goal": "minimize"}])

    def objective(params, **_):
        return {"objective_terms": {"abs_x": abs(params["x"])}}

    evaluator = optimization.FunctionEvaluator(space, objective, fitness=fitness)
    evaluation = evaluator.evaluate([0.25])
    assert evaluation.fitness == pytest.approx(0.25)
    assert evaluation.parameters["x"] == pytest.approx(0.25)


def test_firefly_candidate_exceptions_become_diagnostic_failure(tmp_path):
    space = optimization.SearchSpace([optimization.ParameterSpec("x", "continuous", 0.0, 1.0)])

    def broken(_params, **_):
        raise RuntimeError("synthetic candidate failure")

    evaluator = optimization.FunctionEvaluator(space, broken)
    with pytest.raises(RuntimeError, match="every candidate evaluation returned an error"):
        optimization.optimize(
            space,
            evaluator,
            config=optimization.FireflyConfig(population_size=2, iterations=1, seed=3),
            runtime=optimization.OptimizationRuntime(candidate_parallelism="serial"),
            output_path=str(tmp_path),
        )
    payload = json.loads((tmp_path / "optimization_result.json").read_text())
    assert payload["metadata"]["optimization_failed"] is True
    assert "synthetic candidate failure" in (tmp_path / "optimization_history.csv").read_text()


def test_optimizer_registry_contains_firefly():
    assert "firefly" in optimization.available_optimizers()


def test_optimization_import_does_not_require_torch(monkeypatch):
    # The generic optimization surface is NumPy-only unless a runtime explicitly
    # chooses to integrate with an installed Torch/CUDA environment.
    assert "torch" not in optimization.space.__dict__
