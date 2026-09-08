from __future__ import annotations

from typing import Any

from .config import FireflyConfig
from .registry import create_optimizer
from .runtime import OptimizationRuntime


def optimize(
    search_space,
    evaluator,
    *,
    algorithm: str = "firefly",
    config: FireflyConfig | dict[str, Any] | None = None,
    runtime: OptimizationRuntime | dict[str, Any] | None = None,
    output_path: str | None = None,
    metadata: dict[str, Any] | None = None,
):
    if config is None:
        config = FireflyConfig()
    elif isinstance(config, dict):
        config = FireflyConfig(**config)
    if runtime is None:
        runtime = OptimizationRuntime()
    elif isinstance(runtime, dict):
        runtime = OptimizationRuntime(**runtime)
    return create_optimizer(
        algorithm,
        search_space=search_space,
        evaluator=evaluator,
        population_size=config.population_size,
        iterations=config.iterations,
        gamma=config.gamma,
        alpha=config.alpha,
        mutation_probability=config.mutation_probability,
        mutation_scale_fraction=config.mutation_scale_fraction,
        alpha_lognormal_sigma=config.alpha_lognormal_sigma,
        fitness_difference_clip=config.fitness_difference_clip,
        failure_fitness=config.failure_fitness,
        runtime=runtime,
        output_path=output_path,
        seed=config.seed,
        scope=config.scope,
        metadata=metadata,
    ).optimize()
