from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FireflyConfig:
    population_size: int = 3
    iterations: int = 3
    gamma: float = 0.1
    alpha: float = 0.02
    mutation_probability: float = 0.10
    mutation_scale_fraction: float = 0.05
    alpha_lognormal_sigma: float = 0.10
    fitness_difference_clip: float = 50.0
    failure_fitness: float = 1e9
    seed: int | None = 42
    scope: str = "model"

    def __post_init__(self):
        if self.population_size < 1:
            raise ValueError("population_size must be >= 1")
        if self.iterations < 0:
            raise ValueError("iterations must be >= 0")
        if self.gamma < 0:
            raise ValueError("gamma must be >= 0")
        if self.alpha < 0:
            raise ValueError("alpha must be >= 0")
        if not 0 <= self.mutation_probability <= 1:
            raise ValueError("mutation_probability must be in [0,1]")
        if self.mutation_scale_fraction < 0:
            raise ValueError("mutation_scale_fraction must be >= 0")
        if self.alpha_lognormal_sigma < 0:
            raise ValueError("alpha_lognormal_sigma must be >= 0")
        if self.fitness_difference_clip <= 0:
            raise ValueError("fitness_difference_clip must be > 0")
