from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .config import FireflyConfig, RandomSearchConfig
from .registry import create_optimizer, normalize_optimizer_name
from .runtime import OptimizationRuntime


def optimize(
    search_space, evaluator, *, algorithm: str = "firefly",
    config: FireflyConfig | RandomSearchConfig | dict[str, Any] | None = None,
    runtime: OptimizationRuntime | dict[str, Any] | None = None,
    output_path: str | None = None, metadata: dict[str, Any] | None = None,
    should_stop=None,
):
    """Dispatch a typed optimizer without leaking algorithm-specific settings."""
    name = normalize_optimizer_name(algorithm)
    default_type = {"firefly": FireflyConfig, "random_search": RandomSearchConfig}.get(name)
    if config is None:
        if default_type is None:
            raise ValueError(f"Optimizer {name!r} requires explicit configuration")
        config = default_type()
    elif isinstance(config, dict):
        if default_type is None:
            settings = dict(config)
        else:
            config = default_type(**config)
    if not isinstance(config, dict):
        if default_type is not None and not isinstance(config, default_type):
            raise TypeError(f"Optimizer {name!r} requires {default_type.__name__}")
        settings = asdict(config)
    if runtime is None:
        runtime = OptimizationRuntime()
    elif isinstance(runtime, dict):
        runtime = OptimizationRuntime(**runtime)
    return create_optimizer(
        name, search_space=search_space, evaluator=evaluator, **settings, runtime=runtime,
        output_path=output_path, metadata=metadata,
        **({"should_stop": should_stop} if should_stop is not None else {}),
    ).optimize()
