from __future__ import annotations

from typing import Any

import numpy as np

from .graph import generate_planted_clique_graph, greedy_clique, is_clique, validate_adjacency


def _resolve_graph(graph: Any, generated: dict[str, Any] | None) -> tuple[np.ndarray, list[int] | None]:
    if graph is not None and generated is not None:
        raise ValueError("provide either graph or generated settings, not both")
    if graph is not None:
        return validate_adjacency(graph), None
    settings = generated or {}
    matrix, planted = generate_planted_clique_graph(**settings)
    return matrix, planted


def run(
    *,
    mode: str = "describe",
    graph: Any = None,
    generated: dict[str, Any] | None = None,
    initial_clique: list[int] | None = None,
    dqn_config: dict[str, Any] | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    if mode == "describe":
        return {
            "experiment": "max_clique_rl",
            "status": "experimental",
            "source_concepts": [
                "maximum-clique graph environment",
                "clique-valid action masking",
                "prototype DQN search",
            ],
            "available_modes": ["describe", "greedy", "dqn"],
            "warnings": [
                "This is graph-specific research code, not a stable general reinforcement-learning API.",
                "Legacy partitioned-RBM and optimizer code were intentionally excluded from this import.",
            ],
        }

    matrix, planted = _resolve_graph(graph, generated)
    if mode == "greedy":
        clique = greedy_clique(matrix)
        return {
            "clique": clique,
            "clique_size": len(clique),
            "valid": is_clique(matrix, clique),
            "planted_clique": planted,
        }

    if mode == "dqn":
        from .dqn import DQNConfig, train_dqn_clique

        result = train_dqn_clique(
            matrix,
            initial_clique=initial_clique,
            config=DQNConfig(**(dqn_config or {})),
            device=device,
        )
        return {
            "clique": result["clique"],
            "clique_size": result["clique_size"],
            "valid": result["valid"],
            "episode_rewards": result["episode_rewards"],
            "final_epsilon": result["final_epsilon"],
            "config": result["config"],
            "device": result["device"],
            "planted_clique": planted,
        }

    raise ValueError(f"unknown max_clique_rl mode: {mode}")
