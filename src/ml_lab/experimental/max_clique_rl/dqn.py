from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Any

import numpy as np

from .graph import is_clique, valid_additions, validate_adjacency


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "The experimental max-clique DQN requires PyTorch. "
            "Install ML Lab with the experimental-neural extra or install torch separately."
        ) from exc
    return torch, nn


@dataclass(frozen=True, slots=True)
class DQNConfig:
    episodes: int = 100
    learning_rate: float = 1e-3
    gamma: float = 0.9
    epsilon: float = 1.0
    epsilon_min: float = 0.1
    epsilon_decay: float = 0.995
    hidden_dim: int = 128
    seed: int = 42

    def validate(self) -> None:
        if self.episodes <= 0 or self.learning_rate <= 0 or self.hidden_dim <= 0:
            raise ValueError("episodes, learning_rate, and hidden_dim must be positive")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 <= self.epsilon_min <= self.epsilon <= 1.0:
            raise ValueError("epsilon settings must satisfy 0 <= epsilon_min <= epsilon <= 1")
        if not 0.0 < self.epsilon_decay <= 1.0:
            raise ValueError("epsilon_decay must be in (0, 1]")

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def train_dqn_clique(
    graph,
    *,
    initial_clique: list[int] | None = None,
    config: DQNConfig | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Run the recovered graph-specific DQN idea as an isolated prototype.

    This deliberately remains a simple one-step Q-learning style prototype; it
    does not claim to be ML Lab's future general DQN implementation.
    """
    torch, nn = _require_torch()
    matrix = validate_adjacency(graph)
    config = config or DQNConfig()
    config.validate()
    if initial_clique is None:
        initial_clique = []
    initial_clique = [int(node) for node in initial_clique]
    if initial_clique and not is_clique(matrix, initial_clique):
        raise ValueError("initial_clique is not a valid clique")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    n_nodes = matrix.shape[0]

    class DQN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(n_nodes, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, n_nodes),
            )

        def forward(self, x):
            return self.network(x)

    model = DQN().to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    epsilon = config.epsilon
    best = list(initial_clique)
    episode_rewards: list[int] = []

    def state_tensor(clique):
        state = torch.zeros(n_nodes, dtype=torch.float32, device=resolved_device)
        if clique:
            state[torch.tensor(clique, dtype=torch.long, device=resolved_device)] = 1.0
        return state

    for _ in range(config.episodes):
        clique = list(initial_clique)
        state = state_tensor(clique)
        while True:
            valid = valid_additions(matrix, clique)
            if not valid:
                break
            if random.random() < epsilon:
                action = random.choice(valid)
            else:
                with torch.no_grad():
                    q_values = model(state)
                    mask = torch.full_like(q_values, float("-inf"))
                    mask[torch.tensor(valid, dtype=torch.long, device=resolved_device)] = q_values[
                        torch.tensor(valid, dtype=torch.long, device=resolved_device)
                    ]
                    action = int(torch.argmax(mask).item())

            next_clique = clique + [action]
            next_state = state_tensor(next_clique)
            reward = float(len(next_clique))
            prediction = model(state)[action]
            with torch.no_grad():
                next_valid = valid_additions(matrix, next_clique)
                if next_valid:
                    q_next = model(next_state)[torch.tensor(next_valid, dtype=torch.long, device=resolved_device)]
                    target = reward + config.gamma * torch.max(q_next)
                else:
                    target = torch.tensor(reward, device=resolved_device)
            loss = loss_fn(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            clique = next_clique
            state = next_state

        if len(clique) > len(best):
            best = clique
        episode_rewards.append(len(clique))
        epsilon = max(config.epsilon_min, epsilon * config.epsilon_decay)

    return {
        "clique": best,
        "clique_size": len(best),
        "valid": is_clique(matrix, best),
        "episode_rewards": episode_rewards,
        "final_epsilon": epsilon,
        "config": config.to_record(),
        "device": str(resolved_device),
        "model": model,
    }
