from __future__ import annotations

from dataclasses import dataclass
import math

from .config import AdaptiveMomentumConfig


@dataclass
class CDKControlState:
    momentum: float
    previous_energy_score: float | None = None


def update_momentum(state: CDKControlState, *, energy_gap: float, config: AdaptiveMomentumConfig) -> None:
    score = abs(float(energy_gap))
    if config.enabled and state.previous_energy_score is not None and math.isfinite(score):
        relative = (score - state.previous_energy_score) / (abs(state.previous_energy_score) + 1e-8)
        if relative < config.improve_threshold:
            state.momentum = min(state.momentum + config.improve_increment, config.improve_cap)
        elif abs(relative) <= config.stable_threshold:
            state.momentum = min(state.momentum + config.stable_increment, config.stable_cap)
        else:
            state.momentum = max(state.momentum * config.failure_scale, config.minimum)
    state.previous_energy_score = score
