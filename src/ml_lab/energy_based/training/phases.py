from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NegativePhase:
    visible: object
    persistent_state: object | None


def prepare_visible_batch(batch, family: str, *, clamp_visible: bool, epsilon: float = 1e-6):
    if family == "bernoulli" and clamp_visible:
        return batch.clamp(epsilon, 1.0 - epsilon)
    return batch


def negative_phase(
    model,
    batch,
    *,
    gibbs_steps: int,
    num_chains: int,
    persistent: bool,
    persistent_state=None,
    clamp_visible: bool,
):
    batch_size, visible_dim = batch.shape
    expected_shape = (int(num_chains), int(batch_size), int(visible_dim))
    if persistent and persistent_state is not None and tuple(persistent_state.shape) == expected_shape:
        states = persistent_state.detach().clone()
    else:
        states = batch.detach().unsqueeze(0).expand(num_chains, -1, -1).clone()

    for _ in range(int(gibbs_steps)):
        flat = states.reshape(num_chains * batch_size, visible_dim)
        flat = model.gibbs_step(flat)
        flat = prepare_visible_batch(flat, model.family, clamp_visible=clamp_visible)
        states = flat.reshape(num_chains, batch_size, visible_dim)

    return NegativePhase(
        visible=states.reshape(num_chains * batch_size, visible_dim).detach(),
        persistent_state=states.detach().clone() if persistent else None,
    )
