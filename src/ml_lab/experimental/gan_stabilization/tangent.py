from __future__ import annotations

from typing import Any


def score_tangent_negatives(discriminator: Any, samples: Any, *, epsilon: float = 0.05, count: int = 1):
    """Generate first-order perturbations tangent to discriminator-score level sets.

    Random directions are projected orthogonally to the local discriminator-score
    gradient. The resulting perturbation therefore has zero first-order directional
    derivative of the score, up to numerical error.
    """
    import torch

    if samples.ndim != 2:
        raise ValueError("samples must be 2-D")
    x = samples.detach().clone().requires_grad_(True)
    logits = discriminator(x)
    grad = torch.autograd.grad(logits.sum(), x, create_graph=False)[0]
    grad_norm = grad.norm(dim=1, keepdim=True).clamp_min(1e-12)
    normal = grad / grad_norm
    outputs = []
    for _ in range(int(count)):
        noise = torch.randn_like(x)
        tangent = noise - (noise * normal).sum(dim=1, keepdim=True) * normal
        tangent = tangent / tangent.norm(dim=1, keepdim=True).clamp_min(1e-12)
        outputs.append((x + float(epsilon) * tangent).detach())
    return torch.cat(outputs, dim=0)
