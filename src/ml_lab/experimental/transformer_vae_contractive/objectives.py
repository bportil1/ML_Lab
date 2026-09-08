from __future__ import annotations

from typing import Any


def _apply_mask(torch: Any, gradient: Any, element_mask: Any | None):
    if element_mask is None:
        return gradient
    if element_mask.shape != gradient.shape:
        raise ValueError("element_mask must match the input tensor shape")
    return gradient * element_mask.to(device=gradient.device, dtype=gradient.dtype)


def _reduce_penalty(
    torch: Any,
    squared_gradient: Any,
    element_mask: Any | None,
    *,
    normalize_by_active_inputs: bool,
):
    per_sample = squared_gradient.reshape(squared_gradient.shape[0], -1).sum(dim=1)
    if normalize_by_active_inputs:
        if element_mask is None:
            active = torch.full_like(
                per_sample,
                float(squared_gradient[0].numel()),
            )
        else:
            active = (
                element_mask.to(dtype=squared_gradient.dtype)
                .reshape(element_mask.shape[0], -1)
                .sum(dim=1)
                .clamp_min(1.0)
            )
        per_sample = per_sample / active
    return per_sample.mean()


def contractive_jacobian_penalty(
    torch: Any,
    posterior_mean: Any,
    inputs: Any,
    *,
    element_mask: Any | None = None,
    estimator: str = "exact",
    hutchinson_samples: int = 1,
    normalize_by_active_inputs: bool = True,
    create_graph: bool = True,
):
    """Estimate ``||d mu(x) / d x||_F^2`` for a minibatch.

    This is a first-order contractive regularizer on the deterministic posterior
    representation map. Padded token elements can be excluded with
    ``element_mask``. With normalization enabled, each sample's squared Jacobian
    norm is divided by its number of active input elements before batch averaging.
    """
    if posterior_mean.ndim != 2:
        raise ValueError("posterior_mean must have shape (batch, latent_dim)")
    if inputs.ndim < 2 or int(inputs.shape[0]) != int(posterior_mean.shape[0]):
        raise ValueError("inputs and posterior_mean must share the batch dimension")
    if not inputs.requires_grad:
        raise ValueError("inputs must require gradients for a contractive penalty")
    if estimator not in {"exact", "hutchinson"}:
        raise ValueError("estimator must be 'exact' or 'hutchinson'")
    if hutchinson_samples < 1:
        raise ValueError("hutchinson_samples must be positive")

    if estimator == "exact":
        total = None
        latent_dim = int(posterior_mean.shape[1])
        for latent_index in range(latent_dim):
            gradient = torch.autograd.grad(
                posterior_mean[:, latent_index].sum(),
                inputs,
                create_graph=create_graph,
                retain_graph=True,
                allow_unused=False,
            )[0]
            gradient = _apply_mask(torch, gradient, element_mask)
            squared = gradient.pow(2)
            total = squared if total is None else total + squared
        return _reduce_penalty(
            torch,
            total,
            element_mask,
            normalize_by_active_inputs=normalize_by_active_inputs,
        )

    estimates = []
    for _ in range(hutchinson_samples):
        # Independent Rademacher vectors per sample give an unbiased estimate of
        # the squared Frobenius norm through E ||J^T v||^2 = ||J||_F^2.
        direction = torch.empty_like(posterior_mean).bernoulli_(0.5).mul_(2.0).sub_(1.0)
        gradient = torch.autograd.grad(
            (posterior_mean * direction).sum(),
            inputs,
            create_graph=create_graph,
            retain_graph=True,
            allow_unused=False,
        )[0]
        gradient = _apply_mask(torch, gradient, element_mask)
        estimates.append(
            _reduce_penalty(
                torch,
                gradient.pow(2),
                element_mask,
                normalize_by_active_inputs=normalize_by_active_inputs,
            )
        )
    return torch.stack(estimates).mean()


__all__ = ["contractive_jacobian_penalty"]
