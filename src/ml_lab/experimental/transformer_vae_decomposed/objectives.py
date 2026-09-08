from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(slots=True)
class DecomposedKLEstimate:
    """Differentiable minibatch estimate of the beta-TCVAE KL decomposition."""

    icmi: Any
    total_correlation: Any
    dimension_wise_kl: Any
    estimated_kl: Any
    analytic_kl: Any

    def weighted(self, *, icmi_weight: float, tc_weight: float, dwkl_weight: float):
        return (
            float(icmi_weight) * self.icmi
            + float(tc_weight) * self.total_correlation
            + float(dwkl_weight) * self.dimension_wise_kl
        )


def _log_normal_diag(torch: Any, value: Any, mean: Any, logvar: Any):
    return -0.5 * (
        math.log(2.0 * math.pi)
        + logvar
        + (value - mean).pow(2) * torch.exp(-logvar)
    )


def estimate_decomposed_kl(torch: Any, z: Any, mu: Any, logvar: Any) -> DecomposedKLEstimate:
    """Estimate ICMI, total correlation, and dimension-wise KL from a minibatch.

    ``q(z)`` and the marginal ``q(z_j)`` densities are estimated as equally weighted
    mixtures of the minibatch posteriors. This is the standard minibatch-mixture form
    of the beta-TCVAE decomposition, not an exact full-dataset density estimate.

    The terms satisfy, up to floating-point error,::

        estimated_kl = icmi + total_correlation + dimension_wise_kl

    where ``estimated_kl`` is the corresponding minibatch density-ratio estimate of
    ``E[log q(z|x) - log p(z)]``. ``analytic_kl`` is also returned as an independent
    diagonal-Gaussian diagnostic against the N(0, I) prior.
    """
    if z.ndim != 2 or mu.ndim != 2 or logvar.ndim != 2:
        raise ValueError("z, mu, and logvar must have shape (batch, latent_dim)")
    if z.shape != mu.shape or mu.shape != logvar.shape:
        raise ValueError("z, mu, and logvar must have matching shapes")
    batch_size = int(z.shape[0])
    if batch_size < 2:
        raise ValueError("decomposed KL estimation requires at least two samples")

    # log q(z_i | x_i)
    log_q_z_given_x = _log_normal_diag(torch, z, mu, logvar).sum(dim=1)

    # Pairwise log q(z_i | x_j), retaining dimensions so both q(z) and q(z_j)
    # are derived from the same minibatch-mixture estimator.
    pairwise = _log_normal_diag(
        torch,
        z.unsqueeze(1),
        mu.unsqueeze(0),
        logvar.unsqueeze(0),
    )  # (B, B, D)
    log_batch = math.log(float(batch_size))
    log_q_z = torch.logsumexp(pairwise.sum(dim=2), dim=1) - log_batch
    log_q_z_product = (
        torch.logsumexp(pairwise, dim=1) - log_batch
    ).sum(dim=1)

    zeros = torch.zeros_like(z)
    log_p_z = _log_normal_diag(torch, z, zeros, zeros).sum(dim=1)

    icmi = (log_q_z_given_x - log_q_z).mean()
    total_correlation = (log_q_z - log_q_z_product).mean()
    dimension_wise_kl = (log_q_z_product - log_p_z).mean()
    estimated_kl = icmi + total_correlation + dimension_wise_kl
    analytic_kl = -0.5 * torch.mean(
        torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    )
    return DecomposedKLEstimate(
        icmi=icmi,
        total_correlation=total_correlation,
        dimension_wise_kl=dimension_wise_kl,
        estimated_kl=estimated_kl,
        analytic_kl=analytic_kl,
    )


__all__ = ["DecomposedKLEstimate", "estimate_decomposed_kl"]
