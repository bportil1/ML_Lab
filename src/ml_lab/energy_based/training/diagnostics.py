from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DistributionDiagnostics:
    wasserstein_feature_mean: float
    wasserstein_feature_max: float
    sinkhorn_distance: float
    generated_sample_count: int


@dataclass(frozen=True)
class DistributionSnapshot:
    reference: Any
    generated: Any
    diagnostics: DistributionDiagnostics


@dataclass(frozen=True)
class PartitionEstimate:
    log_z: float
    method: str
    sample_count: int
    effective_sample_size: float | None
    proposal: str

    def to_record(self) -> dict[str, Any]:
        return {
            "log_z": float(self.log_z),
            "method": self.method,
            "sample_count": int(self.sample_count),
            "effective_sample_size": self.effective_sample_size,
            "proposal": self.proposal,
        }


def checkpoint_due(epoch: int, total_epochs: int, *, mode: str, interval: int) -> bool:
    epoch = int(epoch)
    total_epochs = int(total_epochs)
    interval = int(interval)
    if interval < 1:
        raise ValueError("interval must be >= 1")
    if total_epochs < 1:
        raise ValueError("total_epochs must be >= 1")
    if epoch <= 0 or epoch >= total_epochs - 1:
        return True
    if mode == "final_only":
        return False
    if mode == "interval":
        return epoch % interval == 0
    logarithmic = epoch > 0 and (epoch & (epoch - 1) == 0)
    if mode == "logarithmic":
        return logarithmic
    if mode == "hybrid":
        return logarithmic or epoch % interval == 0
    raise ValueError(f"unknown checkpoint mode {mode!r}")


def mean_log_unnormalized_probability(model, data) -> float:
    with __import__("torch").no_grad():
        values = -model.energy(data)
        return float(values.mean().detach().cpu())


def reconstruction_error(model, data) -> float:
    torch = __import__("torch")
    x = model._coerce_visible(data)
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            hidden = model.hidden_probability(x)
            mean = model.visible_mean(hidden)
            return float(torch.mean((x - mean) ** 2).detach().cpu())
    finally:
        model.train(was_training)


def matrix_condition_number(matrix) -> float:
    torch = __import__("torch")
    with torch.no_grad():
        singular = torch.linalg.svdvals(matrix.detach())
        if singular.numel() == 0:
            return float("nan")
        largest = singular.max()
        smallest = singular.min()
        if float(smallest.detach().cpu()) <= 1e-12:
            return float("inf")
        return float((largest / smallest).detach().cpu())


def collect_epoch_metrics(model, data, *, update_norms: list[float], cd_objectives: list[float], energy_sample_size: int) -> dict[str, float]:
    torch = __import__("torch")
    x = model._coerce_visible(data)
    state = _save_rng(torch, model)
    model.eval()
    try:
        with torch.no_grad():
            count = min(int(energy_sample_size), int(x.shape[0]))
            sample = x[torch.randperm(x.shape[0], device=x.device)[:count]]
            positive_energy, negative_energy, energy_gap = model.compute_energy_gap(sample)
            hidden = model.hidden_probability(sample).clamp(1e-8, 1.0 - 1e-8)
            hidden_entropy = -(hidden * torch.log(hidden) + (1.0 - hidden) * torch.log(1.0 - hidden))
    finally:
        _restore_rng(torch, model, state)

    record = {
        "energy_positive": float(positive_energy),
        "energy_negative": float(negative_energy),
        "energy_gap": float(energy_gap),
        "reconstruction_error": reconstruction_error(model, x),
        "cd_objective": float(np.mean(cd_objectives)) if cd_objectives else float("nan"),
        "update_norm": float(np.mean(update_norms)) if update_norms else 0.0,
        "weight_norm": float(model.W.detach().norm().cpu()),
        "condition_number_W": matrix_condition_number(model.W),
        "visible_bias_norm": float(model.v_bias.detach().norm().cpu()),
        "hidden_bias_norm": float(model.h_bias.detach().norm().cpu()),
        "hidden_activation_mean": float(hidden.mean().detach().cpu()),
        "hidden_activation_std": float(hidden.std(unbiased=False).detach().cpu()),
        "hidden_entropy_mean": float(hidden_entropy.mean().detach().cpu()),
        "mean_log_unnormalized_probability": mean_log_unnormalized_probability(model, sample),
    }
    if model.family in {"gaussian", "student_t_poe"}:
        sigma = torch.exp(model.log_sigma).detach()
        record.update({
            "log_sigma_mean": float(model.log_sigma.detach().mean().cpu()),
            "log_sigma_std": float(model.log_sigma.detach().std(unbiased=False).cpu()),
            "sigma_mean": float(sigma.mean().cpu()),
            "sigma_std": float(sigma.std(unbiased=False).cpu()),
        })
    if model.family == "student_t_poe":
        nu = torch.exp(model.log_nu).detach()
        record.update({"log_nu": float(model.log_nu.detach().cpu()), "nu": float(nu.cpu())})
    return record


def _wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    count = max(len(a), len(b))
    q = (np.arange(count, dtype=float) + 0.5) / count
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def featurewise_wasserstein(reference, generated) -> tuple[float, float]:
    x = reference.detach().cpu().numpy()
    y = generated.detach().cpu().numpy()
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("reference and generated samples must be 2-D with the same feature count")
    distances = np.asarray([_wasserstein_1d(x[:, i], y[:, i]) for i in range(x.shape[1])], dtype=float)
    return float(distances.mean()), float(distances.max())


def sinkhorn_distance(reference, generated, *, p: int = 2, regularization: float = 0.1, iterations: int = 50) -> float:
    """Balanced entropic-regularized Wasserstein approximation.

    This is intentionally reported as a Sinkhorn distance, not exact EMD.
    """
    torch = __import__("torch")
    x = reference.float()
    y = generated.float()
    cost = torch.cdist(x, y, p=float(p)).pow(int(p))
    eps = float(regularization)
    log_kernel = -cost / eps
    log_a = torch.full((x.shape[0],), -math.log(x.shape[0]), device=x.device, dtype=x.dtype)
    log_b = torch.full((y.shape[0],), -math.log(y.shape[0]), device=y.device, dtype=y.dtype)
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for _ in range(int(iterations)):
        log_u = log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0)
    plan = torch.exp(log_u.unsqueeze(1) + log_kernel + log_v.unsqueeze(0))
    value = (plan * cost).sum().clamp_min(0)
    if int(p) > 1:
        value = value.pow(1.0 / int(p))
    return float(value.detach().cpu())


def _save_rng(torch, model):
    cpu = torch.random.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    mode = bool(model.training)
    return cpu, cuda, mode


def _restore_rng(torch, model, state) -> None:
    cpu, cuda, mode = state
    torch.random.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)
    model.train(mode)


def collect_distribution_snapshot(
    model,
    reference,
    *,
    sample_size: int,
    gibbs_steps: int,
    sinkhorn_enabled: bool,
    sinkhorn_p: int,
    sinkhorn_regularization: float,
    sinkhorn_iterations: int,
) -> DistributionSnapshot:
    torch = __import__("torch")
    state = _save_rng(torch, model)
    model.eval()
    try:
        x = model._coerce_visible(reference)
        count = min(int(sample_size), int(x.shape[0]))
        indices = torch.randperm(x.shape[0], device=x.device)[:count]
        comparison = x[indices].detach().clone()
        generated = comparison.clone()
        with torch.no_grad():
            for _ in range(int(gibbs_steps)):
                generated = model.gibbs_step(generated)
        w_mean, w_max = featurewise_wasserstein(comparison, generated)
        sinkhorn = (
            sinkhorn_distance(
                comparison,
                generated,
                p=sinkhorn_p,
                regularization=sinkhorn_regularization,
                iterations=sinkhorn_iterations,
            )
            if sinkhorn_enabled else float("nan")
        )
        return DistributionSnapshot(
            reference=comparison.detach().cpu(),
            generated=generated.detach().cpu(),
            diagnostics=DistributionDiagnostics(
                wasserstein_feature_mean=w_mean,
                wasserstein_feature_max=w_max,
                sinkhorn_distance=sinkhorn,
                generated_sample_count=int(generated.shape[0]),
            ),
        )
    finally:
        _restore_rng(torch, model, state)


def _exact_bernoulli_log_partition(model, *, chunk_size: int) -> PartitionEstimate:
    torch = __import__("torch")
    visible_dim = int(model.num_visible)
    total = 1 << visible_dim
    running = None
    shifts = torch.arange(visible_dim, device=model.W.device, dtype=torch.int64)
    with torch.no_grad():
        for start in range(0, total, int(chunk_size)):
            values = torch.arange(start, min(start + chunk_size, total), device=model.W.device, dtype=torch.int64)
            states = ((values[:, None] >> shifts[None, :]) & 1).to(dtype=model.W.dtype)
            chunk = torch.logsumexp(-model.energy(states), dim=0)
            running = chunk if running is None else torch.logaddexp(running, chunk)
    return PartitionEstimate(
        log_z=float(running.detach().cpu()),
        method="exact_bernoulli_enumeration",
        sample_count=int(total),
        effective_sample_size=float(total),
        proposal="enumerated_visible_states",
    )


def _importance_log_partition(
    model,
    *,
    sample_count: int,
    seed: int,
    bernoulli_probability: float,
    gaussian_std: float,
    student_t_df: float,
) -> PartitionEstimate:
    torch = __import__("torch")
    state = _save_rng(torch, model)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model.eval()
    try:
        n = int(sample_count)
        d = int(model.num_visible)
        device = model.W.device
        dtype = model.W.dtype
        if model.family == "bernoulli":
            p = float(bernoulli_probability)
            samples = torch.bernoulli(torch.full((n, d), p, device=device, dtype=dtype))
            log_q = (samples * math.log(p) + (1.0 - samples) * math.log(1.0 - p)).sum(dim=1)
            proposal = f"independent_bernoulli(p={p:g})"
        elif model.family == "gaussian":
            std = float(gaussian_std)
            samples = torch.randn(n, d, device=device, dtype=dtype) * std
            log_q = -0.5 * ((samples / std).square() + math.log(2.0 * math.pi * std * std)).sum(dim=1)
            proposal = f"independent_normal(std={std:g})"
        else:
            df = float(student_t_df)
            distribution = torch.distributions.StudentT(df, loc=torch.tensor(0.0, device=device), scale=torch.tensor(1.0, device=device))
            samples = distribution.sample((n, d)).to(dtype=dtype)
            log_q = distribution.log_prob(samples).sum(dim=1)
            proposal = f"independent_student_t(df={df:g})"
        with torch.no_grad():
            log_weights = -model.energy(samples) - log_q
            log_sum = torch.logsumexp(log_weights, dim=0)
            log_z = log_sum - math.log(n)
            normalized = torch.exp(log_weights - log_sum)
            ess = 1.0 / normalized.square().sum().clamp_min(1e-30)
        return PartitionEstimate(
            log_z=float(log_z.detach().cpu()),
            method="importance_sampling",
            sample_count=n,
            effective_sample_size=float(ess.detach().cpu()),
            proposal=proposal,
        )
    finally:
        _restore_rng(torch, model, state)


def estimate_log_partition(model, config, *, epoch: int = 0) -> PartitionEstimate:
    estimator = config.estimator
    if estimator == "exact_bernoulli" and model.family != "bernoulli":
        raise ValueError("exact_bernoulli partition estimation is only available for Bernoulli RBMs")
    use_exact = (
        model.family == "bernoulli"
        and estimator in {"auto", "exact_bernoulli"}
        and int(model.num_visible) <= int(config.exact_bernoulli_visible_limit)
    )
    if estimator == "exact_bernoulli" and not use_exact:
        raise ValueError(
            "Bernoulli RBM exceeds exact_bernoulli_visible_limit; use estimator='importance' or raise the limit explicitly"
        )
    if use_exact:
        return _exact_bernoulli_log_partition(model, chunk_size=config.exact_chunk_size)
    return _importance_log_partition(
        model,
        sample_count=config.sample_count,
        seed=int(config.seed) + int(epoch),
        bernoulli_probability=config.bernoulli_proposal_probability,
        gaussian_std=config.gaussian_proposal_std,
        student_t_df=config.student_t_proposal_df,
    )


def mean_log_likelihood(model, data, partition: PartitionEstimate) -> float:
    return mean_log_unnormalized_probability(model, data) - float(partition.log_z)
