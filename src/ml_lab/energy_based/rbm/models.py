from __future__ import annotations

from typing import Any

from ml_lab.neural.backend import require_torch

from .sampling import hidden_dropout, sharp_sigmoid, validate_dropout, validate_sharpness


torch = require_torch(purpose="RBM models")
nn = torch.nn
F = torch.nn.functional


class _RBMBase(nn.Module):
    family: str

    def __init__(self, visible_dim: int, hidden_dim: int, *, sharpness: float, dropout: float, settings: dict[str, Any]):
        super().__init__()
        self.num_visible = int(visible_dim)
        self.num_hidden = int(hidden_dim)
        if self.num_visible <= 0 or self.num_hidden <= 0:
            raise ValueError("RBM visible and hidden dimensions must be > 0")
        self.settings = dict(settings)
        self.sharpness = validate_sharpness(sharpness)
        self.dropout_probability = validate_dropout(dropout)

    @property
    def dropout(self) -> float:
        return self.dropout_probability

    def _coerce_visible(self, value):
        tensor = torch.as_tensor(value, device=self.W.device, dtype=self.W.dtype)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[1] != self.num_visible:
            raise ValueError(
                f"visible input must have shape (batch, {self.num_visible}); got {tuple(tensor.shape)}"
            )
        return tensor

    def _coerce_hidden(self, value):
        tensor = torch.as_tensor(value, device=self.W.device, dtype=self.W.dtype)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[1] != self.num_hidden:
            raise ValueError(
                f"hidden input must have shape (batch, {self.num_hidden}); got {tuple(tensor.shape)}"
            )
        return tensor

    def hidden_activation(self, visible):
        return hidden_dropout(
            self.hidden_probability(visible),
            self.dropout_probability,
            training=self.training,
        )

    def sample_hidden(self, visible):
        probability = self.hidden_probability(visible)
        sample = torch.bernoulli(probability)
        sample = hidden_dropout(sample, self.dropout_probability, training=self.training)
        return probability, sample

    def forward(self, visible):
        return self.sample_hidden(visible)

    def compute_energy_gap(self, visible):
        visible = self._coerce_visible(visible)
        negative = self.gibbs_step(visible)
        positive_energy = self.energy(visible).mean()
        negative_energy = self.energy(negative).mean()
        # Positive means the observed samples receive lower energy than one-step negatives.
        return (
            float(positive_energy.detach().cpu()),
            float(negative_energy.detach().cpu()),
            float((negative_energy - positive_energy).detach().cpu()),
        )

    def configuration(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "visible_dim": self.num_visible,
            "hidden_dim": self.num_hidden,
            "sharpness": self.sharpness,
            "dropout": self.dropout_probability,
            "settings": dict(self.settings),
        }


class BernoulliRBM(_RBMBase):
    family = "bernoulli"

    def __init__(self, visible_dim: int, hidden_dim: int, *, sharpness: float, dropout: float, settings: dict[str, Any]):
        super().__init__(visible_dim, hidden_dim, sharpness=sharpness, dropout=dropout, settings=settings)
        self.W = nn.Parameter(torch.randn(self.num_visible, self.num_hidden) * float(settings["weight_init_std"]))
        self.v_bias = nn.Parameter(torch.randn(self.num_visible) * float(settings["visible_bias_init_std"]))
        self.h_bias = nn.Parameter(torch.randn(self.num_hidden) * float(settings["hidden_bias_init_std"]))

    def hidden_probability(self, visible):
        visible = self._coerce_visible(visible)
        logits = visible @ self.W + self.h_bias
        return sharp_sigmoid(logits, self.sharpness).clamp(0.0, 1.0)

    def visible_mean(self, hidden):
        hidden = self._coerce_hidden(hidden)
        logits = hidden @ self.W.T + self.v_bias
        return sharp_sigmoid(logits, self.sharpness).clamp(0.0, 1.0)

    def sample_visible(self, hidden):
        probability = self.visible_mean(hidden)
        return probability, torch.bernoulli(probability)

    def gibbs_step(self, visible):
        _, hidden = self.sample_hidden(visible)
        _, new_visible = self.sample_visible(hidden)
        return new_visible.detach()

    def energy(self, visible):
        visible = self._coerce_visible(visible)
        visible_bias_term = visible @ self.v_bias
        hidden_term = F.softplus(visible @ self.W + self.h_bias).sum(dim=1)
        return -visible_bias_term - hidden_term


class GaussianRBM(_RBMBase):
    family = "gaussian"

    def __init__(self, visible_dim: int, hidden_dim: int, *, sharpness: float, dropout: float, settings: dict[str, Any]):
        super().__init__(visible_dim, hidden_dim, sharpness=sharpness, dropout=dropout, settings=settings)
        self.W = nn.Parameter(torch.randn(self.num_visible, self.num_hidden) * float(settings["weight_init_std"]))
        self.v_bias = nn.Parameter(torch.randn(self.num_visible) * float(settings["visible_bias_init_std"]))
        self.h_bias = nn.Parameter(torch.randn(self.num_hidden) * float(settings["hidden_bias_init_std"]))
        self.log_sigma = nn.Parameter(torch.full((self.num_visible,), float(settings["initial_log_sigma"])))

    def _sigma(self, *, minimum: float):
        return torch.exp(self.log_sigma).clamp(float(minimum), float(self.settings["sigma_max"]))

    def hidden_probability(self, visible, eps: float = 1e-6):
        visible = self._coerce_visible(visible)
        sigma = self._sigma(minimum=float(self.settings["hidden_sigma_min"]))
        input_clip = float(self.settings["hidden_input_clip"])
        preactivation_clip = float(self.settings["preactivation_clip"])
        scaled = torch.clamp(visible / sigma, -input_clip, input_clip)
        preactivation = torch.clamp(scaled @ self.W + self.h_bias, -preactivation_clip, preactivation_clip)
        preactivation = torch.nan_to_num(preactivation, nan=0.0, posinf=preactivation_clip, neginf=-preactivation_clip)
        probability = sharp_sigmoid(preactivation, self.sharpness)
        probability = torch.nan_to_num(probability, nan=0.5, posinf=1.0, neginf=0.0)
        return probability.clamp(eps, 1.0 - eps)

    def visible_mean(self, hidden):
        hidden = self._coerce_hidden(hidden)
        return hidden @ self.W.T + self.v_bias

    def sample_visible(self, hidden):
        mean = self.visible_mean(hidden)
        sigma = torch.exp(self.log_sigma).clamp_min(1e-8)
        noise = torch.randn_like(mean) * sigma * float(self.settings["visible_noise_scale"])
        sample = mean + noise
        return (
            torch.nan_to_num(mean, nan=0.0, posinf=1e6, neginf=-1e6),
            torch.nan_to_num(sample, nan=0.0, posinf=1e6, neginf=-1e6),
        )

    def gibbs_step(self, visible):
        _, hidden = self.sample_hidden(visible)
        _, new_visible = self.sample_visible(hidden)
        return new_visible.detach()

    def energy(self, visible):
        visible = self._coerce_visible(visible)
        sigma = self._sigma(minimum=float(self.settings["energy_sigma_min"]))
        sigma2 = sigma.square()
        visible_term = 0.5 * (((visible - self.v_bias).square()) / sigma2).sum(dim=1)
        input_clip = float(self.settings["hidden_input_clip"])
        scaled = torch.clamp(visible / sigma, -input_clip, input_clip)
        hidden_term = F.softplus(scaled @ self.W + self.h_bias).sum(dim=1)
        return visible_term - hidden_term


class StudentTPoERBM(_RBMBase):
    family = "student_t_poe"

    def __init__(self, visible_dim: int, hidden_dim: int, *, sharpness: float, dropout: float, settings: dict[str, Any]):
        super().__init__(visible_dim, hidden_dim, sharpness=sharpness, dropout=dropout, settings=settings)
        self.nv = self.num_visible
        self.nh = self.num_hidden
        self.W = nn.Parameter(torch.randn(self.num_visible, self.num_hidden) * float(settings["weight_init_std"]))
        self.v_bias = nn.Parameter(torch.randn(self.num_visible) * float(settings["visible_bias_init_std"]))
        self.h_bias = nn.Parameter(torch.randn(self.num_hidden) * float(settings["hidden_bias_init_std"]))
        self.log_sigma = nn.Parameter(torch.full((self.num_visible,), float(settings["initial_log_sigma"])))
        self.log_nu = nn.Parameter(torch.tensor(float(settings["initial_log_nu"])))

    def _nu(self):
        return torch.exp(self.log_nu).clamp(float(self.settings["nu_min"]), float(self.settings["nu_max"]))

    def _sigma(self):
        # ``log_sigma`` has one unambiguous parametrization throughout this model.
        return torch.exp(self.log_sigma).clamp(
            float(self.settings["sigma_sample_min"]),
            float(self.settings["sigma_sample_max"]),
        )

    def hidden_probability(self, visible):
        visible = self._coerce_visible(visible)
        sigma2 = self._sigma().square()
        clip = float(self.settings["preactivation_clip"])
        preactivation = torch.clamp((visible / sigma2) @ self.W + self.h_bias, -clip, clip)
        probability = sharp_sigmoid(preactivation, self.sharpness)
        return torch.nan_to_num(probability, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    def compute_lambda(self, visible, hidden):
        visible = self._coerce_visible(visible)
        hidden = self._coerce_hidden(hidden)
        mean = hidden @ self.W.T + self.v_bias
        sigma2 = self._sigma().square()
        delta = ((visible - mean).square() / sigma2).sum(dim=1)
        nu = self._nu()
        shape = 0.5 * (nu + self.num_visible)
        rate = 0.5 * (nu + delta)
        expectation = shape / (rate + 1e-8)
        return expectation.clamp(float(self.settings["lambda_min"]), float(self.settings["lambda_max"]))

    def sample_lambda(self, visible, hidden):
        visible = self._coerce_visible(visible)
        hidden = self._coerce_hidden(hidden)
        mean = hidden @ self.W.T + self.v_bias
        sigma2 = self._sigma().square()
        delta = ((visible - mean).square() / sigma2).sum(dim=1)
        nu = self._nu()
        concentration = 0.5 * (nu + self.num_visible)
        rate = 0.5 * (nu + delta)
        distribution = torch.distributions.Gamma(concentration, rate)
        return distribution.sample().clamp(float(self.settings["lambda_min"]), float(self.settings["lambda_max"]))

    def visible_mean(self, hidden):
        hidden = self._coerce_hidden(hidden)
        return hidden @ self.W.T + self.v_bias

    def sample_visible(self, hidden, lam=None):
        hidden = self._coerce_hidden(hidden)
        if lam is None:
            lam = torch.ones(hidden.shape[0], device=self.W.device, dtype=self.W.dtype)
        lam = torch.as_tensor(lam, device=self.W.device, dtype=self.W.dtype).reshape(-1)
        if lam.shape[0] != hidden.shape[0]:
            raise ValueError("lambda must contain one precision value per hidden sample")
        mean = self.visible_mean(hidden)
        sigma2 = self._sigma().square()
        noise = torch.randn_like(mean) * torch.sqrt(sigma2 / lam[:, None].clamp_min(1e-12))
        sample = mean + noise
        return (
            torch.nan_to_num(mean, nan=0.0, posinf=1e6, neginf=-1e6),
            torch.nan_to_num(sample, nan=0.0, posinf=1e6, neginf=-1e6),
        )

    def gibbs_step(self, visible):
        visible = self._coerce_visible(visible)
        _, hidden = self.sample_hidden(visible)
        lam = self.compute_lambda(visible, hidden)
        _, new_visible = self.sample_visible(hidden, lam)
        return new_visible.detach()

    def energy(self, visible):
        visible = self._coerce_visible(visible)
        sigma = self._sigma()
        sigma2 = sigma.square()
        centered = visible - self.v_bias
        quadratic = (centered.square() / sigma2).sum(dim=1)
        nu = self._nu()
        visible_term = 0.5 * (nu + self.num_visible) * torch.log1p(quadratic / nu)
        # Keep the hidden conditional and energy parametrization consistent by using sigma^2.
        hidden_term = F.softplus((visible / sigma2) @ self.W + self.h_bias).sum(dim=1)
        return visible_term - hidden_term
