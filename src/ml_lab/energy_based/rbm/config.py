from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class BernoulliRBMSettings:
    weight_init_std: float = 0.01
    visible_bias_init_std: float = 0.01
    hidden_bias_init_std: float = 0.01


@dataclass(frozen=True)
class GaussianRBMSettings:
    weight_init_std: float = 1.0
    visible_bias_init_std: float = 1.0
    hidden_bias_init_std: float = 1.0
    initial_log_sigma: float = 0.0
    hidden_sigma_min: float = 0.1
    energy_sigma_min: float = 0.01
    sigma_max: float = 5.0
    hidden_input_clip: float = 5.0
    preactivation_clip: float = 20.0
    visible_noise_scale: float = 0.1
    gradient_sigma_min: float = 0.01
    gradient_sigma_max: float = 5.0
    log_sigma_update_min: float = -5.0
    log_sigma_update_max: float = 5.0


@dataclass(frozen=True)
class StudentTPoERBMSettings:
    weight_init_std: float = 0.01
    visible_bias_init_std: float = 0.01
    hidden_bias_init_std: float = 0.01
    initial_log_sigma: float = 0.0
    initial_log_nu: float = math.log(5.0)
    preactivation_clip: float = 20.0
    nu_min: float = 2.0
    nu_max: float = 50.0
    sigma_sample_min: float = 1e-4
    sigma_sample_max: float = 1e4
    lambda_min: float = 1e-4
    lambda_max: float = 1e4
    gradient_sigma_min: float = 1e-4
    gradient_sigma_max: float = 1e4
    gradient_nu_min: float = 1e-3
    gradient_nu_max: float = 1e3


FamilySettings = BernoulliRBMSettings | GaussianRBMSettings | StudentTPoERBMSettings


@dataclass(frozen=True)
class RBMConfig:
    family: str
    visible_dim: int
    hidden_dim: int
    sharpness: float = 1.0
    dropout: float = 0.0
    settings: Mapping[str, Any] | FamilySettings | None = None

    def __post_init__(self) -> None:
        if int(self.visible_dim) <= 0:
            raise ValueError("visible_dim must be > 0")
        if int(self.hidden_dim) <= 0:
            raise ValueError("hidden_dim must be > 0")
        if float(self.sharpness) <= 0:
            raise ValueError("sharpness must be > 0")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")

    def resolved_settings(self) -> dict[str, Any]:
        defaults = default_settings(self.family)
        if self.settings is None:
            return defaults
        supplied = asdict(self.settings) if hasattr(self.settings, "__dataclass_fields__") else dict(self.settings)
        unknown = sorted(set(supplied) - set(defaults))
        if unknown:
            raise ValueError(f"Unknown {canonical_family(self.family)!r} RBM settings: {unknown}")
        return {**defaults, **supplied}

    def to_record(self) -> dict[str, Any]:
        return {
            "family": canonical_family(self.family),
            "visible_dim": int(self.visible_dim),
            "hidden_dim": int(self.hidden_dim),
            "sharpness": float(self.sharpness),
            "dropout": float(self.dropout),
            "settings": self.resolved_settings(),
        }


_ALIASES = {
    "bern": "bernoulli",
    "bernoulli": "bernoulli",
    "gauss": "gaussian",
    "gaussian": "gaussian",
    "stud_t": "student_t_poe",
    "student_t": "student_t_poe",
    "student-t": "student_t_poe",
    "student_t_poe": "student_t_poe",
    "student-t-poe": "student_t_poe",
    "stpoe": "student_t_poe",
}


def canonical_family(family: str) -> str:
    key = str(family).strip().lower()
    try:
        return _ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown RBM family {family!r}. Available families: bernoulli, gaussian, student_t_poe"
        ) from exc


def default_settings(family: str) -> dict[str, Any]:
    canonical = canonical_family(family)
    if canonical == "bernoulli":
        return asdict(BernoulliRBMSettings())
    if canonical == "gaussian":
        return asdict(GaussianRBMSettings())
    return asdict(StudentTPoERBMSettings())
