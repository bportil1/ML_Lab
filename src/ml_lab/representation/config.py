from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ml_lab.neural.config import DeviceName, NeuralTrainingConfig

ScalingMode = Literal["none", "standard", "minmax", "robust"]
ActivationName = Literal["relu", "gelu", "tanh", "leaky_relu"]
OutputActivationName = Literal["none", "sigmoid", "tanh"]

@dataclass(frozen=True, slots=True)
class PCARepresentationConfig:
    n_components: int | float | None = 2
    scaling: ScalingMode = "standard"
    random_state: int = 42

    def __post_init__(self) -> None:
        if isinstance(self.n_components, int) and self.n_components < 1:
            raise ValueError("n_components must be positive")
        if isinstance(self.n_components, float) and not 0.0 < self.n_components <= 1.0:
            raise ValueError("float n_components must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class MLPAutoencoderConfig:
    hidden_dims: tuple[int, ...] = (64, 32)
    latent_dim: int = 8
    activation: ActivationName = "relu"
    output_activation: OutputActivationName = "none"
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if not self.hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")
        if any(int(dim) < 1 for dim in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive integers")
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class AutoencoderTrainingConfig(NeuralTrainingConfig):
    scaling: ScalingMode = "standard"

    def __post_init__(self) -> None:
        NeuralTrainingConfig.__post_init__(self)
