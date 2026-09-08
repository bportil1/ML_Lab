from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ml_lab.neural.config import NeuralTrainingConfig

ScalingMode = Literal["none", "standard", "minmax", "robust"]
ActivationName = Literal["relu", "gelu", "tanh", "leaky_relu"]
OutputActivationName = Literal["none", "sigmoid", "tanh"]
TransformerActivationName = Literal["relu", "gelu"]

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
class TransformerAutoencoderConfig:
    token_width: int = 1
    model_dim: int = 64
    latent_dim: int = 16
    nhead: int = 4
    encoder_layers: int = 2
    decoder_layers: int = 2
    feedforward_dim: int = 128
    dropout: float = 0.0
    transformer_activation: TransformerActivationName = "gelu"
    output_activation: OutputActivationName = "none"
    max_tokens: int = 1024
    norm_first: bool = False

    def __post_init__(self) -> None:
        if self.token_width < 1:
            raise ValueError("token_width must be positive")
        if self.model_dim < 1 or self.latent_dim < 1 or self.feedforward_dim < 1:
            raise ValueError("model_dim, latent_dim, and feedforward_dim must be positive")
        if self.nhead < 1 or self.model_dim % self.nhead != 0:
            raise ValueError("nhead must be positive and evenly divide model_dim")
        if self.encoder_layers < 1 or self.decoder_layers < 1:
            raise ValueError("encoder_layers and decoder_layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")


@dataclass(frozen=True, slots=True)
class AutoencoderTrainingConfig(NeuralTrainingConfig):
    scaling: ScalingMode = "standard"

    def __post_init__(self) -> None:
        NeuralTrainingConfig.__post_init__(self)
