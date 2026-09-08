from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ml_lab.neural.config import NeuralTrainingConfig

GANArchitecture = Literal["mlp", "transformer"]
ActivationName = Literal["relu", "gelu", "tanh", "leaky_relu"]
OutputActivationName = Literal["none", "sigmoid", "tanh"]
ScalingMode = Literal["none", "standard", "minmax", "robust"]
MonitorName = Literal["generator_loss", "discriminator_loss"]


@dataclass(frozen=True, slots=True)
class GANModelConfig:
    latent_dim: int = 32
    generator_type: GANArchitecture = "mlp"
    discriminator_type: GANArchitecture = "mlp"
    generator_hidden_dims: tuple[int, ...] = (64, 128)
    discriminator_hidden_dims: tuple[int, ...] = (128, 64)
    generator_activation: ActivationName = "relu"
    discriminator_activation: ActivationName = "leaky_relu"
    dropout: float = 0.0
    output_activation: OutputActivationName = "none"
    token_width: int = 4
    model_dim: int = 64
    nhead: int = 4
    transformer_layers: int = 2
    feedforward_dim: int = 128
    transformer_dropout: float = 0.0
    transformer_activation: Literal["relu", "gelu"] = "gelu"
    max_tokens: int = 1024
    norm_first: bool = False

    def __post_init__(self) -> None:
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        if not self.generator_hidden_dims or not self.discriminator_hidden_dims:
            raise ValueError("generator_hidden_dims and discriminator_hidden_dims cannot be empty")
        if any(int(dim) < 1 for dim in self.generator_hidden_dims + self.discriminator_hidden_dims):
            raise ValueError("hidden dimensions must contain positive integers")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.token_width < 1:
            raise ValueError("token_width must be positive")
        if self.model_dim < 1 or self.feedforward_dim < 1:
            raise ValueError("model_dim and feedforward_dim must be positive")
        if self.nhead < 1 or self.model_dim % self.nhead != 0:
            raise ValueError("nhead must be positive and evenly divide model_dim")
        if self.transformer_layers < 1:
            raise ValueError("transformer_layers must be positive")
        if not 0.0 <= self.transformer_dropout < 1.0:
            raise ValueError("transformer_dropout must be in [0, 1)")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")


@dataclass(frozen=True, slots=True)
class GANTrainingConfig(NeuralTrainingConfig):
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 2e-4
    validation_fraction: float = 0.0
    patience: int | None = None
    restore_best: bool = False
    scaling: ScalingMode = "standard"
    generator_learning_rate: float | None = None
    discriminator_learning_rate: float | None = None
    adam_beta1: float = 0.5
    adam_beta2: float = 0.999
    generator_steps: int = 1
    discriminator_steps: int = 1
    real_label_smoothing: float = 0.0
    gradient_clip_norm: float | None = None
    generated_sample_count: int | None = None
    monitor: MonitorName = "generator_loss"

    def __post_init__(self) -> None:
        NeuralTrainingConfig.__post_init__(self)
        if self.validation_fraction != 0.0:
            raise ValueError("GANTrainingConfig validation_fraction must remain 0.0; stable GAN training does not use a validation split")
        for name, value in (
            ("generator_learning_rate", self.generator_learning_rate),
            ("discriminator_learning_rate", self.discriminator_learning_rate),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when provided")
        if not 0.0 <= self.adam_beta1 < 1.0 or not 0.0 <= self.adam_beta2 < 1.0:
            raise ValueError("Adam beta values must be in [0, 1)")
        if self.generator_steps < 1 or self.discriminator_steps < 1:
            raise ValueError("generator_steps and discriminator_steps must be positive")
        if not 0.0 <= self.real_label_smoothing < 1.0:
            raise ValueError("real_label_smoothing must be in [0, 1)")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive when provided")
        if self.generated_sample_count is not None and self.generated_sample_count < 1:
            raise ValueError("generated_sample_count must be positive when provided")
