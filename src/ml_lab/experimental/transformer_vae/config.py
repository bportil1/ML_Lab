from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ml_lab.neural.config import NeuralTrainingConfig
from ml_lab.representation.config import OutputActivationName, ScalingMode, TransformerActivationName


@dataclass(frozen=True, slots=True)
class TransformerVAEConfig:
    """Architecture controls for the experimental Transformer VAE."""

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
    logvar_min: float = -12.0
    logvar_max: float = 8.0

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
        if self.logvar_min >= self.logvar_max:
            raise ValueError("logvar_min must be smaller than logvar_max")


@dataclass(frozen=True, slots=True)
class TransformerVAETrainingConfig(NeuralTrainingConfig):
    """Training controls for the baseline experimental VAE objective."""

    scaling: ScalingMode = "standard"
    beta: float = 1.0
    kl_warmup_epochs: int = 0

    def __post_init__(self) -> None:
        NeuralTrainingConfig.__post_init__(self)
        if self.beta < 0:
            raise ValueError("beta cannot be negative")
        if self.kl_warmup_epochs < 0:
            raise ValueError("kl_warmup_epochs cannot be negative")
