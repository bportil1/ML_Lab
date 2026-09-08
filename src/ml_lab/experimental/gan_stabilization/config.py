from __future__ import annotations

from dataclasses import dataclass

from ml_lab.generative.gan.config import GANTrainingConfig


@dataclass(frozen=True, slots=True)
class GANStabilizationConfig(GANTrainingConfig):
    """Experimental training mechanisms layered around the stable GAN models."""

    similarity_critic_enabled: bool = False
    similarity_embedding_dim: int = 32
    similarity_model_dim: int = 32
    similarity_heads: int = 4
    similarity_temperature: float = 0.1
    similarity_learning_rate: float = 2e-4
    real_view_noise_std: float = 0.01
    feature_matching_weight: float = 0.0

    attention_conditioning_strength: float = 0.0
    attention_conditioning_warmup_epochs: int = 0

    tangent_negative_weight: float = 0.0
    tangent_epsilon: float = 0.05
    tangent_samples: int = 1

    coupled_discriminator: bool = False
    transformer_blend_start: float = 0.0
    transformer_blend_end: float = 1.0
    transformer_blend_warmup_epochs: int = 0

    def __post_init__(self) -> None:
        GANTrainingConfig.__post_init__(self)
        if self.similarity_embedding_dim < 2 or self.similarity_model_dim < 2:
            raise ValueError("similarity dimensions must be at least 2")
        if self.similarity_heads < 1 or self.similarity_model_dim % self.similarity_heads != 0:
            raise ValueError("similarity_heads must evenly divide similarity_model_dim")
        if self.similarity_temperature <= 0:
            raise ValueError("similarity_temperature must be positive")
        if self.similarity_learning_rate <= 0:
            raise ValueError("similarity_learning_rate must be positive")
        if self.real_view_noise_std < 0:
            raise ValueError("real_view_noise_std cannot be negative")
        if self.feature_matching_weight < 0:
            raise ValueError("feature_matching_weight cannot be negative")
        if self.attention_conditioning_strength < 0:
            raise ValueError("attention_conditioning_strength cannot be negative")
        if self.attention_conditioning_warmup_epochs < 0:
            raise ValueError("attention_conditioning_warmup_epochs cannot be negative")
        if self.tangent_negative_weight < 0:
            raise ValueError("tangent_negative_weight cannot be negative")
        if self.tangent_epsilon <= 0:
            raise ValueError("tangent_epsilon must be positive")
        if self.tangent_samples < 1:
            raise ValueError("tangent_samples must be positive")
        for name, value in (
            ("transformer_blend_start", self.transformer_blend_start),
            ("transformer_blend_end", self.transformer_blend_end),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.transformer_blend_warmup_epochs < 0:
            raise ValueError("transformer_blend_warmup_epochs cannot be negative")
        if (self.attention_conditioning_strength > 0 or self.feature_matching_weight > 0) and not self.similarity_critic_enabled:
            raise ValueError("attention conditioning/feature matching require similarity_critic_enabled")
