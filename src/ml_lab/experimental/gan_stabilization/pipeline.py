from __future__ import annotations

from typing import Any

import numpy as np


def run(*, mode: str = "describe", X: Any = None, model_config: dict[str, Any] | None = None, training_config: dict[str, Any] | None = None) -> dict[str, Any]:
    if mode == "describe":
        return {
            "experiment": "gan_stabilization",
            "status": "experimental",
            "available_modes": ["describe", "fit_generate"],
            "mechanisms": [
                "self_supervised_attention_similarity_critic",
                "critic_context_generator_conditioning",
                "score_tangent_fake_negatives",
                "scheduled_mlp_transformer_discriminator_blend",
                "critic_embedding_distribution_feature_matching",
            ],
            "warnings": [
                "All mechanisms are opt-in research training schemes; stable GAN model definitions are reused unchanged.",
                "The similarity critic is trained on two noisy views of real samples rather than arbitrary fake/real positive pairs.",
                "Tangent negatives are first-order tangent to discriminator-score level sets, not guaranteed data-manifold tangents.",
            ],
        }
    if mode != "fit_generate":
        raise ValueError(f"unknown gan_stabilization mode: {mode}")
    if X is None:
        raise ValueError("X is required for fit_generate mode")

    from ml_lab.generative.gan import GANModelConfig
    from .config import GANStabilizationConfig
    from .training import train_stabilized_gan

    result = train_stabilized_gan(
        np.asarray(X, dtype=np.float32),
        model_config=GANModelConfig(**(model_config or {})),
        training_config=GANStabilizationConfig(**(training_config or {})),
    )
    record = result.to_record()
    record["generated_samples"] = result.generated_samples.tolist()
    return record
