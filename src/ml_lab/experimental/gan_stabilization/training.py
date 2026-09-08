from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ml_lab.generative.gan.config import GANModelConfig
from ml_lab.generative.gan.data import make_scaler, validate_matrix
from ml_lab.generative.gan.evaluation import distribution_metrics
from ml_lab.generative.gan.models import build_discriminator, build_generator
from ml_lab.neural import resolve_device, seed_everything, torch_generator
from ml_lab.neural.backend import require_torch

from .config import GANStabilizationConfig
from .models import build_attention_similarity_critic, build_coupled_discriminator, build_latent_conditioner
from .objectives import embedding_moment_loss, symmetric_infonce
from .tangent import score_tangent_negatives


@dataclass(slots=True)
class GANStabilizationResult:
    generated_samples: np.ndarray
    metrics: dict[str, float]
    history: list[dict[str, float | int]]
    metadata: dict[str, Any]
    generator: Any
    discriminator: Any
    similarity_critic: Any | None = None
    conditioner: Any | None = None
    transformer: Any | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "task": "experimental_gan_stabilization",
            "generated_shape": list(self.generated_samples.shape),
            "metrics": self.metrics,
            "history": self.history,
            "metadata": self.metadata,
        }


def _schedule(target: float, epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return float(target)
    return float(target) * min(1.0, max(0.0, epoch / warmup_epochs))


def _blend(config: GANStabilizationConfig, epoch: int) -> float:
    if config.transformer_blend_warmup_epochs <= 0:
        return float(config.transformer_blend_end)
    frac = min(1.0, max(0.0, epoch / config.transformer_blend_warmup_epochs))
    return float(config.transformer_blend_start + frac * (config.transformer_blend_end - config.transformer_blend_start))


def _set_requires_grad(module: Any, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def train_stabilized_gan(
    X: Any,
    *,
    model_config: GANModelConfig | None = None,
    training_config: GANStabilizationConfig | None = None,
) -> GANStabilizationResult:
    model_config = model_config or GANModelConfig()
    config = training_config or GANStabilizationConfig()
    _, values, feature_names = validate_matrix(X)
    if config.similarity_critic_enabled and len(values) < 2:
        raise ValueError("similarity critic requires at least two training rows")
    scaler = make_scaler(config.scaling)
    training_values = values if scaler is None else scaler.fit_transform(values).astype(np.float32)

    torch = require_torch(purpose="experimental GAN stabilization")
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    seed_everything(config.random_state, deterministic=config.deterministic, torch_module=torch)
    device = resolve_device(config.device, torch_module=torch)
    generator = build_generator(values.shape[1], model_config).to(device)
    discriminator = (
        build_coupled_discriminator(values.shape[1], model_config)
        if config.coupled_discriminator
        else build_discriminator(values.shape[1], model_config)
    ).to(device)

    critic = None
    conditioner = None
    if config.similarity_critic_enabled:
        critic = build_attention_similarity_critic(
            values.shape[1],
            token_width=model_config.token_width,
            model_dim=config.similarity_model_dim,
            heads=config.similarity_heads,
            embedding_dim=config.similarity_embedding_dim,
            max_tokens=model_config.max_tokens,
        ).to(device)
        if config.attention_conditioning_strength > 0:
            conditioner = build_latent_conditioner(config.similarity_model_dim, model_config.latent_dim).to(device)

    opt_g_params = list(generator.parameters()) + (list(conditioner.parameters()) if conditioner is not None else [])
    opt_g = torch.optim.Adam(opt_g_params, lr=config.generator_learning_rate or config.learning_rate, betas=(config.adam_beta1, config.adam_beta2), weight_decay=config.weight_decay)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=config.discriminator_learning_rate or config.learning_rate, betas=(config.adam_beta1, config.adam_beta2), weight_decay=config.weight_decay)
    opt_c = None if critic is None else torch.optim.Adam(critic.parameters(), lr=config.similarity_learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    tensor = torch.as_tensor(training_values, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor), batch_size=min(config.batch_size, len(tensor)), shuffle=True, generator=torch_generator(config.random_state, torch_module=torch), drop_last=(len(tensor) > 1 and min(config.batch_size, len(tensor)) > 1))
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        generator.train(); discriminator.train()
        if critic is not None: critic.train()
        if conditioner is not None: conditioner.train()
        blend = _blend(config, epoch) if config.coupled_discriminator else 0.0
        if config.coupled_discriminator:
            discriminator.set_blend(blend)
        conditioning_strength = _schedule(config.attention_conditioning_strength, epoch, config.attention_conditioning_warmup_epochs)
        totals = {"g": 0.0, "d": 0.0, "critic": 0.0, "feature": 0.0, "tangent": 0.0, "rows": 0}

        for (real_cpu,) in loader:
            real = real_cpu.to(device)
            batch = int(real.shape[0])
            if batch < 2 and critic is not None:
                continue
            context = None
            real_embedding = None
            if critic is not None:
                opt_c.zero_grad(set_to_none=True)
                view1 = real + config.real_view_noise_std * torch.randn_like(real)
                view2 = real + config.real_view_noise_std * torch.randn_like(real)
                emb1 = critic(view1)
                emb2 = critic(view2)
                critic_loss = symmetric_infonce(emb1, emb2, temperature=config.similarity_temperature)
                critic_loss.backward(); opt_c.step()
                with torch.no_grad():
                    real_embedding, pooled, _ = critic.encode(real, return_attention=True)
                    context = pooled.mean(dim=0, keepdim=True)
                totals["critic"] += float(critic_loss.detach().cpu()) * batch

            for _ in range(config.discriminator_steps):
                opt_d.zero_grad(set_to_none=True)
                with torch.no_grad():
                    z = torch.randn(batch, model_config.latent_dim, device=device)
                    if conditioner is not None and context is not None:
                        z = conditioner(z, context, strength=conditioning_strength)
                    fake = generator(z)
                real_logits = discriminator(real)
                fake_logits = discriminator(fake)
                real_target = torch.full_like(real_logits, 1.0 - config.real_label_smoothing)
                d_loss = 0.5 * (criterion(real_logits, real_target) + criterion(fake_logits, torch.zeros_like(fake_logits)))
                tangent_loss = torch.zeros((), device=device)
                if config.tangent_negative_weight > 0:
                    tangents = score_tangent_negatives(discriminator, fake, epsilon=config.tangent_epsilon, count=config.tangent_samples)
                    tangent_logits = discriminator(tangents)
                    tangent_loss = criterion(tangent_logits, torch.zeros_like(tangent_logits))
                    d_loss = d_loss + config.tangent_negative_weight * tangent_loss
                d_loss.backward(); opt_d.step()
                totals["d"] += float(d_loss.detach().cpu()) * batch
                totals["tangent"] += float(tangent_loss.detach().cpu()) * batch

            for _ in range(config.generator_steps):
                opt_g.zero_grad(set_to_none=True)
                z = torch.randn(batch, model_config.latent_dim, device=device)
                if conditioner is not None and context is not None:
                    z = conditioner(z, context, strength=conditioning_strength)
                fake = generator(z)
                adv_loss = criterion(discriminator(fake), torch.ones(batch, device=device))
                feature_loss = torch.zeros((), device=device)
                if critic is not None and config.feature_matching_weight > 0 and real_embedding is not None:
                    _set_requires_grad(critic, False)
                    try:
                        fake_embedding = critic(fake)
                        feature_loss = embedding_moment_loss(fake_embedding, real_embedding.detach())
                    finally:
                        _set_requires_grad(critic, True)
                g_loss = adv_loss + config.feature_matching_weight * feature_loss
                g_loss.backward(); opt_g.step()
                totals["g"] += float(g_loss.detach().cpu()) * batch
                totals["feature"] += float(feature_loss.detach().cpu()) * batch
            totals["rows"] += batch

        rows = max(int(totals["rows"]), 1)
        history.append({
            "epoch": epoch,
            "generator_loss": totals["g"] / rows,
            "discriminator_loss": totals["d"] / rows,
            "similarity_critic_loss": totals["critic"] / rows,
            "feature_matching_loss": totals["feature"] / rows,
            "tangent_negative_loss": totals["tangent"] / rows,
            "attention_conditioning_strength": conditioning_strength,
            "transformer_discriminator_blend": blend,
        })

    generator.eval(); discriminator.eval()
    if critic is not None: critic.eval()
    if conditioner is not None: conditioner.eval()
    count = config.generated_sample_count or len(values)
    seed_everything(config.random_state + 1, deterministic=config.deterministic, torch_module=torch)
    with torch.no_grad():
        z = torch.randn(count, model_config.latent_dim, device=device)
        if conditioner is not None and critic is not None:
            real_tensor = torch.as_tensor(training_values, dtype=torch.float32, device=device)
            _, pooled, _ = critic.encode(real_tensor, return_attention=True)
            z = conditioner(z, pooled.mean(dim=0, keepdim=True), strength=config.attention_conditioning_strength)
        generated_training = generator(z).cpu().numpy()
    generated = generated_training if scaler is None else scaler.inverse_transform(generated_training)
    generated = np.asarray(generated, dtype=float)
    metrics = distribution_metrics(values, generated)
    if history:
        metrics.update({f"{key}_final": float(value) for key, value in history[-1].items() if key != "epoch"})

    return GANStabilizationResult(
        generated_samples=generated,
        metrics=metrics,
        history=history,
        metadata={
            "feature_names": feature_names,
            "feature_dim": int(values.shape[1]),
            "latent_dim": model_config.latent_dim,
            "generator_type": model_config.generator_type,
            "base_discriminator_type": model_config.discriminator_type,
            "similarity_critic_enabled": config.similarity_critic_enabled,
            "feature_matching_weight": config.feature_matching_weight,
            "attention_conditioning_strength": config.attention_conditioning_strength,
            "tangent_negative_weight": config.tangent_negative_weight,
            "tangent_definition": "random perturbation projected orthogonal to local discriminator-score gradient",
            "coupled_discriminator": config.coupled_discriminator,
            "transformer_blend_start": config.transformer_blend_start,
            "transformer_blend_end": config.transformer_blend_end,
            "status": "experimental",
        },
        generator=generator,
        discriminator=discriminator,
        similarity_critic=critic,
        conditioner=conditioner,
        transformer=scaler,
    )
