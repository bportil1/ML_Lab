from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from ml_lab.neural import (
    BestModelCheckpoint,
    CallbackList,
    EarlyStopping,
    NeuralTrainer,
    NeuralTrainingResult,
    TrainingHistory,
    resolve_device,
    seed_everything,
    torch_generator,
)
from ml_lab.neural.backend import require_torch

from .config import GANModelConfig, GANTrainingConfig
from .data import make_scaler, validate_matrix
from .evaluation import distribution_metrics
from .models import build_gan_models
from .results import GANResult


def _torch_parts():
    torch = require_torch(purpose="GAN training")
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    return torch, nn, DataLoader, TensorDataset


class GANTrainer(NeuralTrainer):
    """Conservative BCE-with-logits GAN trainer using shared neural infrastructure."""

    def __init__(
        self,
        generator: Any,
        discriminator: Any,
        model_config: GANModelConfig,
        training_config: GANTrainingConfig,
        *,
        callbacks: Iterable[Any] | None = None,
    ) -> None:
        self.generator = generator
        self.discriminator = discriminator
        self.model_config = model_config
        self.config = training_config
        self.callbacks = CallbackList(callbacks)

    def fit(self, training_values: np.ndarray) -> NeuralTrainingResult:
        torch, nn, DataLoader, TensorDataset = _torch_parts()
        seed_everything(
            self.config.random_state,
            deterministic=self.config.deterministic,
            torch_module=torch,
        )
        device = resolve_device(self.config.device, torch_module=torch)
        generator = self.generator.to(device)
        discriminator = self.discriminator.to(device)

        class GANBundle(nn.Module):
            def __init__(self, generator: Any, discriminator: Any) -> None:
                super().__init__()
                self.generator = generator
                self.discriminator = discriminator

        bundle = GANBundle(generator, discriminator).to(device)
        generator_lr = self.config.generator_learning_rate or self.config.learning_rate
        discriminator_lr = self.config.discriminator_learning_rate or self.config.learning_rate
        optimizer_g = torch.optim.Adam(
            generator.parameters(),
            lr=generator_lr,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.weight_decay,
        )
        optimizer_d = torch.optim.Adam(
            discriminator.parameters(),
            lr=discriminator_lr,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss()

        tensor = torch.as_tensor(training_values, dtype=torch.float32)
        batch_size = min(self.config.batch_size, len(tensor))
        loader = DataLoader(
            TensorDataset(tensor),
            batch_size=batch_size,
            shuffle=True,
            generator=torch_generator(self.config.random_state, torch_module=torch),
        )
        history = TrainingHistory()
        early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta,
            mode="min",
        )
        checkpoint = BestModelCheckpoint(
            monitor=self.config.monitor,
            mode="min",
            min_delta=self.config.min_delta,
            path=self.config.checkpoint_path,
        )
        context: dict[str, Any] = {
            "generator": generator,
            "discriminator": discriminator,
            "config": self.config,
            "model_config": self.model_config,
            "device": device,
            "monitor": self.config.monitor,
        }
        self.callbacks.on_train_begin(context)

        stopped_epoch = self.config.epochs
        for epoch in range(1, self.config.epochs + 1):
            generator.train()
            discriminator.train()
            g_total = 0.0
            g_rows = 0
            d_total = 0.0
            d_rows = 0
            real_correct = 0
            fake_correct = 0
            d_examples = 0

            for (real_cpu,) in loader:
                real = real_cpu.to(device)
                current_batch = int(real.shape[0])

                for _ in range(self.config.discriminator_steps):
                    optimizer_d.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        z = torch.randn(current_batch, self.model_config.latent_dim, device=device)
                        fake = generator(z)
                    real_logits = discriminator(real)
                    fake_logits = discriminator(fake)
                    real_target = torch.full_like(
                        real_logits, 1.0 - self.config.real_label_smoothing
                    )
                    fake_target = torch.zeros_like(fake_logits)
                    d_loss = 0.5 * (
                        criterion(real_logits, real_target) + criterion(fake_logits, fake_target)
                    )
                    d_loss.backward()
                    if self.config.gradient_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            discriminator.parameters(), self.config.gradient_clip_norm
                        )
                    optimizer_d.step()
                    d_total += float(d_loss.detach().cpu()) * current_batch
                    d_rows += current_batch
                    real_correct += int((real_logits.detach() >= 0).sum().cpu())
                    fake_correct += int((fake_logits.detach() < 0).sum().cpu())
                    d_examples += current_batch

                for _ in range(self.config.generator_steps):
                    optimizer_g.zero_grad(set_to_none=True)
                    z = torch.randn(current_batch, self.model_config.latent_dim, device=device)
                    fake = generator(z)
                    logits = discriminator(fake)
                    target = torch.ones_like(logits)
                    g_loss = criterion(logits, target)
                    g_loss.backward()
                    if self.config.gradient_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            generator.parameters(), self.config.gradient_clip_norm
                        )
                    optimizer_g.step()
                    g_total += float(g_loss.detach().cpu()) * current_batch
                    g_rows += current_batch

            generator_loss = g_total / max(g_rows, 1)
            discriminator_loss = d_total / max(d_rows, 1)
            record = history.append(
                epoch,
                generator_loss=generator_loss,
                discriminator_loss=discriminator_loss,
                discriminator_real_accuracy=(real_correct / max(d_examples, 1)),
                discriminator_fake_accuracy=(fake_correct / max(d_examples, 1)),
            )
            monitored = (
                generator_loss
                if self.config.monitor == "generator_loss"
                else discriminator_loss
            )
            checkpoint.update(
                bundle,
                monitored,
                epoch,
                metadata={
                    "device": str(device),
                    "generator_type": self.model_config.generator_type,
                    "discriminator_type": self.model_config.discriminator_type,
                },
            )
            _, should_stop = early_stopping.update(monitored, epoch)
            self.callbacks.on_epoch_end(epoch, dict(record), context)
            if should_stop:
                stopped_epoch = epoch
                break

        if self.config.restore_best:
            checkpoint.restore(bundle)
        generator.eval()
        discriminator.eval()
        result = NeuralTrainingResult(
            model=bundle,
            history=history.to_records(),
            monitor=self.config.monitor,
            best_metric=checkpoint.best_value,
            best_epoch=checkpoint.best_epoch,
            stopped_epoch=stopped_epoch,
            device=str(device),
            checkpoint_path=(str(self.config.checkpoint_path) if self.config.checkpoint_path else None),
            metadata={
                "epochs_requested": self.config.epochs,
                "epochs_completed": len(history.records),
                "deterministic": self.config.deterministic,
                "restore_best": self.config.restore_best,
                "generator_steps": self.config.generator_steps,
                "discriminator_steps": self.config.discriminator_steps,
            },
        )
        context["result"] = result
        self.callbacks.on_train_end(context)
        return result


def train_gan(
    X: Any,
    *,
    model_config: GANModelConfig | None = None,
    training_config: GANTrainingConfig | None = None,
    callbacks: Iterable[Any] | None = None,
) -> GANResult:
    model_config = model_config or GANModelConfig()
    training_config = training_config or GANTrainingConfig()
    _, values, feature_names = validate_matrix(X)
    scaler = make_scaler(training_config.scaling)
    training_values = values if scaler is None else scaler.fit_transform(values).astype(np.float32)

    torch = require_torch(purpose="GAN training")
    seed_everything(
        training_config.random_state,
        deterministic=training_config.deterministic,
        torch_module=torch,
    )
    generator, discriminator = build_gan_models(values.shape[1], model_config)
    trainer = GANTrainer(
        generator,
        discriminator,
        model_config,
        training_config,
        callbacks=callbacks,
    )
    training_result = trainer.fit(training_values)
    bundle = training_result.model
    generator = bundle.generator
    discriminator = bundle.discriminator
    device = resolve_device(training_config.device, torch_module=torch)

    sample_count = training_config.generated_sample_count or len(values)
    seed_everything(
        training_config.random_state + 1,
        deterministic=training_config.deterministic,
        torch_module=torch,
    )
    with torch.no_grad():
        noise = torch.randn(sample_count, model_config.latent_dim, device=device)
        generated_training = generator(noise).detach().cpu().numpy()
        real_tensor = torch.as_tensor(training_values, dtype=torch.float32, device=device)
        generated_tensor = torch.as_tensor(generated_training, dtype=torch.float32, device=device)
        real_logits = discriminator(real_tensor)
        fake_logits = discriminator(generated_tensor)
        final_real_accuracy = float((real_logits >= 0).float().mean().detach().cpu())
        final_fake_accuracy = float((fake_logits < 0).float().mean().detach().cpu())

    generated = (
        generated_training
        if scaler is None
        else scaler.inverse_transform(generated_training)
    )
    generated = np.asarray(generated, dtype=float)
    metrics = distribution_metrics(values, generated)
    metrics["discriminator_real_accuracy_final"] = final_real_accuracy
    metrics["discriminator_fake_accuracy_final"] = final_fake_accuracy
    if training_result.history:
        metrics["generator_loss_final"] = float(training_result.history[-1]["generator_loss"])
        metrics["discriminator_loss_final"] = float(
            training_result.history[-1]["discriminator_loss"]
        )

    return GANResult(
        generated_samples=generated,
        metrics=metrics,
        feature_names=feature_names,
        generator=generator,
        discriminator=discriminator,
        transformer=scaler,
        history=training_result.history,
        metadata={
            "feature_dim": int(values.shape[1]),
            "training_rows": int(len(values)),
            "generated_sample_count": int(sample_count),
            "latent_dim": model_config.latent_dim,
            "generator_type": model_config.generator_type,
            "discriminator_type": model_config.discriminator_type,
            "generator_hidden_dims": list(model_config.generator_hidden_dims),
            "discriminator_hidden_dims": list(model_config.discriminator_hidden_dims),
            "output_activation": model_config.output_activation,
            "token_width": model_config.token_width,
            "model_dim": model_config.model_dim,
            "nhead": model_config.nhead,
            "transformer_layers": model_config.transformer_layers,
            "scaling": training_config.scaling,
            "device": training_result.device,
            "epochs_requested": training_config.epochs,
            "epochs_completed": len(training_result.history),
            "best_epoch": training_result.best_epoch,
            "best_monitored_value": training_result.best_metric,
            "monitor": training_result.monitor,
            "checkpoint_path": training_result.checkpoint_path,
            "restore_best": training_config.restore_best,
            "objective": "binary_cross_entropy_with_logits",
            "noise_distribution": "standard_normal",
        },
    )
