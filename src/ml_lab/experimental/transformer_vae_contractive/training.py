from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from ml_lab.experimental.transformer_vae.model import build_transformer_vae
from ml_lab.experimental.transformer_vae.training import _kl_divergence, _masked_mse
from ml_lab.neural.backend import require_torch
from ml_lab.neural.callbacks import CallbackList
from ml_lab.neural.checkpoint import BestModelCheckpoint
from ml_lab.neural.data import split_validation_indices
from ml_lab.neural.history import TrainingHistory
from ml_lab.neural.runtime import resolve_device, seed_everything, torch_generator
from ml_lab.neural.selection import EarlyStopping
from ml_lab.representation.evaluation import reconstruction_metrics
from ml_lab.representation.transformer_data import (
    TransformerPreparedData,
    prepare_transformer_data,
    restore_transformer_reconstruction,
)

from .config import ContractiveTransformerVAETrainingConfig, TransformerVAEConfig
from .objectives import contractive_jacobian_penalty


@dataclass(slots=True)
class ContractiveTransformerVAETrainingResult:
    model: Any
    posterior_mean: np.ndarray
    posterior_logvar: np.ndarray
    reconstruction: np.ndarray
    metrics: dict[str, float]
    history: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)
    scaler: Any = None
    prior_samples: np.ndarray | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "task": "experimental_transformer_vae_contractive",
            "posterior_mean_shape": list(self.posterior_mean.shape),
            "posterior_logvar_shape": list(self.posterior_logvar.shape),
            "reconstruction_shape": list(self.reconstruction.shape),
            "prior_samples_shape": (
                list(self.prior_samples.shape) if self.prior_samples is not None else None
            ),
            "metrics": dict(self.metrics),
            "history": [dict(record) for record in self.history],
            "metadata": dict(self.metadata),
        }


def _warmup_factor(config: ContractiveTransformerVAETrainingConfig, epoch: int) -> float:
    if config.contractive_warmup_epochs <= 0:
        return 1.0
    return min(1.0, float(epoch) / float(config.contractive_warmup_epochs))


def _beta_for_epoch(config: ContractiveTransformerVAETrainingConfig, epoch: int) -> float:
    if config.kl_warmup_epochs <= 0:
        return float(config.beta)
    return float(config.beta) * min(1.0, float(epoch) / float(config.kl_warmup_epochs))


def _second_order_attention_context():
    """Use the math SDPA kernel for Transformer double-backward support.

    Optimized flash/memory-efficient scaled-dot-product attention kernels do not
    consistently implement the second derivatives required by a contractive
    Jacobian loss. Restricting only this encoder pass to the math backend keeps
    the experiment mathematically differentiable without changing other models.
    """
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel(SDPBackend.MATH)
    except (ImportError, AttributeError):
        return nullcontext()


def _contractive(
    torch: Any,
    model: Any,
    tokens: Any,
    mask: Any,
    config: ContractiveTransformerVAETrainingConfig,
    *,
    create_graph: bool,
):
    inputs = tokens.detach().clone().requires_grad_(True)
    with _second_order_attention_context():
        mu, logvar = model.encode_distribution(inputs, element_mask=mask)
        penalty = contractive_jacobian_penalty(
            torch,
            mu,
            inputs,
            element_mask=mask,
            estimator=config.contractive_estimator,
            hutchinson_samples=config.hutchinson_samples,
            normalize_by_active_inputs=config.normalize_by_active_inputs,
            create_graph=create_graph,
        )
    return inputs, mu, logvar, penalty


def _restore_prior_samples(
    model: Any,
    prepared: TransformerPreparedData,
    *,
    count: int,
    device: Any,
    torch: Any,
) -> np.ndarray | None:
    if count <= 0:
        return None
    prior_mask = torch.as_tensor(
        np.repeat(prepared.element_mask[:1], count, axis=0),
        dtype=torch.bool,
        device=device,
    )
    with torch.no_grad():
        prior_tokens = model.sample_prior(
            count,
            token_count=prepared.token_count,
            element_mask=prior_mask,
            device=device,
        ).detach().cpu().numpy()
    synthetic = TransformerPreparedData(
        original=np.zeros((count, *prepared.input_shape[1:]), dtype=np.float32),
        tokens=np.zeros((count, prepared.token_count, prepared.token_dim), dtype=np.float32),
        element_mask=np.repeat(prepared.element_mask[:1], count, axis=0),
        feature_names=list(prepared.feature_names),
        input_shape=(count, *prepared.input_shape[1:]),
        scaler=prepared.scaler,
        token_width=prepared.token_width,
    )
    return restore_transformer_reconstruction(synthetic, prior_tokens)


def _final_contractive_penalty(
    torch: Any,
    model: Any,
    tokens: Any,
    mask: Any,
    config: ContractiveTransformerVAETrainingConfig,
) -> float:
    batch_size = min(config.batch_size, int(tokens.shape[0]))
    weighted = 0.0
    count = 0
    for start in range(0, int(tokens.shape[0]), batch_size):
        batch_tokens = tokens[start : start + batch_size]
        batch_mask = mask[start : start + batch_size]
        _, _, _, penalty = _contractive(
            torch,
            model,
            batch_tokens,
            batch_mask,
            config,
            create_graph=False,
        )
        rows = int(batch_tokens.shape[0])
        weighted += float(penalty.detach().cpu()) * rows
        count += rows
    return weighted / max(count, 1)


def train_contractive_transformer_vae(
    X: Any,
    *,
    model_config: TransformerVAEConfig | None = None,
    training_config: ContractiveTransformerVAETrainingConfig | None = None,
    callbacks: Iterable[Any] | None = None,
    prior_sample_count: int = 0,
) -> ContractiveTransformerVAETrainingResult:
    model_config = model_config or TransformerVAEConfig()
    training_config = training_config or ContractiveTransformerVAETrainingConfig()
    if prior_sample_count < 0:
        raise ValueError("prior_sample_count cannot be negative")

    prepared = prepare_transformer_data(
        X,
        scaling=training_config.scaling,
        token_width=model_config.token_width,
    )
    if prepared.token_count > model_config.max_tokens:
        raise ValueError(
            f"prepared input has {prepared.token_count} tokens but max_tokens="
            f"{model_config.max_tokens}; increase max_tokens or token_width"
        )

    torch = require_torch(purpose="experimental contractive Transformer VAE training")
    from torch.utils.data import DataLoader, TensorDataset

    seed_everything(
        training_config.random_state,
        deterministic=training_config.deterministic,
        torch_module=torch,
    )
    split = split_validation_indices(
        len(prepared.tokens),
        training_config.validation_fraction,
        random_state=training_config.random_state,
    )
    device = resolve_device(training_config.device, torch_module=torch)
    model = build_transformer_vae(prepared.token_dim, model_config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    train_tokens = torch.as_tensor(prepared.tokens[split.train_indices], dtype=torch.float32)
    train_mask = torch.as_tensor(prepared.element_mask[split.train_indices], dtype=torch.bool)
    loader = DataLoader(
        TensorDataset(train_tokens, train_mask),
        batch_size=min(training_config.batch_size, len(train_tokens)),
        shuffle=True,
        generator=torch_generator(training_config.random_state, torch_module=torch),
    )

    validation_tokens = None
    validation_mask = None
    if split.validation_count:
        validation_tokens = torch.as_tensor(
            prepared.tokens[split.validation_indices], dtype=torch.float32, device=device
        )
        validation_mask = torch.as_tensor(
            prepared.element_mask[split.validation_indices], dtype=torch.bool, device=device
        )

    monitor = "validation_loss" if validation_tokens is not None else "train_loss"
    history = TrainingHistory()
    early_stopping = EarlyStopping(
        patience=training_config.patience,
        min_delta=training_config.min_delta,
        mode="min",
    )
    checkpoint = BestModelCheckpoint(
        monitor=monitor,
        mode="min",
        min_delta=training_config.min_delta,
        path=training_config.checkpoint_path,
    )
    callback_list = CallbackList(callbacks)
    context: dict[str, Any] = {
        "model": model,
        "optimizer": optimizer,
        "config": training_config,
        "device": device,
        "monitor": monitor,
    }
    callback_list.on_train_begin(context)

    stopped_epoch = training_config.epochs
    for epoch in range(1, training_config.epochs + 1):
        beta = _beta_for_epoch(training_config, epoch)
        contractive_factor = _warmup_factor(training_config, epoch)
        model.train()
        weighted_reconstruction = 0.0
        weighted_kl = 0.0
        weighted_contractive = 0.0
        sample_count = 0
        valid_elements = 0

        for batch_tokens, batch_mask in loader:
            batch_tokens = batch_tokens.to(device)
            batch_mask = batch_mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            inputs, mu, logvar, contractive_loss = _contractive(
                torch,
                model,
                batch_tokens,
                batch_mask,
                training_config,
                create_graph=True,
            )
            z = model.reparameterize(mu, logvar)
            reconstruction = model.decode(
                z,
                token_count=int(inputs.shape[1]),
                element_mask=batch_mask,
            )
            reconstruction_loss = _masked_mse(torch, reconstruction, inputs, batch_mask)
            kl_loss = _kl_divergence(torch, mu, logvar)
            loss = (
                reconstruction_loss
                + beta * kl_loss
                + float(training_config.contractive_weight)
                * contractive_factor
                * contractive_loss
            )
            loss.backward()
            optimizer.step()

            rows = int(inputs.shape[0])
            active = int(batch_mask.sum().detach().cpu())
            weighted_reconstruction += float(reconstruction_loss.detach().cpu()) * active
            weighted_kl += float(kl_loss.detach().cpu()) * rows
            weighted_contractive += float(contractive_loss.detach().cpu()) * rows
            valid_elements += active
            sample_count += rows

        train_reconstruction = weighted_reconstruction / max(valid_elements, 1)
        train_kl = weighted_kl / max(sample_count, 1)
        train_contractive = weighted_contractive / max(sample_count, 1)
        train_loss = (
            train_reconstruction
            + beta * train_kl
            + float(training_config.contractive_weight)
            * contractive_factor
            * train_contractive
        )

        validation_loss = None
        validation_reconstruction = None
        validation_kl = None
        validation_contractive = None
        monitored_loss = train_loss
        if validation_tokens is not None and validation_mask is not None:
            model.eval()
            inputs, mu, logvar, contractive_loss = _contractive(
                torch,
                model,
                validation_tokens,
                validation_mask,
                training_config,
                create_graph=False,
            )
            with torch.no_grad():
                reconstruction = model.decode(
                    mu,
                    token_count=int(inputs.shape[1]),
                    element_mask=validation_mask,
                )
                validation_reconstruction = float(
                    _masked_mse(torch, reconstruction, inputs, validation_mask).detach().cpu()
                )
                validation_kl = float(_kl_divergence(torch, mu, logvar).detach().cpu())
            validation_contractive = float(contractive_loss.detach().cpu())
            validation_loss = (
                validation_reconstruction
                + beta * validation_kl
                + float(training_config.contractive_weight)
                * contractive_factor
                * validation_contractive
            )
            monitored_loss = validation_loss

        record = history.append(
            epoch,
            beta=beta,
            contractive_factor=contractive_factor,
            contractive_weight=training_config.contractive_weight,
            train_loss=train_loss,
            train_reconstruction_mse=train_reconstruction,
            train_kl=train_kl,
            train_contractive_penalty=train_contractive,
            validation_loss=validation_loss,
            validation_reconstruction_mse=validation_reconstruction,
            validation_kl=validation_kl,
            validation_contractive_penalty=validation_contractive,
        )
        checkpoint.update(
            model,
            monitored_loss,
            epoch,
            optimizer=optimizer,
            metadata={
                "device": str(device),
                "beta": beta,
                "contractive_factor": contractive_factor,
            },
        )
        _, should_stop = early_stopping.update(monitored_loss, epoch)
        callback_list.on_epoch_end(epoch, dict(record), context)
        if should_stop:
            stopped_epoch = epoch
            break

    if training_config.restore_best:
        checkpoint.restore(model)
    model.eval()

    full_tokens = torch.as_tensor(prepared.tokens, dtype=torch.float32, device=device)
    full_mask = torch.as_tensor(prepared.element_mask, dtype=torch.bool, device=device)
    with torch.no_grad():
        posterior_mean, posterior_logvar = model.encode_distribution(
            full_tokens,
            element_mask=full_mask,
        )
        reconstructed_tokens = model.decode(
            posterior_mean,
            token_count=prepared.token_count,
            element_mask=full_mask,
        )
    posterior_mean_np = posterior_mean.detach().cpu().numpy()
    posterior_logvar_np = posterior_logvar.detach().cpu().numpy()
    reconstructed_tokens_np = reconstructed_tokens.detach().cpu().numpy()
    reconstruction = restore_transformer_reconstruction(prepared, reconstructed_tokens_np)
    metrics = reconstruction_metrics(prepared.original, reconstruction)
    token_error = (prepared.tokens - reconstructed_tokens_np) ** 2
    mask_np = prepared.element_mask.astype(np.float32)
    metrics["training_space_mse"] = float(
        (token_error * mask_np).sum() / max(mask_np.sum(), 1.0)
    )
    metrics["mean_posterior_kl"] = float(
        -0.5
        * np.mean(
            np.sum(
                1.0
                + posterior_logvar_np
                - np.square(posterior_mean_np)
                - np.exp(posterior_logvar_np),
                axis=1,
            )
        )
    )
    metrics["mean_contractive_penalty"] = _final_contractive_penalty(
        torch,
        model,
        full_tokens,
        full_mask,
        training_config,
    )

    prior_samples = _restore_prior_samples(
        model,
        prepared,
        count=prior_sample_count,
        device=device,
        torch=torch,
    )

    result = ContractiveTransformerVAETrainingResult(
        model=model,
        posterior_mean=np.asarray(posterior_mean_np, dtype=float),
        posterior_logvar=np.asarray(posterior_logvar_np, dtype=float),
        reconstruction=np.asarray(reconstruction, dtype=float),
        metrics=metrics,
        history=history.to_records(),
        scaler=prepared.scaler,
        prior_samples=(np.asarray(prior_samples, dtype=float) if prior_samples is not None else None),
        metadata={
            "input_shape": list(prepared.input_shape),
            "token_count": prepared.token_count,
            "token_dim": prepared.token_dim,
            "latent_dim": model_config.latent_dim,
            "beta": training_config.beta,
            "kl_warmup_epochs": training_config.kl_warmup_epochs,
            "contractive_weight": training_config.contractive_weight,
            "contractive_warmup_epochs": training_config.contractive_warmup_epochs,
            "contractive_estimator": training_config.contractive_estimator,
            "hutchinson_samples": training_config.hutchinson_samples,
            "normalize_by_active_inputs": training_config.normalize_by_active_inputs,
            "contractive_target": "posterior_mean_jacobian_wrt_input",
            "device": str(device),
            "epochs_requested": training_config.epochs,
            "epochs_completed": len(history.records),
            "stopped_epoch": stopped_epoch,
            "best_epoch": checkpoint.best_epoch,
            "best_loss": checkpoint.best_value,
            "monitor": monitor,
            "validation_rows": split.validation_count,
            "deterministic": training_config.deterministic,
            "checkpoint_path": training_config.checkpoint_path,
            "restore_best": training_config.restore_best,
        },
    )
    context["result"] = result
    callback_list.on_train_end(context)
    return result


__all__ = [
    "ContractiveTransformerVAETrainingResult",
    "train_contractive_transformer_vae",
]
