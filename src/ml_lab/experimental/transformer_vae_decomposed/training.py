from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from ml_lab.experimental.transformer_vae.model import build_transformer_vae
from ml_lab.experimental.transformer_vae.training import _masked_mse
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

from .config import DecomposedTransformerVAETrainingConfig, TransformerVAEConfig
from .objectives import DecomposedKLEstimate, estimate_decomposed_kl


@dataclass(slots=True)
class DecomposedTransformerVAETrainingResult:
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
            "task": "experimental_transformer_vae_decomposed",
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


def _warmup_factor(config: DecomposedTransformerVAETrainingConfig, epoch: int) -> float:
    if config.decomposition_warmup_epochs <= 0:
        return 1.0
    return min(1.0, float(epoch) / float(config.decomposition_warmup_epochs))


def _objective(
    reconstruction_loss: Any,
    decomposition: DecomposedKLEstimate,
    config: DecomposedTransformerVAETrainingConfig,
    *,
    warmup: float,
):
    regularization = decomposition.weighted(
        icmi_weight=config.icmi_weight,
        tc_weight=config.tc_weight,
        dwkl_weight=config.dwkl_weight,
    )
    return (
        float(config.reconstruction_weight) * reconstruction_loss
        + float(warmup) * regularization
    )


def _deterministic_latent_sample(torch: Any, mu: Any, logvar: Any, *, seed: int):
    """Sample q(z|x) without consuming PyTorch's training RNG stream."""
    rng = np.random.default_rng(seed)
    epsilon = torch.as_tensor(
        rng.standard_normal(tuple(mu.shape)).astype(np.float32),
        device=mu.device,
        dtype=mu.dtype,
    )
    return mu + torch.exp(0.5 * logvar) * epsilon


def _float_components(estimate: DecomposedKLEstimate) -> dict[str, float]:
    return {
        "icmi": float(estimate.icmi.detach().cpu()),
        "tc": float(estimate.total_correlation.detach().cpu()),
        "dwkl": float(estimate.dimension_wise_kl.detach().cpu()),
        "decomposed_kl": float(estimate.estimated_kl.detach().cpu()),
        "analytic_kl": float(estimate.analytic_kl.detach().cpu()),
    }


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


def train_decomposed_transformer_vae(
    X: Any,
    *,
    model_config: TransformerVAEConfig | None = None,
    training_config: DecomposedTransformerVAETrainingConfig | None = None,
    callbacks: Iterable[Any] | None = None,
    prior_sample_count: int = 0,
) -> DecomposedTransformerVAETrainingResult:
    model_config = model_config or TransformerVAEConfig()
    training_config = training_config or DecomposedTransformerVAETrainingConfig()
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

    torch = require_torch(purpose="experimental decomposed Transformer VAE training")
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
    if len(split.train_indices) < 2:
        raise ValueError("decomposed VAE training requires at least two training samples")

    device = resolve_device(training_config.device, torch_module=torch)
    model = build_transformer_vae(prepared.token_dim, model_config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    train_tokens = torch.as_tensor(prepared.tokens[split.train_indices], dtype=torch.float32)
    train_mask = torch.as_tensor(prepared.element_mask[split.train_indices], dtype=torch.bool)
    effective_batch_size = min(training_config.batch_size, len(train_tokens))
    if effective_batch_size < 2:
        raise ValueError("decomposed KL estimation requires batches of at least two samples")
    # A singleton final minibatch cannot estimate an aggregated posterior. Drop only
    # that degenerate remainder; all other minibatches remain in the epoch.
    drop_last = len(train_tokens) > effective_batch_size and len(train_tokens) % effective_batch_size == 1
    loader = DataLoader(
        TensorDataset(train_tokens, train_mask),
        batch_size=effective_batch_size,
        shuffle=True,
        drop_last=drop_last,
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
        warmup = _warmup_factor(training_config, epoch)
        model.train()
        sums = {
            "reconstruction": 0.0,
            "icmi": 0.0,
            "tc": 0.0,
            "dwkl": 0.0,
            "decomposed_kl": 0.0,
            "analytic_kl": 0.0,
        }
        sample_count = 0
        valid_elements = 0

        for batch_tokens, batch_mask in loader:
            batch_tokens = batch_tokens.to(device)
            batch_mask = batch_mask.to(device)
            if int(batch_tokens.shape[0]) < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            reconstruction, mu, logvar, z = model(
                batch_tokens,
                element_mask=batch_mask,
                sample=True,
            )
            reconstruction_loss = _masked_mse(torch, reconstruction, batch_tokens, batch_mask)
            decomposition = estimate_decomposed_kl(torch, z, mu, logvar)
            loss = _objective(
                reconstruction_loss,
                decomposition,
                training_config,
                warmup=warmup,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("decomposed Transformer VAE produced a non-finite training loss")
            loss.backward()
            optimizer.step()

            batch_samples = int(batch_tokens.shape[0])
            batch_valid_elements = int(batch_mask.sum().detach().cpu())
            components = _float_components(decomposition)
            sums["reconstruction"] += float(reconstruction_loss.detach().cpu()) * batch_valid_elements
            for name in ("icmi", "tc", "dwkl", "decomposed_kl", "analytic_kl"):
                sums[name] += components[name] * batch_samples
            valid_elements += batch_valid_elements
            sample_count += batch_samples

        if sample_count < 2:
            raise RuntimeError("no valid minibatch was available for decomposed KL estimation")
        train = {
            "reconstruction": sums["reconstruction"] / max(valid_elements, 1),
            **{
                name: sums[name] / sample_count
                for name in ("icmi", "tc", "dwkl", "decomposed_kl", "analytic_kl")
            },
        }
        train_loss = (
            training_config.reconstruction_weight * train["reconstruction"]
            + warmup
            * (
                training_config.icmi_weight * train["icmi"]
                + training_config.tc_weight * train["tc"]
                + training_config.dwkl_weight * train["dwkl"]
            )
        )

        validation: dict[str, float] | None = None
        validation_loss: float | None = None
        monitored_loss = train_loss
        if validation_tokens is not None and validation_mask is not None:
            model.eval()
            with torch.no_grad():
                reconstruction, mu, logvar, _ = model(
                    validation_tokens,
                    element_mask=validation_mask,
                    sample=False,
                )
                reconstruction_loss = _masked_mse(
                    torch, reconstruction, validation_tokens, validation_mask
                )
                if int(mu.shape[0]) >= 2:
                    z = _deterministic_latent_sample(
                        torch,
                        mu,
                        logvar,
                        seed=training_config.random_state + 100_000 + epoch,
                    )
                    decomposition = estimate_decomposed_kl(torch, z, mu, logvar)
                    validation = {
                        "reconstruction": float(reconstruction_loss.detach().cpu()),
                        **_float_components(decomposition),
                    }
                    validation_loss = (
                        training_config.reconstruction_weight * validation["reconstruction"]
                        + warmup
                        * (
                            training_config.icmi_weight * validation["icmi"]
                            + training_config.tc_weight * validation["tc"]
                            + training_config.dwkl_weight * validation["dwkl"]
                        )
                    )
                else:
                    # The decomposed density estimate is undefined for one validation row.
                    # Use reconstruction only for model selection rather than inventing a value.
                    validation = {
                        "reconstruction": float(reconstruction_loss.detach().cpu()),
                        "icmi": float("nan"),
                        "tc": float("nan"),
                        "dwkl": float("nan"),
                        "decomposed_kl": float("nan"),
                        "analytic_kl": float(
                            -0.5
                            * torch.mean(
                                torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                            ).detach().cpu()
                        ),
                    }
                    validation_loss = validation["reconstruction"]
            monitored_loss = validation_loss

        record = history.append(
            epoch,
            warmup_factor=warmup,
            icmi_weight=training_config.icmi_weight,
            tc_weight=training_config.tc_weight,
            dwkl_weight=training_config.dwkl_weight,
            train_loss=train_loss,
            train_reconstruction_mse=train["reconstruction"],
            train_icmi=train["icmi"],
            train_tc=train["tc"],
            train_dwkl=train["dwkl"],
            train_decomposed_kl=train["decomposed_kl"],
            train_analytic_kl=train["analytic_kl"],
            validation_loss=validation_loss,
            validation_reconstruction_mse=(validation["reconstruction"] if validation else None),
            validation_icmi=(validation["icmi"] if validation else None),
            validation_tc=(validation["tc"] if validation else None),
            validation_dwkl=(validation["dwkl"] if validation else None),
            validation_decomposed_kl=(validation["decomposed_kl"] if validation else None),
            validation_analytic_kl=(validation["analytic_kl"] if validation else None),
        )
        checkpoint.update(
            model,
            monitored_loss,
            epoch,
            optimizer=optimizer,
            metadata={
                "device": str(device),
                "warmup_factor": warmup,
                "icmi_weight": training_config.icmi_weight,
                "tc_weight": training_config.tc_weight,
                "dwkl_weight": training_config.dwkl_weight,
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
        diagnostic_z = _deterministic_latent_sample(
            torch,
            posterior_mean,
            posterior_logvar,
            seed=training_config.random_state + 900_001,
        )
        final_decomposition = estimate_decomposed_kl(
            torch,
            diagnostic_z,
            posterior_mean,
            posterior_logvar,
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
    final_components = _float_components(final_decomposition)
    metrics.update(
        {
            "icmi_estimate": final_components["icmi"],
            "total_correlation_estimate": final_components["tc"],
            "dimension_wise_kl_estimate": final_components["dwkl"],
            "decomposed_kl_estimate": final_components["decomposed_kl"],
            "analytic_kl": final_components["analytic_kl"],
        }
    )

    prior_samples = _restore_prior_samples(
        model,
        prepared,
        count=prior_sample_count,
        device=device,
        torch=torch,
    )
    result = DecomposedTransformerVAETrainingResult(
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
            "model_dim": model_config.model_dim,
            "nhead": model_config.nhead,
            "encoder_layers": model_config.encoder_layers,
            "decoder_layers": model_config.decoder_layers,
            "objective": "reconstruction + icmi + tc + dwkl",
            "density_estimator": "minibatch_mixture",
            "density_estimator_note": (
                "q(z) and q(z_j) are minibatch mixture estimates; values depend on batch composition"
            ),
            "reconstruction_weight": training_config.reconstruction_weight,
            "icmi_weight": training_config.icmi_weight,
            "tc_weight": training_config.tc_weight,
            "dwkl_weight": training_config.dwkl_weight,
            "decomposition_warmup_epochs": training_config.decomposition_warmup_epochs,
            "final_diagnostic_seed": training_config.random_state + 900_001,
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
    "DecomposedTransformerVAETrainingResult",
    "train_decomposed_transformer_vae",
]
