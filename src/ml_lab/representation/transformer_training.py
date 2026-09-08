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
    split_validation_indices,
    torch_generator,
)
from ml_lab.neural.backend import require_torch

from .config import AutoencoderTrainingConfig, TransformerAutoencoderConfig
from .evaluation import reconstruction_metrics
from .results import RepresentationResult
from .transformer_autoencoder import build_transformer_autoencoder
from .transformer_data import prepare_transformer_data, restore_transformer_reconstruction


def _masked_mse(torch: Any, prediction: Any, target: Any, element_mask: Any) -> Any:
    weights = element_mask.to(dtype=prediction.dtype)
    squared = (prediction - target).pow(2) * weights
    denominator = weights.sum().clamp_min(1.0)
    return squared.sum() / denominator


class TransformerAutoencoderTrainer(NeuralTrainer):
    """Stable Transformer reconstruction trainer using shared neural infrastructure."""

    def __init__(
        self,
        model: Any,
        config: AutoencoderTrainingConfig,
        *,
        callbacks: Iterable[Any] | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.callbacks = CallbackList(callbacks)

    def fit(
        self,
        tokens: np.ndarray,
        element_mask: np.ndarray,
        *,
        train_indices: np.ndarray,
        validation_indices: np.ndarray,
    ) -> NeuralTrainingResult:
        torch = require_torch(purpose="Transformer autoencoder training")
        from torch.utils.data import DataLoader, TensorDataset

        seed_everything(
            self.config.random_state,
            deterministic=self.config.deterministic,
            torch_module=torch,
        )
        device = resolve_device(self.config.device, torch_module=torch)
        model = self.model.to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        train_tokens = torch.as_tensor(tokens[train_indices], dtype=torch.float32)
        train_mask = torch.as_tensor(element_mask[train_indices], dtype=torch.bool)
        batch_size = min(self.config.batch_size, len(train_tokens))
        loader = DataLoader(
            TensorDataset(train_tokens, train_mask),
            batch_size=batch_size,
            shuffle=True,
            generator=torch_generator(self.config.random_state, torch_module=torch),
        )

        validation_tokens = None
        validation_mask = None
        if len(validation_indices):
            validation_tokens = torch.as_tensor(
                tokens[validation_indices], dtype=torch.float32, device=device
            )
            validation_mask = torch.as_tensor(
                element_mask[validation_indices], dtype=torch.bool, device=device
            )

        monitor = "validation_mse" if validation_tokens is not None else "train_mse"
        history = TrainingHistory()
        early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta,
            mode="min",
        )
        checkpoint = BestModelCheckpoint(
            monitor=monitor,
            mode="min",
            min_delta=self.config.min_delta,
            path=self.config.checkpoint_path,
        )
        context: dict[str, Any] = {
            "model": model,
            "optimizer": optimizer,
            "config": self.config,
            "device": device,
            "monitor": monitor,
        }
        self.callbacks.on_train_begin(context)

        stopped_epoch = self.config.epochs
        for epoch in range(1, self.config.epochs + 1):
            model.train()
            weighted_loss = 0.0
            valid_elements = 0
            for batch_tokens, batch_mask in loader:
                batch_tokens = batch_tokens.to(device)
                batch_mask = batch_mask.to(device)
                optimizer.zero_grad(set_to_none=True)
                reconstruction = model(batch_tokens, element_mask=batch_mask)
                loss = _masked_mse(torch, reconstruction, batch_tokens, batch_mask)
                loss.backward()
                optimizer.step()
                count = int(batch_mask.sum().detach().cpu())
                weighted_loss += float(loss.detach().cpu()) * count
                valid_elements += count
            train_loss = weighted_loss / max(valid_elements, 1)

            validation_loss: float | None = None
            monitored_loss = train_loss
            if validation_tokens is not None and validation_mask is not None:
                model.eval()
                with torch.no_grad():
                    validation_loss = float(
                        _masked_mse(
                            torch,
                            model(validation_tokens, element_mask=validation_mask),
                            validation_tokens,
                            validation_mask,
                        ).detach().cpu()
                    )
                monitored_loss = validation_loss

            record = history.append(
                epoch,
                train_mse=train_loss,
                validation_mse=validation_loss,
            )
            checkpoint.update(
                model,
                monitored_loss,
                epoch,
                optimizer=optimizer,
                metadata={"device": str(device)},
            )
            _, should_stop = early_stopping.update(monitored_loss, epoch)
            self.callbacks.on_epoch_end(epoch, dict(record), context)
            if should_stop:
                stopped_epoch = epoch
                break

        if self.config.restore_best:
            checkpoint.restore(model)
        model.eval()
        result = NeuralTrainingResult(
            model=model,
            history=history.to_records(),
            monitor=monitor,
            best_metric=checkpoint.best_value,
            best_epoch=checkpoint.best_epoch,
            stopped_epoch=stopped_epoch,
            device=str(device),
            checkpoint_path=(str(self.config.checkpoint_path) if self.config.checkpoint_path else None),
            metadata={
                "epochs_requested": self.config.epochs,
                "epochs_completed": len(history.records),
                "train_rows": int(len(train_indices)),
                "validation_rows": int(len(validation_indices)),
                "deterministic": self.config.deterministic,
                "restore_best": self.config.restore_best,
            },
        )
        context["result"] = result
        self.callbacks.on_train_end(context)
        return result


def train_transformer_autoencoder(
    X: Any,
    *,
    model_config: TransformerAutoencoderConfig | None = None,
    training_config: AutoencoderTrainingConfig | None = None,
    callbacks: Iterable[Any] | None = None,
) -> RepresentationResult:
    model_config = model_config or TransformerAutoencoderConfig()
    training_config = training_config or AutoencoderTrainingConfig()
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

    torch = require_torch(purpose="Transformer autoencoder support")
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
    model = build_transformer_autoencoder(prepared.token_dim, model_config)
    trainer = TransformerAutoencoderTrainer(model, training_config, callbacks=callbacks)
    training_result = trainer.fit(
        prepared.tokens,
        prepared.element_mask,
        train_indices=split.train_indices,
        validation_indices=split.validation_indices,
    )

    model = training_result.model
    device = resolve_device(training_config.device, torch_module=torch)
    full_tokens = torch.as_tensor(prepared.tokens, dtype=torch.float32, device=device)
    full_mask = torch.as_tensor(prepared.element_mask, dtype=torch.bool, device=device)
    with torch.no_grad():
        latent = model.encode(full_tokens, element_mask=full_mask).detach().cpu().numpy()
        reconstructed_tokens = model(full_tokens, element_mask=full_mask).detach().cpu().numpy()

    reconstruction = restore_transformer_reconstruction(prepared, reconstructed_tokens)
    metrics = reconstruction_metrics(prepared.original, reconstruction)
    mask = prepared.element_mask.astype(np.float32)
    token_error = (prepared.tokens - reconstructed_tokens) ** 2
    metrics["training_space_mse"] = float((token_error * mask).sum() / max(mask.sum(), 1.0))

    return RepresentationResult(
        method="transformer_autoencoder",
        latent=np.asarray(latent, dtype=float),
        reconstruction=np.asarray(reconstruction, dtype=float),
        metrics=metrics,
        feature_names=prepared.feature_names,
        model=model,
        transformer=prepared.scaler,
        history=training_result.history,
        metadata={
            "input_shape": list(prepared.input_shape),
            "token_count": prepared.token_count,
            "token_dim": prepared.token_dim,
            "token_width": model_config.token_width,
            "model_dim": model_config.model_dim,
            "latent_dim": model_config.latent_dim,
            "nhead": model_config.nhead,
            "encoder_layers": model_config.encoder_layers,
            "decoder_layers": model_config.decoder_layers,
            "feedforward_dim": model_config.feedforward_dim,
            "dropout": model_config.dropout,
            "transformer_activation": model_config.transformer_activation,
            "output_activation": model_config.output_activation,
            "max_tokens": model_config.max_tokens,
            "norm_first": model_config.norm_first,
            "scaling": training_config.scaling,
            "device": training_result.device,
            "epochs_requested": training_config.epochs,
            "epochs_completed": len(training_result.history),
            "stopped_epoch": training_result.stopped_epoch,
            "best_epoch": training_result.best_epoch,
            "best_monitored_mse": training_result.best_metric,
            "monitor": training_result.monitor,
            "validation_rows": split.validation_count,
            "deterministic": training_config.deterministic,
            "checkpoint_path": training_result.checkpoint_path,
            "restore_best": training_config.restore_best,
        },
    )
