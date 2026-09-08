from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

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

from .autoencoder import build_mlp_autoencoder
from .config import AutoencoderTrainingConfig, MLPAutoencoderConfig
from .evaluation import reconstruction_metrics
from .results import RepresentationResult


def _torch_training_parts():
    torch = require_torch(purpose="MLP autoencoder support")
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    return torch, nn, DataLoader, TensorDataset


def _matrix(X: Any) -> tuple[np.ndarray, list[str]]:
    if isinstance(X, pd.DataFrame):
        feature_names = [str(column) for column in X.columns]
        values = X.to_numpy(dtype=np.float32)
    else:
        values = np.asarray(X, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("X must be a two-dimensional feature matrix")
        feature_names = [f"feature_{index}" for index in range(values.shape[1])]
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("X must contain at least two rows and one feature")
    if not np.isfinite(values).all():
        raise ValueError("X must contain only finite values; imputation is not implicit")
    return values, feature_names


def _scaler(mode: str):
    if mode == "none":
        return None
    return {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }[mode]


class MLPAutoencoderTrainer(NeuralTrainer):
    """Task-specific trainer built on ML_Lab's shared neural infrastructure."""

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
        training_values: np.ndarray,
        *,
        train_indices: np.ndarray,
        validation_indices: np.ndarray,
    ) -> NeuralTrainingResult:
        torch, nn, DataLoader, TensorDataset = _torch_training_parts()
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
        criterion = nn.MSELoss()

        train_tensor = torch.as_tensor(training_values[train_indices], dtype=torch.float32)
        batch_size = min(self.config.batch_size, len(train_tensor))
        loader = DataLoader(
            TensorDataset(train_tensor),
            batch_size=batch_size,
            shuffle=True,
            generator=torch_generator(self.config.random_state, torch_module=torch),
        )
        validation_tensor = None
        if len(validation_indices):
            validation_tensor = torch.as_tensor(
                training_values[validation_indices], dtype=torch.float32, device=device
            )

        monitor = "validation_mse" if validation_tensor is not None else "train_mse"
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
            total_loss = 0.0
            total_rows = 0
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                reconstruction = model(batch)
                loss = criterion(reconstruction, batch)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * len(batch)
                total_rows += len(batch)
            train_loss = total_loss / max(total_rows, 1)

            validation_loss: float | None = None
            monitored_loss = train_loss
            if validation_tensor is not None:
                model.eval()
                with torch.no_grad():
                    validation_loss = float(
                        criterion(model(validation_tensor), validation_tensor).detach().cpu()
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


def train_mlp_autoencoder(
    X: Any,
    *,
    model_config: MLPAutoencoderConfig | None = None,
    training_config: AutoencoderTrainingConfig | None = None,
    callbacks: Iterable[Any] | None = None,
) -> RepresentationResult:
    model_config = model_config or MLPAutoencoderConfig()
    training_config = training_config or AutoencoderTrainingConfig()
    values, feature_names = _matrix(X)
    torch = require_torch(purpose="MLP autoencoder support")
    seed_everything(
        training_config.random_state,
        deterministic=training_config.deterministic,
        torch_module=torch,
    )

    scaler = _scaler(training_config.scaling)
    training_values = values if scaler is None else scaler.fit_transform(values).astype(np.float32)
    split = split_validation_indices(
        len(training_values),
        training_config.validation_fraction,
        random_state=training_config.random_state,
    )

    model = build_mlp_autoencoder(values.shape[1], model_config)
    trainer = MLPAutoencoderTrainer(model, training_config, callbacks=callbacks)
    training_result = trainer.fit(
        training_values,
        train_indices=split.train_indices,
        validation_indices=split.validation_indices,
    )
    model = training_result.model
    device = resolve_device(training_config.device, torch_module=torch)
    full_tensor = torch.as_tensor(training_values, dtype=torch.float32, device=device)
    with torch.no_grad():
        latent = model.encode(full_tensor).detach().cpu().numpy()
        reconstructed_training = model(full_tensor).detach().cpu().numpy()

    reconstruction = (
        reconstructed_training if scaler is None else scaler.inverse_transform(reconstructed_training)
    )
    reconstruction = np.asarray(reconstruction, dtype=float)
    metrics = reconstruction_metrics(values, reconstruction)
    metrics["training_space_mse"] = float(np.mean((training_values - reconstructed_training) ** 2))

    return RepresentationResult(
        method="mlp_autoencoder",
        latent=np.asarray(latent, dtype=float),
        reconstruction=reconstruction,
        metrics=metrics,
        feature_names=feature_names,
        model=model,
        transformer=scaler,
        history=training_result.history,
        metadata={
            "input_dim": int(values.shape[1]),
            "hidden_dims": list(model_config.hidden_dims),
            "latent_dim": model_config.latent_dim,
            "activation": model_config.activation,
            "output_activation": model_config.output_activation,
            "dropout": model_config.dropout,
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
