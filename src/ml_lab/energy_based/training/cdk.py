from __future__ import annotations

import math
import time
from typing import Any, Iterable

import numpy as np

from ml_lab.neural.backend import require_torch
from ml_lab.neural.callbacks import CallbackList
from ml_lab.neural.runtime import resolve_device, seed_everything

from .base import TrainingResult, TrainingScheme
from .config import CDKTrainingConfig
from .control import CDKControlState, update_momentum
from .data import iter_batches, resolve_batching, validate_matrix
from .diagnostics import (
    checkpoint_due,
    collect_distribution_snapshot,
    collect_epoch_metrics,
    estimate_log_partition,
    mean_log_likelihood,
)
from .model_selection import BestModelTracker, selection_metric_value
from .objective import contrastive_divergence_objective
from .phases import negative_phase, prepare_visible_batch
from .updates import build_optimizer, optimizer_step, set_optimizer_momentum


class CDKTrainer(TrainingScheme):
    name = "cdk"

    def __init__(self, config: CDKTrainingConfig | None = None):
        self.config = config or CDKTrainingConfig()

    def fit(
        self,
        model,
        data,
        *,
        feature_names: list[str] | None = None,
        callbacks: Iterable[Any] | None = None,
        verbose: bool = False,
    ) -> TrainingResult:
        torch = require_torch(purpose="CD-k RBM training")
        values, inferred_names = validate_matrix(data)
        if values.shape[1] != int(model.num_visible):
            raise ValueError(
                f"training feature count {values.shape[1]} does not match RBM visible_dim {model.num_visible}"
            )
        names = list(feature_names or inferred_names)
        if len(names) != values.shape[1]:
            raise ValueError("feature_names length must match the RBM visible dimension")

        seed_everything(self.config.random_state, deterministic=self.config.deterministic, torch_module=torch)
        device = resolve_device(self.config.device, torch_module=torch)
        model.to(device=device)
        model.train()
        x_all = torch.as_tensor(values, dtype=model.W.dtype, device="cpu")
        batching = resolve_batching(len(x_all), self.config.batch_size, self.config.drop_last_batch)
        optimizer = build_optimizer(
            model,
            learning_rate=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )
        control = CDKControlState(momentum=float(self.config.momentum))
        selection = BestModelTracker(self.config.model_selection)
        callback_list = CallbackList(callbacks)
        persistent_by_batch_size: dict[int, Any] = {}
        history: list[dict[str, Any]] = []
        stop_reason: str | None = None

        context = {
            "scheme": self.name,
            "model": model,
            "optimizer": optimizer,
            "config": self.config,
            "device": str(device),
        }
        callback_list.on_train_begin(context)

        for epoch in range(int(self.config.epochs)):
            started = time.time()
            epoch_objectives: list[float] = []
            update_norms: list[float] = []
            epoch_momentum = float(control.momentum)
            set_optimizer_momentum(optimizer, epoch_momentum)
            generator = torch.Generator().manual_seed(int(self.config.random_state) + int(epoch))

            for raw_batch in iter_batches(
                x_all,
                batching,
                shuffle=self.config.shuffle_batches,
                generator=generator,
            ):
                batch = raw_batch.to(device=device, dtype=model.W.dtype)
                batch = prepare_visible_batch(
                    batch,
                    model.family,
                    clamp_visible=self.config.clamp_visible,
                )
                batch_size = int(batch.shape[0])
                phase = negative_phase(
                    model,
                    batch,
                    gibbs_steps=self.config.gibbs_steps,
                    num_chains=self.config.num_chains,
                    persistent=self.config.persistent,
                    persistent_state=persistent_by_batch_size.get(batch_size),
                    clamp_visible=self.config.clamp_visible,
                )
                if self.config.persistent:
                    persistent_by_batch_size[batch_size] = phase.persistent_state
                objective = contrastive_divergence_objective(model, batch, phase.visible)
                update_norm = optimizer_step(
                    model,
                    optimizer,
                    objective,
                    max_weight_row_norm=self.config.max_weight_row_norm,
                )
                epoch_objectives.append(float(objective.detach().cpu()))
                update_norms.append(update_norm)

            metrics = collect_epoch_metrics(
                model,
                x_all,
                update_norms=update_norms,
                cd_objectives=epoch_objectives,
                energy_sample_size=self.config.energy_sample_size,
            )

            nonfinite = any(
                not math.isfinite(float(metrics[key]))
                for key in ("energy_gap", "reconstruction_error", "cd_objective", "update_norm")
            )
            selection_value = selection_metric_value(
                self.config.model_selection.metric,
                energy_gap=metrics["energy_gap"],
                reconstruction_error=metrics["reconstruction_error"],
                cd_objective=metrics["cd_objective"],
            )
            selection_update = selection.update(model, metric=selection_value, epoch=epoch)

            pending_stop = None
            if self.config.stop_on_nonfinite and nonfinite:
                pending_stop = "nonfinite_training_metric"
            elif selection_update.should_stop:
                pending_stop = "model_selection_patience"
            elif epoch >= self.config.epochs - 1:
                pending_stop = "epochs_completed"

            force_checkpoint = pending_stop is not None
            distribution_record = {
                "distribution_metric_checkpoint": False,
                "wasserstein_feature_mean": float("nan"),
                "wasserstein_feature_max": float("nan"),
                "sinkhorn_distance": float("nan"),
                "generated_sample_count": 0,
            }
            if (
                self.config.distribution_monitoring.enabled
                and not nonfinite
                and (
                    force_checkpoint
                    or checkpoint_due(
                        epoch,
                        self.config.epochs,
                        mode="interval",
                        interval=self.config.distribution_monitoring.interval,
                    )
                )
            ):
                snapshot = collect_distribution_snapshot(
                    model,
                    x_all,
                    sample_size=self.config.distribution_monitoring.sample_size,
                    gibbs_steps=self.config.distribution_monitoring.gibbs_steps,
                    sinkhorn_enabled=self.config.distribution_monitoring.sinkhorn_enabled,
                    sinkhorn_p=self.config.distribution_monitoring.sinkhorn_p,
                    sinkhorn_regularization=self.config.distribution_monitoring.sinkhorn_regularization,
                    sinkhorn_iterations=self.config.distribution_monitoring.sinkhorn_iterations,
                )
                distribution_record.update({
                    "distribution_metric_checkpoint": True,
                    "wasserstein_feature_mean": snapshot.diagnostics.wasserstein_feature_mean,
                    "wasserstein_feature_max": snapshot.diagnostics.wasserstein_feature_max,
                    "sinkhorn_distance": snapshot.diagnostics.sinkhorn_distance,
                    "generated_sample_count": snapshot.diagnostics.generated_sample_count,
                })

            partition_record = {
                "partition_checkpoint": False,
                "log_partition_estimate": float("nan"),
                "mean_log_likelihood_estimate": float("nan"),
                "partition_effective_sample_size": float("nan"),
                "partition_method": None,
                "partition_proposal": None,
            }
            if (
                self.config.partition_monitoring.enabled
                and not nonfinite
                and (
                    force_checkpoint
                    or checkpoint_due(
                        epoch,
                        self.config.epochs,
                        mode=self.config.partition_monitoring.schedule,
                        interval=self.config.partition_monitoring.interval,
                    )
                )
            ):
                partition = estimate_log_partition(model, self.config.partition_monitoring, epoch=epoch)
                partition_record.update({
                    "partition_checkpoint": True,
                    "log_partition_estimate": partition.log_z,
                    "mean_log_likelihood_estimate": mean_log_likelihood(model, x_all, partition),
                    "partition_effective_sample_size": partition.effective_sample_size,
                    "partition_method": partition.method,
                    "partition_proposal": partition.proposal,
                })

            record = {
                "epoch": int(epoch),
                "epoch_time_s": float(time.time() - started),
                "gibbs_steps": int(self.config.gibbs_steps),
                "num_chains": int(self.config.num_chains),
                "persistent": bool(self.config.persistent),
                "momentum": epoch_momentum,
                "weight_decay": float(self.config.weight_decay),
                "requested_batch_size": batching.requested_batch_size,
                "effective_batch_size": batching.effective_batch_size,
                "batch_count": batching.batch_count,
                **metrics,
                **distribution_record,
                **partition_record,
                "selection_metric": selection_update.metric,
                "selection_best_metric": selection_update.best_metric,
                "selection_best_epoch": selection_update.best_epoch,
                "selection_new_best": selection_update.new_best,
                "selection_significant_improvement": selection_update.significant_improvement,
                "selection_epochs_since_improvement": selection_update.epochs_since_improvement,
                "stop_reason": pending_stop,
            }
            history.append(record)
            callback_list.on_epoch_end(epoch, record, context)

            update_momentum(
                control,
                energy_gap=metrics["energy_gap"],
                config=self.config.adaptive_momentum,
            )

            if verbose:
                suffix = ""
                if distribution_record["distribution_metric_checkpoint"]:
                    suffix += f" W1={distribution_record['wasserstein_feature_mean']:.4g}"
                if partition_record["partition_checkpoint"]:
                    suffix += f" logZ={partition_record['log_partition_estimate']:.4g}"
                print(
                    f"[CD-k][{model.family}] epoch={epoch + 1}/{self.config.epochs} "
                    f"objective={metrics['cd_objective']:.6g} energy_gap={metrics['energy_gap']:.6g} "
                    f"recon={metrics['reconstruction_error']:.6g}{suffix}"
                )

            if pending_stop is not None:
                stop_reason = pending_stop
                break

        selection.restore(model)
        model.eval()

        final_metrics = collect_epoch_metrics(
            model,
            x_all,
            update_norms=[],
            cd_objectives=[],
            energy_sample_size=self.config.energy_sample_size,
        )
        final_metrics.pop("cd_objective", None)
        final_metrics.pop("update_norm", None)
        final_snapshot = None
        if self.config.distribution_monitoring.enabled:
            final_snapshot = collect_distribution_snapshot(
                model,
                x_all,
                sample_size=self.config.distribution_monitoring.sample_size,
                gibbs_steps=self.config.distribution_monitoring.gibbs_steps,
                sinkhorn_enabled=self.config.distribution_monitoring.sinkhorn_enabled,
                sinkhorn_p=self.config.distribution_monitoring.sinkhorn_p,
                sinkhorn_regularization=self.config.distribution_monitoring.sinkhorn_regularization,
                sinkhorn_iterations=self.config.distribution_monitoring.sinkhorn_iterations,
            )
            final_metrics.update({
                "wasserstein_feature_mean": final_snapshot.diagnostics.wasserstein_feature_mean,
                "wasserstein_feature_max": final_snapshot.diagnostics.wasserstein_feature_max,
                "sinkhorn_distance": final_snapshot.diagnostics.sinkhorn_distance,
                "generated_sample_count": final_snapshot.diagnostics.generated_sample_count,
            })

        final_partition = None
        if self.config.partition_monitoring.enabled:
            final_partition = estimate_log_partition(
                model,
                self.config.partition_monitoring,
                epoch=len(history),
            )
            final_metrics.update({
                "log_partition_estimate": final_partition.log_z,
                "mean_log_likelihood_estimate": mean_log_likelihood(model, x_all, final_partition),
                "partition_effective_sample_size": final_partition.effective_sample_size,
            })

        metadata = {
            "model": model.configuration(),
            "config": self.config.to_record(),
            "device": str(device),
            "batching": batching.to_record(),
            "model_selection": selection.metadata(),
            "gradient_method": "autograd_cd_energy_difference",
            "negative_chain_gradients": "detached",
            "weight_decay_scope": "W_only",
            "monitoring": {
                "mean_log_unnormalized_probability": "mean(-E(v)); epoch-wise and does not require Z",
                "sinkhorn_distance": "entropic-regularized balanced transport approximation, not exact EMD",
                "partition": None if final_partition is None else final_partition.to_record(),
            },
        }
        context["result_metadata"] = metadata
        callback_list.on_train_end(context)

        return TrainingResult(
            model=model,
            history=history,
            scheme=self.name,
            stop_reason=stop_reason or "epochs_completed",
            feature_names=names,
            metrics=final_metrics,
            generated_samples=(None if final_snapshot is None else final_snapshot.generated.numpy()),
            reference_samples=(None if final_snapshot is None else final_snapshot.reference.numpy()),
            metadata=metadata,
        )
