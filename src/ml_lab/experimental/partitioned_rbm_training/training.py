from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

import numpy as np

from ml_lab.energy_based import rbm
from ml_lab.energy_based.training import CDKTrainingConfig, TrainingResult
from ml_lab.energy_based.training.api import fit as fit_cdk
from ml_lab.energy_based.training.data import validate_matrix
from ml_lab.neural.backend import require_torch
from ml_lab.neural.runtime import resolve_device, seed_everything

from .config import PartitionedRBMTrainingConfig
from .parameters import initialize_global_cross_weights, initialize_local_from_global, write_local_to_global
from .partitioning import PartitionBlock, initial_blocks, merge_blocks


@dataclass(slots=True)
class PartitionStageResult:
    level: int
    block_index: int
    block: PartitionBlock
    epochs_requested: int
    result: TrainingResult

    def to_record(self) -> dict[str, Any]:
        return {
            "level": int(self.level),
            "block_index": int(self.block_index),
            "block": self.block.to_record(),
            "epochs_requested": int(self.epochs_requested),
            "epochs_completed": len(self.result.history),
            "stop_reason": self.result.stop_reason,
            "metrics": dict(self.result.metrics),
        }


@dataclass(slots=True)
class PartitionedRBMResult:
    model: Any
    feature_names: list[str]
    family: str
    hidden_dim: int
    partition_config: PartitionedRBMTrainingConfig
    base_training_config: CDKTrainingConfig
    stages: list[PartitionStageResult]

    @property
    def final_stage(self) -> PartitionStageResult:
        return self.stages[-1]

    def to_record(self) -> dict[str, Any]:
        return {
            "experiment": "partitioned_rbm_training",
            "family": self.family,
            "visible_dim": int(self.model.num_visible),
            "hidden_dim": int(self.model.num_hidden),
            "feature_names": list(self.feature_names),
            "partition_config": self.partition_config.to_record(),
            "base_training_config": self.base_training_config.to_record(),
            "stage_count": len(self.stages),
            "stages": [stage.to_record() for stage in self.stages],
            "final_metrics": dict(self.final_stage.result.metrics),
            "final_stop_reason": self.final_stage.result.stop_reason,
            "model_config": self.model.configuration(),
            "method": "hierarchical_feature_blocks_then_merged_refinement",
            "scientific_note": (
                "Visible features and hidden units are partitioned independently. Samples are never partitioned. "
                "Each merged stage warm-starts previously trained within-block parameters and learns newly exposed cross-block weights."
            ),
        }


def _stage_training_config(
    base: CDKTrainingConfig,
    *,
    epochs: int,
    final_stage: bool,
    monitor_intermediate: bool,
    level: int,
) -> CDKTrainingConfig:
    distribution = base.distribution_monitoring
    partition = base.partition_monitoring
    if not final_stage and not monitor_intermediate:
        distribution = replace(distribution, enabled=False)
        partition = replace(partition, enabled=False)
    return replace(
        base,
        epochs=int(epochs),
        random_state=int(base.random_state) + int(level) * 1009,
        distribution_monitoring=distribution,
        partition_monitoring=partition,
    )


def fit_partitioned(
    X,
    *,
    family: str = "bernoulli",
    hidden_dim: int = 8,
    model_config: dict[str, Any] | None = None,
    training_config: CDKTrainingConfig | dict[str, Any] | None = None,
    partition_config: PartitionedRBMTrainingConfig | dict[str, Any] | None = None,
    feature_names: list[str] | None = None,
    callbacks: Iterable[Any] | None = None,
    verbose: bool = False,
) -> PartitionedRBMResult:
    values, inferred_names = validate_matrix(X)
    names = list(feature_names or inferred_names)
    if len(names) != values.shape[1]:
        raise ValueError("feature_names length must match visible feature count")

    base_training = (
        training_config
        if isinstance(training_config, CDKTrainingConfig)
        else CDKTrainingConfig.from_mapping(training_config)
    )
    partitioning = (
        partition_config
        if isinstance(partition_config, PartitionedRBMTrainingConfig)
        else PartitionedRBMTrainingConfig.from_mapping(partition_config)
    )
    if partitioning.initial_partitions > values.shape[1]:
        raise ValueError("initial_partitions cannot exceed the input feature count")
    if partitioning.initial_partitions > int(hidden_dim):
        raise ValueError("initial_partitions cannot exceed hidden_dim")

    options = dict(model_config or {})
    sharpness = float(options.pop("sharpness", 1.0))
    dropout = float(options.pop("dropout", 0.0))
    settings = options.pop("settings", None)
    if options:
        raise ValueError(f"unknown RBM model_config fields: {sorted(options)}")

    torch = require_torch(purpose="experimental partitioned RBM training")
    seed_everything(base_training.random_state, deterministic=base_training.deterministic, torch_module=torch)
    device = resolve_device(base_training.device, torch_module=torch)
    global_model = rbm.create(
        family,
        visible_dim=values.shape[1],
        hidden_dim=int(hidden_dim),
        sharpness=sharpness,
        dropout=dropout,
        settings=settings,
        device=str(device),
    )
    initialize_global_cross_weights(global_model, mode=partitioning.cross_weight_initialization)

    blocks = initial_blocks(values.shape[1], int(hidden_dim), partitioning.initial_partitions)
    stages: list[PartitionStageResult] = []
    level = 0
    while True:
        final_level = len(blocks) == 1
        epochs = partitioning.epochs_for_level(level, base_epochs=base_training.epochs)
        next_blocks: list[PartitionBlock] = []

        for block_index, block in enumerate(blocks):
            block_seed = int(base_training.random_state) + level * 1009 + block_index * 97
            seed_everything(block_seed, deterministic=base_training.deterministic, torch_module=torch)
            local = rbm.create(
                family,
                visible_dim=block.visible_dim,
                hidden_dim=block.hidden_dim,
                sharpness=sharpness,
                dropout=dropout,
                settings=settings,
                device=str(device),
            )
            if level > 0:
                initialize_local_from_global(local, global_model, block)

            local_data = values[:, block.visible_start : block.visible_end]
            local_names = names[block.visible_start : block.visible_end]
            stage_config = _stage_training_config(
                base_training,
                epochs=epochs,
                final_stage=final_level,
                monitor_intermediate=partitioning.monitor_intermediate_stages,
                level=level,
            )
            # The block-specific seed controls stochastic training without depending on execution order.
            stage_config = replace(stage_config, random_state=block_seed)
            result = fit_cdk(
                local,
                local_data,
                config=stage_config,
                feature_names=local_names,
                callbacks=callbacks,
                verbose=verbose,
            )
            updated_block = write_local_to_global(local, global_model, block, final=final_level)
            next_blocks.append(updated_block)
            stages.append(
                PartitionStageResult(
                    level=level,
                    block_index=block_index,
                    block=updated_block,
                    epochs_requested=epochs,
                    result=result,
                )
            )

        if final_level:
            # The one-block local model covers the complete parameterization. Returning the
            # global copy keeps the same object shape/API used throughout this experiment.
            global_model.train(next_blocks and stages[-1].result.model.training)
            break
        blocks = merge_blocks(next_blocks, merge_factor=partitioning.merge_factor)
        level += 1

    return PartitionedRBMResult(
        model=global_model,
        feature_names=names,
        family=global_model.family,
        hidden_dim=int(hidden_dim),
        partition_config=partitioning,
        base_training_config=base_training,
        stages=stages,
    )
