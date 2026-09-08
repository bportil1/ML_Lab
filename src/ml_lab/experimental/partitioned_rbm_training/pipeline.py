from __future__ import annotations

from pathlib import Path
from typing import Any

from .reporting import save_result
from .training import fit_partitioned


def run(
    *,
    mode: str = "describe",
    X=None,
    family: str = "bernoulli",
    hidden_dim: int = 8,
    model_config: dict[str, Any] | None = None,
    training_config: dict[str, Any] | None = None,
    partition_config: dict[str, Any] | None = None,
    feature_names: list[str] | None = None,
    output_dir: str | None = None,
    verbose: bool = False,
):
    if mode == "describe":
        return {
            "experiment": "partitioned_rbm_training",
            "status": "experimental",
            "method": "hierarchical_feature_blocks_then_merged_refinement",
            "stable_dependencies": ["ml_lab.energy_based.rbm", "ml_lab.energy_based.training.cdk"],
            "supported_families": ["bernoulli", "gaussian", "student_t_poe"],
            "notes": (
                "Visible features and hidden units are partitioned independently. Samples remain intact at every stage. "
                "Local RBMs are trained first, neighboring blocks are merged, and cross-block weights are then refined."
            ),
        }
    if mode not in {"fit", "train"}:
        raise ValueError("mode must be describe, fit, or train")
    if X is None:
        raise ValueError("X is required for partitioned RBM training")
    result = fit_partitioned(
        X,
        family=family,
        hidden_dim=hidden_dim,
        model_config=model_config,
        training_config=training_config,
        partition_config=partition_config,
        feature_names=feature_names,
        verbose=verbose,
    )
    record = result.to_record()
    if output_dir is not None:
        record["output_dir"] = str(save_result(result, Path(output_dir)))
    return record
