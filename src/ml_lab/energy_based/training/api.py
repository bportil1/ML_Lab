from __future__ import annotations

from typing import Any, Iterable

from .config import CDKTrainingConfig
from .data import validate_matrix


def fit(
    model,
    X,
    *,
    config: CDKTrainingConfig | dict[str, Any] | None = None,
    feature_names: list[str] | None = None,
    callbacks: Iterable[Any] | None = None,
    verbose: bool = False,
):
    from .cdk import CDKTrainer

    resolved = config if isinstance(config, CDKTrainingConfig) else CDKTrainingConfig.from_mapping(config)
    return CDKTrainer(resolved).fit(
        model,
        X,
        feature_names=feature_names,
        callbacks=callbacks,
        verbose=verbose,
    )


def run(
    X,
    *,
    family: str = "bernoulli",
    hidden_dim: int = 8,
    model_config: dict[str, Any] | None = None,
    training_config: CDKTrainingConfig | dict[str, Any] | None = None,
    feature_names: list[str] | None = None,
    callbacks: Iterable[Any] | None = None,
    verbose: bool = False,
):
    from ml_lab.energy_based import rbm
    from ml_lab.neural.backend import require_torch
    from ml_lab.neural.runtime import seed_everything

    values, inferred_names = validate_matrix(X)
    resolved_training = (
        training_config
        if isinstance(training_config, CDKTrainingConfig)
        else CDKTrainingConfig.from_mapping(training_config)
    )
    torch = require_torch(purpose="RBM initialization and CD-k training")
    seed_everything(
        resolved_training.random_state,
        deterministic=resolved_training.deterministic,
        torch_module=torch,
    )
    options = dict(model_config or {})
    model = rbm.create(
        family,
        visible_dim=values.shape[1],
        hidden_dim=int(hidden_dim),
        sharpness=float(options.pop("sharpness", 1.0)),
        dropout=float(options.pop("dropout", 0.0)),
        settings=options.pop("settings", None),
    )
    if options:
        raise ValueError(f"unknown RBM model_config fields: {sorted(options)}")
    return fit(
        model,
        values,
        config=resolved_training,
        feature_names=feature_names or inferred_names,
        callbacks=callbacks,
        verbose=verbose,
    )
