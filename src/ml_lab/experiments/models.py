"""Typed, opt-in model selection for project-local experiments (Sprint 3.1).

This is a configuration/catalog layer, NOT a new model runner. It reuses the
existing estimator registries and scientific config classes, without importing
PyTorch, Flask, PAH, or another project's code.
"""
from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
import json
import math
from types import UnionType
from typing import Any, Literal, Mapping, Union, get_args, get_origin, get_type_hints

from .datasets import prepare_model_handoff, verify_selection
from .store import ExperimentStore, ExperimentStoreError

MODEL_SCHEMA = "ml-lab.model-configuration@1"
HANDOFF_SCHEMA = "ml-lab.configured-model-handoff@1"
_FAMILIES = (
    ("classification", "Classification", "supervised", "one", "search"),
    ("regression", "Regression", "supervised", "one", "search"),
    ("clustering", "Clustering", "unsupervised", "optional_one", "search"),
    ("representation", "Representation", "representation", "none", "method"),
    ("rbm", "Energy-based / RBM", "energy_based", "none", "family"),
    ("gan", "Generative / GAN", "generative", "none", "architecture"),
)
_FAMILY_MAP = {row[0]: row for row in _FAMILIES}


class ModelConfigurationError(ExperimentStoreError):
    """Invalid method, typed parameters, or experiment configuration revision."""


def _strict_json_object(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ModelConfigurationError(f"{label} must be a JSON object with string keys")
    try:
        return json.loads(json.dumps(dict(value), allow_nan=False))
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise ModelConfigurationError(f"{label} must be finite and JSON-compatible") from exc


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelConfigurationError(f"{label} must be an object")
    return _strict_json_object(value, label)


def _typed(value: Any, hint: Any, label: str) -> Any:
    """Check JSON controls without coercing strings, bools or numeric types."""
    if hint is Any:
        return value
    origin = get_origin(hint)
    args = get_args(hint)
    if origin is Literal:
        if not any(type(value) is type(v) and value == v for v in args):
            raise ModelConfigurationError(f"{label} must be one of {list(args)!r}")
        return value
    if origin in (Union, UnionType):
        for candidate in args:
            try:
                return _typed(value, candidate, label)
            except ModelConfigurationError:
                continue
        raise ModelConfigurationError(f"{label} has the wrong type")
    if hint is type(None):
        if value is not None:
            raise ModelConfigurationError(f"{label} must be null")
        return None
    if hint in (int, float, bool, str):
        if hint is float:
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ModelConfigurationError(f"{label} must be a finite number")
            return float(value)
        if type(value) is not hint:
            raise ModelConfigurationError(f"{label} must be {hint.__name__}")
        return value
    if origin in (list, tuple):
        if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
            raise ModelConfigurationError(f"{label} must be an array")
        if len(args) == 2 and args[1] is Ellipsis:
            values = [_typed(item, args[0], f"{label}[{i}]") for i, item in enumerate(value)]
        elif args:
            if len(value) != len(args):
                raise ModelConfigurationError(f"{label} must have {len(args)} items")
            values = [_typed(item, part, f"{label}[{i}]") for i, (item, part) in enumerate(zip(value, args))]
        else:
            values = list(value)
        return tuple(values) if origin is tuple else values
    if isinstance(hint, type) and is_dataclass(hint):
        return _dataclass_config(hint, value, label)
    # All current stable model configs use the primitive / Literal / tuple /
    # dataclass types above. Reject new unsupported annotations rather than
    # silently accepting unvalidated controls.
    raise ModelConfigurationError(f"{label}: unsupported configuration type {hint!r}")


def _dataclass_config(cls: type, raw: Any, label: str) -> Any:
    data = _mapping(raw, label)
    field_types = get_type_hints(cls)
    unknown = set(data) - {item.name for item in fields(cls)}
    if unknown:
        raise ModelConfigurationError(f"Unknown {label} fields: {sorted(unknown)}")
    values = {key: _typed(value, field_types[key], f"{label}.{key}") for key, value in data.items()}
    try:
        instance = cls(**values)
        validate = getattr(instance, "validate", None)
        if callable(validate):
            validate()
    except (ValueError, TypeError, OverflowError, ZeroDivisionError) as exc:
        raise ModelConfigurationError(f"Invalid {label}: {exc}") from exc
    return instance


def _model_classes(family: str, method: str) -> dict[str, type]:
    if family == "classification":
        from ml_lab.classification.config import ClassificationSearchConfig
        return {"config": ClassificationSearchConfig}
    if family == "regression":
        from ml_lab.regression.config import RegressionSearchConfig
        return {"config": RegressionSearchConfig}
    if family == "clustering":
        from ml_lab.clustering.config import ClusteringSearchConfig
        return {"config": ClusteringSearchConfig}
    if family == "representation":
        from ml_lab.representation.config import (
            AutoencoderTrainingConfig, MLPAutoencoderConfig,
            PCARepresentationConfig, TransformerAutoencoderConfig,
        )
        if method == "pca":
            return {"config": PCARepresentationConfig}
        return {
            "model_config": (MLPAutoencoderConfig if method == "mlp_autoencoder" else TransformerAutoencoderConfig),
            "training_config": AutoencoderTrainingConfig,
        }
    if family == "gan":
        from ml_lab.generative.gan.config import GANModelConfig, GANTrainingConfig
        return {"model_config": GANModelConfig, "training_config": GANTrainingConfig}
    if family == "rbm":
        from ml_lab.energy_based.training.config import CDKTrainingConfig
        return {"training_config": CDKTrainingConfig}
    raise ModelConfigurationError(f"Unknown model family: {family}")


def list_model_families() -> list[dict[str, str]]:
    """Stable task families; no experimental method is implicitly promoted."""
    return [
        {"id": task, "name": label, "category": category, "targets": targets, "method_kind": kind}
        for task, label, category, targets, kind in _FAMILIES
    ]


def list_model_methods(family: str) -> list[dict[str, str]]:
    """Read the current stable source registries, not a duplicate estimator list."""
    if family not in _FAMILY_MAP:
        raise ModelConfigurationError(f"Unknown model family: {family!r}")
    if family in {"classification", "regression", "clustering"}:
        if family == "classification":
            from ml_lab.classification.registry import list_estimators
        elif family == "regression":
            from ml_lab.regression.registry import list_estimators
        else:
            from ml_lab.clustering.registry import list_estimators
        return ([{"id": "suite", "name": "Multiple estimators"}] +
                [{"id": spec.id, "name": spec.name} for spec in list_estimators()])
    if family == "representation":
        return [
            {"id": "pca", "name": "PCA"},
            {"id": "mlp_autoencoder", "name": "MLP Autoencoder"},
            {"id": "transformer_autoencoder", "name": "Transformer Autoencoder"},
        ]
    if family == "rbm":
        from ml_lab.energy_based.rbm.registry import list_families
        return [{"id": spec.id, "name": spec.name} for spec in list_families()]
    return [{"id": "gan", "name": "GAN (configurable generator / discriminator)"}]


def _method_exists(family: str, method: str) -> None:
    if not isinstance(method, str) or method not in {item["id"] for item in list_model_methods(family)}:
        raise ModelConfigurationError(f"Unsupported {family} model method: {method!r}")


def _type_descriptor(hint: Any) -> dict[str, Any]:
    origin, args = get_origin(hint), get_args(hint)
    if origin is Literal:
        return {"type": "enum", "choices": list(args)}
    if origin in (Union, UnionType):
        variants = [_type_descriptor(v) for v in args if v is not type(None)]
        result = {"type": variants[0]["type"]} if len(variants) == 1 else {"type": "union", "variants": variants}
        if type(None) in args:
            result["nullable"] = True
        return result
    if origin in (tuple, list):
        return {"type": "array", "items": _type_descriptor(args[0]) if args else {"type": "any"}}
    if isinstance(hint, type) and is_dataclass(hint):
        return {"type": "object", "fields": _config_fields(hint)}
    if hint in (str, int, float, bool):
        return {"type": {str: "string", int: "integer", float: "number", bool: "boolean"}[hint]}
    return {"type": "any"}


def _config_fields(cls: type) -> list[dict[str, Any]]:
    hints = get_type_hints(cls)
    defaults = cls()
    result = []
    for field in fields(cls):
        spec = {"name": field.name, **_type_descriptor(hints[field.name])}
        default = getattr(defaults, field.name)
        if is_dataclass(default):
            default = asdict(default)
        if isinstance(default, float) and not math.isfinite(default):
            spec["default_nonfinite"] = True  # legacy NaN default is intentionally not persisted
        else:
            spec["default"] = default
        result.append(spec)
    return result


def describe_model_method(family: str, method: str) -> dict[str, Any]:
    """UI-ready control metadata; inspect only (does not create a model)."""
    _method_exists(family, method)
    controls = {name: _config_fields(cls) for name, cls in _model_classes(family, method).items()}
    if family in {"classification", "regression"}:
        controls["task_options"] = [{"name": "test_size", "type": "number", "default": 0.2}]
    if family in {"classification", "regression", "clustering"} and method == "suite":
        choices = [item["id"] for item in list_model_methods(family) if item["id"] != "suite"]
        controls.setdefault("task_options", []).append(
            {"name": "estimators", "type": "multi_select", "choices": choices, "required": True}
        )
    if family == "rbm":
        from ml_lab.energy_based.rbm.config import (
            BernoulliRBMSettings, GaussianRBMSettings, StudentTPoERBMSettings,
        )
        settings_cls = {"bernoulli": BernoulliRBMSettings, "gaussian": GaussianRBMSettings,
                        "student_t_poe": StudentTPoERBMSettings}[method]
        controls["task_options"] = [{"name": "hidden_dim", "type": "integer", "default": 8}]
        controls["model_config"] = [
            {"name": "sharpness", "type": "number", "default": 1.0},
            {"name": "dropout", "type": "number", "default": 0.0},
            {"name": "settings", "type": "object", "fields": _config_fields(settings_cls)},
        ]
    return {"schema": MODEL_SCHEMA, "family": family, "method": method,
            "task": family, "controls": controls}


def _rbm_model_options(method: str, options: dict[str, Any], n_features: int, hidden_dim: int) -> dict[str, Any]:
    from ml_lab.energy_based.rbm.config import (
        RBMConfig, BernoulliRBMSettings, GaussianRBMSettings, StudentTPoERBMSettings,
    )
    unknown = set(options) - {"sharpness", "dropout", "settings"}
    if unknown:
        raise ModelConfigurationError(f"Unknown rbm.model_config fields: {sorted(unknown)}")
    settings_cls = {"bernoulli": BernoulliRBMSettings, "gaussian": GaussianRBMSettings,
                    "student_t_poe": StudentTPoERBMSettings}[method]
    model = dict(options)
    if "sharpness" in model:
        model["sharpness"] = _typed(model["sharpness"], float, "model_config.sharpness")
    if "dropout" in model:
        model["dropout"] = _typed(model["dropout"], float, "model_config.dropout")
    if "settings" in model:
        model["settings"] = asdict(_dataclass_config(settings_cls, model["settings"], "model_config.settings"))
        # Retain only supplied settings: defaults are resolved by the RBM engine.
        model["settings"] = {k: model["settings"][k] for k in options["settings"]}
    try:
        config = RBMConfig(family=method, visible_dim=n_features, hidden_dim=hidden_dim,
                           sharpness=model.get("sharpness", 1.0), dropout=model.get("dropout", 0.0),
                           settings=model.get("settings"))
        config.resolved_settings()
    except (ValueError, TypeError) as exc:
        raise ModelConfigurationError(f"Invalid RBM model configuration: {exc}") from exc
    return model


def _suite_estimators(family: str, method: str, options: Mapping[str, Any]) -> list[str]:
    if method != "suite":
        return [method]
    chosen = options.get("estimators")
    if not isinstance(chosen, list) or not chosen or any(type(item) is not str for item in chosen):
        raise ModelConfigurationError("suite.estimators must be a nonempty array of estimator IDs")
    valid = {item["id"] for item in list_model_methods(family)} - {"suite"}
    if len(chosen) != len(set(chosen)) or set(chosen) - valid:
        raise ModelConfigurationError("suite.estimators contains duplicate or unknown estimator IDs")
    return chosen


def validate_model_configuration(
    family: str, method: str, settings: Mapping[str, Any] | None = None, *, n_features: int = 1,
) -> dict[str, Any]:
    """Canonical *explicit* settings in the existing execute_task payload shape.

    Defaults are left to the authoritative engines. Nothing is trained here.
    """
    _method_exists(family, method)
    if type(n_features) is not int or n_features < 1:
        raise ModelConfigurationError("n_features must be a positive integer")
    options = _strict_json_object({} if settings is None else settings, "settings")
    classes = _model_classes(family, method)
    allowed = set(classes)
    if family in {"classification", "regression"}:
        allowed.add("test_size")
    if family in {"classification", "regression", "clustering"} and method == "suite":
        allowed.add("estimators")
    if family == "rbm":
        allowed.update(("hidden_dim", "model_config"))
    unknown = set(options) - allowed
    if unknown:
        raise ModelConfigurationError(f"Unsupported {family} settings: {sorted(unknown)}")
    result: dict[str, Any] = {}
    for name, cls in classes.items():
        provided = options.get(name, {})
        # The unified experiment owns project-local output locations. Let the
        # S4 run/artifact lifecycle allocate checkpoint destinations rather
        # than exposing arbitrary paths in a persisted model specification.
        if name == "training_config" and isinstance(provided, Mapping) and provided.get("checkpoint_path") is not None:
            raise ModelConfigurationError("checkpoint_path is reserved for the S4 project-local run manager")
        validated = _dataclass_config(cls, provided, name)
        # Retain only the explicitly configured fields. Default values remain
        # authoritative in the model config class and are never serialized as NaN.
        record = asdict(validated)
        result[name] = _strict_json_object({key: record[key] for key in provided}, name)
    if family in {"classification", "regression"}:
        test_size = _typed(options.get("test_size", 0.2), float, "test_size")
        if not 0 < test_size < 1:
            raise ModelConfigurationError("test_size must be strictly between 0 and 1")
        result["test_size"] = test_size
        result["estimators"] = _suite_estimators(family, method, options)
    elif family == "clustering":
        result["estimators"] = _suite_estimators(family, method, options)
    elif family == "representation":
        result["method"] = method
    elif family == "rbm":
        hidden_dim = _typed(options.get("hidden_dim", 8), int, "hidden_dim")
        if hidden_dim < 1:
            raise ModelConfigurationError("hidden_dim must be positive")
        result["hidden_dim"] = hidden_dim
        result["family"] = method
        result["model_config"] = _rbm_model_options(method, _mapping(options.get("model_config", {}), "model_config"), n_features, hidden_dim)
    # GAN keeps both architecture controls independently configurable; no
    # synthetic-data creation is implied by selecting a GAN model.
    return result


def _targets_compatible(family: str, selected: Mapping[str, Any]) -> None:
    n_targets = len(selected.get("target_columns", []))
    requirement = _FAMILY_MAP[family][3]
    if (requirement == "one" and n_targets != 1) or (requirement == "none" and n_targets != 0) or (requirement == "optional_one" and n_targets > 1):
        raise ModelConfigurationError(f"{family} requires {requirement} selected target columns")


def configure_experiment_model(
    store: ExperimentStore, experiment_id: str, *, family: str, method: str,
    settings: Mapping[str, Any] | None = None, expected_revision: int,
) -> dict[str, Any]:
    """Publish a new model snapshot against the expected dataset revision.

    The SQLite compare-and-swap protects the final write from concurrent edits.
    """
    if type(expected_revision) is not int or expected_revision < 1:
        raise ModelConfigurationError("expected_revision must be a positive integer")
    current = store.get_experiment(experiment_id)
    if current["revision"] != expected_revision:
        raise ModelConfigurationError("Experiment revision changed; reload model configuration")
    if current["metadata"].get("adapter") == "ml-lab.recorded-legacy-task@1":
        raise ModelConfigurationError("Legacy-recorded tasks cannot be reconfigured as selected datasets")
    if not current["dataset_ref"] or current["dataset_ref"].get("schema") != "ml-lab.experiment-dataset@1":
        raise ModelConfigurationError("Select a Data Lab dataset before configuring a model")
    selected = verify_selection(store, current["dataset_ref"])
    if family not in _FAMILY_MAP:
        raise ModelConfigurationError(f"Unknown model family: {family!r}")
    _targets_compatible(family, selected)
    task_settings = validate_model_configuration(family, method, settings, n_features=len(selected["feature_columns"]))
    model_spec = {"schema": MODEL_SCHEMA, "family": family, "method": method,
                  "task": family, "settings": task_settings}
    if current["model_spec"] == model_spec:
        return current
    metadata = {**current["metadata"], "workflow": MODEL_SCHEMA, "model_status": "configured"}
    try:
        return store.revise_experiment(
            experiment_id, dataset_ref=selected, model_spec=model_spec,
            evaluation_plan={}, metadata=metadata, expected_revision=expected_revision,
        )
    except ExperimentStoreError as exc:
        raise ModelConfigurationError(str(exc)) from exc


def prepare_configured_model_handoff(
    store: ExperimentStore, experiment_id: str, *, revision: int | None = None,
) -> dict[str, Any]:
    """Materialize a pinned, validated legacy payload; does not execute it."""
    experiment = store.get_experiment(experiment_id, revision=revision)
    spec = experiment["model_spec"]
    if spec.get("schema") != MODEL_SCHEMA or spec.get("task") != spec.get("family"):
        raise ModelConfigurationError("Experiment does not have a configured stable model")
    family = spec.get("family")
    if family not in _FAMILY_MAP:
        raise ModelConfigurationError("Stored model family is unsupported")
    selected = verify_selection(store, experiment["dataset_ref"])
    _targets_compatible(family, selected)
    options = spec["settings"]
    # Recheck method/typed fields and reject tampered persisted snapshots. No
    # data inputs can be overridden by arbitrary model settings.
    original = validate_model_configuration(
        family, spec["method"], _restore_explicit_settings(family, spec["method"], options),
        n_features=len(selected["feature_columns"]),
    )
    if original != options:
        raise ModelConfigurationError("Stored model configuration is not canonical")
    handoff = prepare_model_handoff(store, experiment_id, family,
                                    revision=experiment["revision"], settings=options)
    return {**handoff, "schema": HANDOFF_SCHEMA, "model_spec": spec}


def _restore_explicit_settings(family: str, method: str, options: Mapping[str, Any]) -> dict[str, Any]:
    """Extract author-facing controls from the stored task adapter payload."""
    result = dict(options)
    for key in ("method", "family"):
        result.pop(key, None)
    if method != "suite":
        result.pop("estimators", None)
    # RBM hidden_dim and model_config remain author-facing controls.
    return result


__all__ = [
    "ModelConfigurationError", "MODEL_SCHEMA", "HANDOFF_SCHEMA",
    "list_model_families", "list_model_methods", "describe_model_method",
    "validate_model_configuration", "configure_experiment_model",
    "prepare_configured_model_handoff",
]
