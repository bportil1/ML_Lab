from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml_lab import classification, clustering, data, energy_based, generative, regression, representation
from ml_lab.core.serialization import to_jsonable


class PayloadError(ValueError):
    """Raised when an application payload cannot be mapped to an ML Lab task."""


def _features(payload: dict[str, Any]) -> pd.DataFrame:
    if "X" not in payload:
        raise PayloadError("payload must contain X")
    values = np.asarray(payload["X"])
    if values.ndim != 2:
        raise PayloadError("X must be a two-dimensional array")
    names = payload.get("feature_names")
    if names is not None and len(names) != values.shape[1]:
        raise PayloadError("feature_names length must match the number of X columns")
    return pd.DataFrame(values, columns=names)


def _classification(payload: dict[str, Any]) -> list[dict[str, Any]]:
    X = _features(payload)
    if "y" not in payload:
        raise PayloadError("classification payload must contain y")
    y = pd.Series(payload["y"], name=payload.get("label_name", "label"))
    if len(y) != len(X):
        raise PayloadError("X and y must contain the same number of rows")
    config = classification.ClassificationSearchConfig(**payload.get("config", {}))
    results = classification.run(
        X,
        y,
        estimators=payload.get("estimators", ("rf", "hgbc", "lda", "qda", "ridge")),
        config=config,
        test_size=float(payload.get("test_size", 0.2)),
        label_name=payload.get("label_name", "label"),
    )
    return [to_jsonable(result.to_record()) for result in results]


def _regression(payload: dict[str, Any]) -> list[dict[str, Any]]:
    X = _features(payload)
    if "y" not in payload:
        raise PayloadError("regression payload must contain y")
    values = np.asarray(payload["y"])
    target_names = payload.get("target_names")
    if values.ndim == 1:
        y: pd.Series | pd.DataFrame = pd.Series(values, name=(target_names or ["target"])[0])
    elif values.ndim == 2:
        if target_names is not None and len(target_names) != values.shape[1]:
            raise PayloadError("target_names length must match the number of y columns")
        y = pd.DataFrame(values, columns=target_names)
    else:
        raise PayloadError("regression y must be one- or two-dimensional")
    if len(y) != len(X):
        raise PayloadError("X and y must contain the same number of rows")
    config = regression.RegressionSearchConfig(**payload.get("config", {}))
    results = regression.run(
        X,
        y,
        estimators=payload.get("estimators", ("ridge", "random_forest", "extra_trees", "svr", "knn")),
        config=config,
        test_size=float(payload.get("test_size", 0.2)),
        target_names=target_names,
    )
    return [to_jsonable(result.to_record()) for result in results]


def _clustering(payload: dict[str, Any]) -> dict[str, Any]:
    X = _features(payload)
    y_true = payload.get("y_true")
    if y_true is not None:
        y_true = np.asarray(y_true)
        if len(y_true) != len(X):
            raise PayloadError("X and y_true must contain the same number of rows")
    config = clustering.ClusteringSearchConfig(**payload.get("config", {}))
    results = clustering.run(
        X,
        estimators=payload.get("estimators", ("kmeans", "agglomerative", "spectral", "birch", "gmm")),
        config=config,
        y_true=y_true,
    )
    agreement = clustering.analyze_agreement(
        results,
        ignore_noise=bool(payload.get("agreement_ignore_noise", config.stability_ignore_noise)),
    )
    return {
        "results": [to_jsonable(result.to_record()) for result in results],
        "algorithm_agreement": to_jsonable(agreement.summary_record()),
        "algorithm_agreement_pairs": to_jsonable(agreement.pairwise_records),
    }


def _representation(payload: dict[str, Any]) -> dict[str, Any]:
    method = payload.get("method", "pca")
    if method in {"pca", "mlp_autoencoder"}:
        X: Any = _features(payload)
    else:
        if "X" not in payload:
            raise PayloadError("payload must contain X")
        values = np.asarray(payload["X"], dtype=float)
        if values.ndim not in (2, 3):
            raise PayloadError("Transformer representation X must be two- or three-dimensional")
        names = payload.get("feature_names")
        if values.ndim == 2 and names is not None:
            if len(names) != values.shape[1]:
                raise PayloadError("feature_names length must match the number of X columns")
            X = pd.DataFrame(values, columns=names)
        else:
            X = values
    if method == "pca":
        result = representation.pca(
            X,
            config=representation.PCARepresentationConfig(**payload.get("config", {})),
        )
    elif method == "mlp_autoencoder":
        result = representation.autoencode(
            X,
            model_config=representation.MLPAutoencoderConfig(**payload.get("model_config", {})),
            training_config=representation.AutoencoderTrainingConfig(**payload.get("training_config", {})),
        )
    elif method == "transformer_autoencoder":
        result = representation.transformer_autoencode(
            X,
            model_config=representation.TransformerAutoencoderConfig(**payload.get("model_config", {})),
            training_config=representation.AutoencoderTrainingConfig(**payload.get("training_config", {})),
        )
    else:
        raise PayloadError(f"unsupported representation method: {method}")
    record = to_jsonable(result.to_record())
    record["latent"] = to_jsonable(result.latent)
    record["reconstruction"] = to_jsonable(result.reconstruction)
    return record



def _gan(payload: dict[str, Any]) -> dict[str, Any]:
    X = _features(payload)
    result = generative.gan.run(
        X,
        model_config=generative.gan.GANModelConfig(**payload.get("model_config", {})),
        training_config=generative.gan.GANTrainingConfig(**payload.get("training_config", {})),
    )
    record = to_jsonable(result.to_record())
    record["generated_samples"] = to_jsonable(result.generated_samples)
    return record


def _rbm(payload: dict[str, Any]) -> dict[str, Any]:
    X = _features(payload)
    family = payload.get("family", "bernoulli")
    hidden_dim = int(payload.get("hidden_dim", 8))
    result = energy_based.training.run(
        X,
        family=family,
        hidden_dim=hidden_dim,
        model_config=payload.get("model_config", {}),
        training_config=payload.get("training_config", {}),
        feature_names=[str(column) for column in X.columns],
        verbose=bool(payload.get("verbose", False)),
    )
    record = to_jsonable(result.to_record())
    if result.generated_samples is not None:
        record["generated_samples"] = to_jsonable(result.generated_samples)
    return record


def _data_inspect(payload: dict[str, Any]) -> dict[str, Any]:
    paths = payload.get("paths")
    if isinstance(paths, (str, bytes)):
        paths = [paths]
    if not isinstance(paths, (list, tuple)) or not paths:
        raise PayloadError("data.inspect payload must contain a non-empty paths list")
    inventory = data.inspect_paths(
        [str(path) for path in paths],
        recursive=bool(payload.get("recursive", True)),
        include_hidden=bool(payload.get("include_hidden", False)),
        preview_rows=int(payload.get("preview_rows", 20)),
    )
    return inventory.to_record()


def _data_profile(payload: dict[str, Any]) -> dict[str, Any]:
    paths = payload.get("paths")
    if isinstance(paths, (str, bytes)):
        paths = [paths]
    if not isinstance(paths, (list, tuple)) or not paths:
        raise PayloadError("data.profile payload must contain a non-empty paths list")
    max_rows_raw = payload.get("max_rows", 100_000)
    max_rows = None if max_rows_raw in (None, 0, "0") else int(max_rows_raw)
    profile = data.profile_paths(
        [str(path) for path in paths],
        recursive=bool(payload.get("recursive", True)),
        include_hidden=bool(payload.get("include_hidden", False)),
        preview_rows=int(payload.get("preview_rows", 20)),
        max_rows=max_rows,
        relationship_rows=int(payload.get("relationship_rows", 5_000)),
        max_relationship_columns=int(payload.get("max_relationship_columns", 25)),
        max_relationship_pairs=int(payload.get("max_relationship_pairs", 200)),
        outlier_iqr_multiplier=float(payload.get("outlier_iqr_multiplier", 1.5)),
        random_state=int(payload.get("random_state", 42)),
    )
    return profile.to_record()



def _data_compare(payload: dict[str, Any]) -> dict[str, Any]:
    paths = payload.get("paths")
    if isinstance(paths, (str, bytes)):
        paths = [paths]
    if not isinstance(paths, (list, tuple)) or not paths:
        raise PayloadError("data.compare payload must contain a non-empty paths list")
    max_pairs_raw = payload.get("max_pairs", 200)
    max_pairs = None if max_pairs_raw in (None, 0, "0") else int(max_pairs_raw)
    return data.compare_paths(
        [str(path) for path in paths],
        recursive=bool(payload.get("recursive", True)),
        include_hidden=bool(payload.get("include_hidden", False)),
        max_pairs=max_pairs,
    )

def _data_table(payload: dict[str, Any]) -> dict[str, Any]:
    path = payload.get("path")
    if not isinstance(path, (str, bytes)) or not str(path):
        raise PayloadError("data.table payload must contain path")
    raw_page_size = payload.get("page_size", 50)
    page_size = None if raw_page_size in (None, "all", "All", 0, "0") else int(raw_page_size)
    filters = payload.get("filters", {})
    if not isinstance(filters, dict):
        raise PayloadError("data.table filters must be an object")
    return data.read_table_page(
        str(path),
        page=int(payload.get("page", 1)),
        page_size=page_size,
        search=str(payload.get("search", "")),
        filters={str(key): str(value) for key, value in filters.items()},
        sort_column=(None if payload.get("sort_column") in (None, "") else str(payload.get("sort_column"))),
        sort_direction=str(payload.get("sort_direction", "asc")),
    )


def _data_transform_preview(payload: dict[str, Any]) -> dict[str, Any]:
    path = payload.get("path")
    if not isinstance(path, (str, bytes)) or not str(path):
        raise PayloadError("data.transform.preview payload must contain path")
    recipe = payload.get("recipe", {})
    if not isinstance(recipe, dict):
        raise PayloadError("data.transform.preview recipe must be an object")
    return data.preview_transformation(
        str(path),
        recipe,
        preview_rows=int(payload.get("preview_rows", 50)),
    )



def _data_lineage(payload: dict[str, Any]) -> dict[str, Any]:
    path = payload.get("path")
    if not isinstance(path, (str, bytes)) or not str(path):
        raise PayloadError("data.lineage payload must contain path")
    return data.trace_lineage(str(path), max_depth=int(payload.get("max_depth", 100)))


def _data_transform_apply(payload: dict[str, Any]) -> dict[str, Any]:
    path = payload.get("path")
    if not isinstance(path, (str, bytes)) or not str(path):
        raise PayloadError("data.transform.apply payload must contain path")
    recipe = payload.get("recipe", {})
    if not isinstance(recipe, dict):
        raise PayloadError("data.transform.apply recipe must be an object")
    output = payload.get("output")
    return data.apply_transformation(
        str(path),
        recipe,
        output=(None if output in (None, "") else str(output)),
        overwrite=bool(payload.get("overwrite", False)),
        preview_rows=int(payload.get("preview_rows", 50)),
    )

def execute_task(task: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Execute a core task from a JSON-shaped payload without importing Flask."""
    if not isinstance(payload, dict):
        raise PayloadError("request payload must be a JSON object")
    runners = {
        "data.inspect": _data_inspect,
        "data.profile": _data_profile,
        "data.compare": _data_compare,
        "data.table": _data_table,
        "data.transform.preview": _data_transform_preview,
        "data.transform.apply": _data_transform_apply,
        "data.lineage": _data_lineage,
        "classification": _classification,
        "regression": _regression,
        "clustering": _clustering,
        "representation": _representation,
        "gan": _gan,
        "rbm": _rbm,
    }
    try:
        runner = runners[task]
    except KeyError as exc:
        raise PayloadError(f"unsupported ML Lab task: {task}") from exc
    output = runner(payload)
    if task in {"data.inspect", "data.profile", "data.compare", "data.table", "data.transform.preview", "data.transform.apply", "data.lineage", "representation", "gan", "rbm"}:
        return {"task": task, "result": output}
    if task == "clustering":
        return {"task": task, **output}
    return {"task": task, "results": output}
