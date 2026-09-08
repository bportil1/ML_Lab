from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml_lab import classification, clustering, energy_based, generative, regression, representation
from ml_lab.core.serialization import to_jsonable


class PayloadError(ValueError):
    """Raised when a Flask/API payload cannot be mapped to an ML Lab task."""


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

def execute_task(task: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Execute a core task from a JSON-shaped payload without importing Flask."""
    if not isinstance(payload, dict):
        raise PayloadError("request payload must be a JSON object")
    runners = {
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
    if task in {"representation", "gan", "rbm"}:
        return {"task": task, "result": output}
    if task == "clustering":
        return {"task": task, **output}
    return {"task": task, "results": output}
