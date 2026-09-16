from __future__ import annotations

from copy import deepcopy
from importlib.util import find_spec
from typing import Any


_TASKS: tuple[dict[str, Any], ...] = (
    {
        "id": "data",
        "name": "Data Lab",
        "group": "Data",
        "summary": "Unknown-data intake and inventory, with profiling and transformation added in later sprints.",
        "status": "ready",
        "task": "data.inspect",
        "ui_route": "data_lab",
        "requires": (),
        "note": "A1 inventories CSV/TSV inputs without modifying them; A2 will add statistical profiling.",
    },
    {
        "id": "classification",
        "name": "Classification",
        "group": "Models",
        "summary": "Supervised classification search, evaluation, and selection.",
        "status": "ready",
        "task": "classification",
        "requires": (),
        "example": {
            "X": [[0.0, 0.0], [0.1, 0.2], [0.2, 0.1], [0.3, 0.2], [1.0, 1.0], [1.1, 1.2], [1.2, 1.1], [1.3, 1.2]],
            "y": [0, 0, 0, 0, 1, 1, 1, 1],
            "feature_names": ["x1", "x2"],
            "estimators": ["ridge"],
            "config": {"cv_folds": 2, "n_jobs": 1},
            "test_size": 0.25,
        },
    },
    {
        "id": "regression",
        "name": "Regression",
        "group": "Models",
        "summary": "Single- or multi-output regression search and holdout evaluation.",
        "status": "ready",
        "task": "regression",
        "requires": (),
        "example": {
            "X": [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
            "y": [0.1, 1.1, 1.9, 3.2, 3.9, 5.1],
            "feature_names": ["x"],
            "estimators": ["ridge"],
            "config": {"cv_folds": 2, "n_jobs": 1},
            "test_size": 0.33,
        },
    },
    {
        "id": "clustering",
        "name": "Clustering",
        "group": "Models",
        "summary": "Candidate clustering, stability, consensus, and algorithm agreement.",
        "status": "ready",
        "task": "clustering",
        "requires": (),
        "example": {
            "X": [[0.0, 0.0], [0.1, 0.2], [5.0, 5.0], [5.1, 5.2]],
            "feature_names": ["x", "y"],
            "estimators": ["kmeans"],
            "config": {"repeats": 1, "random_state": 7},
        },
    },
    {
        "id": "representation",
        "name": "Representation",
        "group": "Representation",
        "summary": "PCA plus optional neural MLP/Transformer autoencoder representations.",
        "status": "ready",
        "task": "representation",
        "requires": (),
        "note": "PCA is available in the base install; neural methods require the neural extra.",
        "example": {
            "X": [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
            "feature_names": ["a", "b", "c"],
            "method": "pca",
            "config": {"n_components": 2},
        },
    },
    {
        "id": "gan",
        "name": "Generative / GAN",
        "group": "Generative",
        "summary": "Generic MLP or Transformer GAN training for numeric tables.",
        "status": "ready",
        "task": "gan",
        "requires": ("torch",),
        "note": "Requires the neural extra.",
        "example": {
            "X": [[0.0, 1.0], [1.0, 0.0], [0.2, 0.8], [0.8, 0.2]],
            "feature_names": ["a", "b"],
            "model_config": {"latent_dim": 2, "generator_hidden_dims": [8], "discriminator_hidden_dims": [8]},
            "training_config": {"epochs": 1, "batch_size": 2, "generated_sample_count": 4, "device": "cpu"},
        },
    },
    {
        "id": "rbm",
        "name": "Energy Based / RBM",
        "group": "Energy Based",
        "summary": "Stable Bernoulli, Gaussian, and Student-t Product-of-Experts RBM training.",
        "status": "ready",
        "task": "rbm",
        "requires": ("torch",),
        "note": "Requires the energy or neural extra.",
        "example": {
            "X": [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
            "feature_names": ["a", "b"],
            "family": "bernoulli",
            "hidden_dim": 2,
            "training_config": {"epochs": 1, "batch_size": 2, "device": "cpu"},
        },
    },
    {
        "id": "optimization",
        "name": "Optimization",
        "group": "Optimization",
        "summary": "Task-neutral search-space, evaluator, and optimizer infrastructure.",
        "status": "library",
        "task": None,
        "requires": (),
        "note": "Stable Python API; a generic visual runner will follow a typed task contract.",
    },
    {
        "id": "experimental",
        "name": "Experimental",
        "group": "Experimental",
        "summary": "Explicitly isolated research modules and incubating training strategies.",
        "status": "experimental",
        "task": None,
        "requires": (),
        "note": "Hidden unless the host or standalone launcher explicitly enables experimental UI.",
    },
)


def _dependency_available(module: str) -> bool:
    return find_spec(module) is not None


def capability_records(*, enable_experimental: bool = False) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw in _TASKS:
        if raw["id"] == "experimental" and not enable_experimental:
            continue
        item = deepcopy(raw)
        missing = [name for name in item.get("requires", ()) if not _dependency_available(name)]
        item["requires"] = list(item.get("requires", ()))
        item["missing_dependencies"] = missing
        item["available"] = not missing and item["status"] not in {"planned"}
        if missing:
            item["effective_status"] = "dependency_missing"
        else:
            item["effective_status"] = item["status"]
        records.append(item)
    return records


def get_capability(capability_id: str, *, enable_experimental: bool = False) -> dict[str, Any]:
    for record in capability_records(enable_experimental=enable_experimental):
        if record["id"] == capability_id:
            return record
    raise KeyError(capability_id)


def capability_groups(*, enable_experimental: bool = False) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in capability_records(enable_experimental=enable_experimental):
        grouped.setdefault(record["group"], []).append(record)
    return [{"name": name, "capabilities": values} for name, values in grouped.items()]
