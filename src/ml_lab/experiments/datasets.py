"""Explicit project-local Data Lab dataset selection and model-input handoff.

S2.1 provides the headless data -> experiment bridge; it does not change
legacy dispatch, launch training, or introduce a PAH database dependency.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .store import ExperimentStore, ExperimentStoreError

_SELECTION_SCHEMA = "ml-lab.experiment-dataset@1"
_HANDOFF_SCHEMA = "ml-lab.dataset-model-handoff@1"
_MODEL_TASKS = frozenset({"classification", "regression", "clustering", "representation", "gan", "rbm"})
_RESERVED_KEYS = frozenset({"X", "y", "y_true", "feature_names", "label_name", "target_names"})


class DatasetSelectionError(ExperimentStoreError):
    """A source, selection, or pinned dataset revision is not safe to reuse."""


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _project_file(store: ExperimentStore, path: str | Path) -> tuple[Path, str]:
    # Keep exactly the same path containment and symlink policy as the
    # experiment artifact registry; do not resolve relative paths against CWD.
    try:
        return store.artifacts._project_file(path)
    except (ExperimentStoreError, OSError, ValueError) as exc:
        raise DatasetSelectionError("Dataset must be an existing, non-symlinked project-local file") from exc


def _columns(raw: Sequence[str] | None, all_columns: Sequence[str], *, label: str) -> list[str] | None:
    if raw is None:
        return None
    if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple)):
        raise DatasetSelectionError(f"{label} must be a list of column names")
    chosen = list(raw)
    if any(not isinstance(item, str) or not item for item in chosen):
        raise DatasetSelectionError(f"{label} must contain nonempty column names")
    if len(chosen) != len(set(chosen)):
        raise DatasetSelectionError(f"{label} contains duplicates")
    unknown = [item for item in chosen if item not in all_columns]
    if unknown:
        raise DatasetSelectionError(f"Unknown {label}: {unknown}")
    return chosen


def _verified_source(store: ExperimentStore, path: str | Path) -> tuple[Path, str, Any]:
    from ml_lab.data import inspect_file

    absolute, relative = _project_file(store, path)
    record = inspect_file(absolute)
    if (record.format != "delimited_text" or record.parse_status != "parsed"
            or record.malformed_row_count or not record.tabular_ready):
        raise DatasetSelectionError("Select a fully parsed CSV/TSV table; materialize XML or repair malformed rows first")
    if not record.columns or len(record.columns) != len(set(record.columns)):
        raise DatasetSelectionError("Dataset must have unique, nonempty column names")
    if not all(record.columns):
        raise DatasetSelectionError("Dataset has an empty column name")
    return absolute, relative, record


def _lineage(store: ExperimentStore, file: Path, identity: Mapping[str, Any]) -> dict[str, str] | None:
    from ml_lab.data import load_provenance_event, provenance_sidecar_path

    sidecar = provenance_sidecar_path(file)
    if not sidecar.exists() and not sidecar.is_symlink():
        return None
    _, relative = _project_file(store, sidecar)
    try:
        event = load_provenance_event(sidecar)
    except (OSError, ValueError, KeyError) as exc:
        raise DatasetSelectionError("Dataset provenance sidecar is unreadable or invalid") from exc
    derived = event.get("derived", {})
    if (not isinstance(derived, dict) or derived.get("sha256") != identity["sha256"]
            or derived.get("dataset_id") != identity["dataset_id"]):
        raise DatasetSelectionError("Dataset bytes or identity do not match their Data Lab provenance sidecar")
    event_id = event.get("event_id")
    if not isinstance(event_id, str) or not event_id:
        raise DatasetSelectionError("Dataset provenance event has no valid event ID")
    return {"event_id": event_id, "path": relative, "sha256": _digest(sidecar)}


def describe_selection(
    store: ExperimentStore, path: str | Path, *,
    feature_columns: Sequence[str] | None = None,
    target_columns: Sequence[str] = (),
) -> dict[str, Any]:
    """Pin the exact Data Lab table and an explicit column mapping.

    The selection is lightweight metadata; original CSV/TSV bytes stay in
    place. Derived datasets inherit Data Lab's own validated provenance.
    """
    from ml_lab.data import describe_dataset

    absolute, relative, inspected = _verified_source(store, path)
    original_sha = inspected.sha256
    try:
        identity = describe_dataset(absolute)
    except (OSError, ValueError) as exc:
        raise DatasetSelectionError("Dataset could not be read as a complete table") from exc
    if identity["sha256"] != original_sha or _digest(absolute) != original_sha:
        raise DatasetSelectionError("Dataset changed during selection")
    names = identity["column_names"]
    if len(names) != len(set(names)) or not names or identity["rows"] < 1:
        raise DatasetSelectionError("Dataset must have rows and unambiguous columns")
    targets = _columns(target_columns, names, label="target_columns") or []
    features = _columns(feature_columns, names, label="feature_columns")
    if features is None:
        features = [column for column in names if column not in targets]
    if not features or set(features).intersection(targets):
        raise DatasetSelectionError("Choose at least one feature and do not reuse target columns as features")
    lineage = _lineage(store, absolute, identity)
    return {
        "schema": _SELECTION_SCHEMA,
        "path": relative,
        "dataset_id": identity["dataset_id"],
        "sha256": identity["sha256"],
        "logical_sha256": identity["logical_sha256"],
        "rows": identity["rows"],
        "columns": identity["columns"],
        "column_names": names,
        "dtypes": identity["dtypes"],
        "feature_columns": features,
        "target_columns": targets,
        "provenance": lineage,
    }


def verify_selection(store: ExperimentStore, selection: Mapping[str, Any]) -> dict[str, Any]:
    """Revalidate a pinned selection; never silently refresh stale identity."""
    if not isinstance(selection, Mapping) or selection.get("schema") != _SELECTION_SCHEMA:
        raise DatasetSelectionError("Experiment has no S2.1 selected dataset reference")
    path = selection.get("path")
    if not isinstance(path, str) or not path or Path(path).is_absolute() or ".." in Path(path).parts:
        raise DatasetSelectionError("Dataset reference path must be project-relative")
    try:
        updated = describe_selection(
            store, path, feature_columns=selection.get("feature_columns"),
            target_columns=selection.get("target_columns", ()),
        )
    except (OSError, ValueError) as exc:
        if isinstance(exc, DatasetSelectionError):
            raise
        raise DatasetSelectionError("Selected dataset is unavailable or unreadable") from exc
    if dict(selection) != updated:
        raise DatasetSelectionError("Selected dataset, schema, or provenance has changed; select a new revision")
    return updated


def select_experiment_dataset(
    store: ExperimentStore, path: str | Path, *,
    experiment_id: str | None = None,
    experiment_name: str | None = None,
    feature_columns: Sequence[str] | None = None,
    target_columns: Sequence[str] = (),
    expected_revision: int | None = None,
) -> dict[str, Any]:
    """Create an experiment selection or publish a new immutable revision.

    Changing data invalidates the earlier model/evaluation configuration by
    making a new revision with those fields empty; historical runs remain.
    """
    if experiment_id is None and expected_revision is not None:
        raise DatasetSelectionError("expected_revision is only valid when revising an experiment")
    selected = describe_selection(store, path, feature_columns=feature_columns, target_columns=target_columns)
    if experiment_id is None:
        return store.create_experiment(
            experiment_name or f"Dataset: {Path(selected['path']).name}",
            dataset_ref=selected,
            metadata={"workflow": "ml-lab.dataset-selection@1", "model_status": "needs_configuration"},
        )
    current = store.get_experiment(experiment_id)
    if expected_revision is not None and current["revision"] != expected_revision:
        raise DatasetSelectionError("Experiment revision changed; reload before selecting a dataset")
    if current["metadata"].get("adapter") == "ml-lab.recorded-legacy-task@1":
        raise DatasetSelectionError("Cannot revise a legacy-recorded task into a dataset-selection experiment")
    if current["dataset_ref"] == selected:
        return current
    return store.revise_experiment(
        experiment_id,
        dataset_ref=selected,
        model_spec={},
        evaluation_plan={},
        metadata={"workflow": "ml-lab.dataset-selection@1", "model_status": "needs_configuration"},
        expected_revision=expected_revision,
    )


def preview_experiment_dataset(
    store: ExperimentStore, experiment_id: str, *, revision: int | None = None,
    page: int = 1, page_size: int = 50,
) -> dict[str, Any]:
    """Return a bounded Data Lab table page without changing selected data."""
    from ml_lab.data import read_table_page

    experiment = store.get_experiment(experiment_id, revision=revision)
    selected = verify_selection(store, experiment["dataset_ref"])
    page_data = read_table_page(store.project_root / selected["path"], page=page, page_size=page_size)
    return {
        "schema": "ml-lab.experiment-dataset-preview@1",
        "experiment_id": experiment_id,
        "revision": experiment["revision"],
        "dataset_ref": selected,
        "table": page_data,
    }


def _labels(frame: Any, column: str) -> list[Any]:
    import pandas as pd
    values = frame[column].tolist()
    if any(pd.isna(value) for value in values):
        raise DatasetSelectionError(f"Target {column!r} contains missing values")
    result: list[Any] = []
    for value in values:
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            raise DatasetSelectionError(f"Target {column!r} contains nonfinite values")
        if not isinstance(value, (str, int, float, bool)):
            raise DatasetSelectionError(f"Target {column!r} has unsupported label type")
        result.append(value)
    return result


def prepare_model_handoff(
    store: ExperimentStore, experiment_id: str, task: str, *,
    revision: int | None = None, settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load selected features/targets into the current legacy task input shape.

    No model runs here. S3 consumes this explicit handoff and connects its
    execution lifecycle to the original experiment revision.
    """
    if task not in _MODEL_TASKS:
        raise DatasetSelectionError(f"Unsupported tabular model task: {task}")
    settings = {} if settings is None else settings
    if (not isinstance(settings, Mapping) or any(not isinstance(key, str) for key in settings)
            or any(key in settings for key in _RESERVED_KEYS)):
        raise DatasetSelectionError("Model settings must not replace selected X/y/column mapping and must use string keys")
    experiment = store.get_experiment(experiment_id, revision=revision)
    selected = verify_selection(store, experiment["dataset_ref"])
    targets = selected["target_columns"]
    if task in {"classification", "regression"} and len(targets) != 1:
        raise DatasetSelectionError(f"{task} requires exactly one selected target column in S2.1")
    if task == "clustering" and len(targets) > 1:
        raise DatasetSelectionError("Clustering supports at most one reference-label column")
    if task in {"representation", "gan", "rbm"} and targets:
        raise DatasetSelectionError(f"{task} does not accept selected targets")

    from ml_lab.data import read_dataset_frame
    import numpy as np
    import pandas as pd

    path = store.project_root / selected["path"]
    try:
        frame = read_dataset_frame(path)
        numeric = frame.loc[:, selected["feature_columns"]].apply(pd.to_numeric, errors="raise")
        values = numeric.to_numpy(dtype=float)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        raise DatasetSelectionError("Selected feature columns must contain numeric values") from exc
    if values.ndim != 2 or not np.isfinite(values).all():
        raise DatasetSelectionError("Selected features contain missing or nonfinite numeric values")
    payload: dict[str, Any] = {**dict(settings), "X": values.tolist(),
                               "feature_names": selected["feature_columns"]}
    if targets:
        if task == "regression":
            try:
                y = pd.to_numeric(frame[targets[0]], errors="raise").to_numpy(dtype=float)
            except (ValueError, TypeError) as exc:
                raise DatasetSelectionError("Regression target must be numeric") from exc
            if not np.isfinite(y).all():
                raise DatasetSelectionError("Regression target contains missing or nonfinite values")
            payload["y"] = y.tolist()
            payload["target_names"] = targets
        else:
            payload["y" if task == "classification" else "y_true"] = _labels(frame, targets[0])
            if task == "classification":
                payload["label_name"] = targets[0]
    # A second verification prevents a changed source from being silently used
    # between the initial validation and full in-memory materialization.
    verify_selection(store, selected)
    return {
        "schema": _HANDOFF_SCHEMA,
        "experiment_id": experiment_id,
        "revision": experiment["revision"],
        "task": task,
        "dataset_ref": selected,
        "payload": payload,
    }


__all__ = [
    "DatasetSelectionError", "describe_selection", "verify_selection",
    "select_experiment_dataset", "preview_experiment_dataset", "prepare_model_handoff",
]
