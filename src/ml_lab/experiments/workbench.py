"""One active dataset binding per project-local ML_Lab workbench instance.

This is a *pointer* to an existing immutable experiment revision, never a
second dataset registry. A workbench may change its active revision without
modifying historical experiments, training runs, or artifacts. The service is
headless and does not depend on Flask, torch, or scikit-learn.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

from .datasets import DatasetSelectionError, select_experiment_dataset, verify_selection
from .store import ExperimentStore, ExperimentStoreError, _utcnow

WORKBENCH_SCHEMA = "ml-lab.workbench-instance@1"
_UUID_PATTERN = re.compile(r"[0-9a-f]{32}\Z")


class WorkbenchInstanceError(ExperimentStoreError):
    """Invalid or stale workbench selection."""


class WorkbenchInstanceNotFound(WorkbenchInstanceError):
    """The instance ID is not present in this project."""


class WorkbenchConflict(WorkbenchInstanceError):
    """A newer workbench generation has already been committed."""


def _instance_id(value: str) -> str:
    if not isinstance(value, str) or _UUID_PATTERN.fullmatch(value) is None:
        raise WorkbenchInstanceNotFound("Unknown workbench instance")
    return value


def _expected_generation(value: int) -> int:
    if type(value) is not int or value < 0:
        raise WorkbenchInstanceError("expected_generation must be a nonnegative integer")
    return value


def create_workbench_instance(store: ExperimentStore) -> dict[str, Any]:
    """Allocate a context in this project's existing experiment SQLite store."""
    identifier = uuid4().hex
    now = _utcnow()
    with store._connection(write=True) as db:
        db.execute(
            "INSERT INTO workbench_instances (instance_id, created_at, updated_at) VALUES (?, ?, ?)",
            (identifier, now, now),
        )
    return get_workbench_instance(store, identifier)


def get_workbench_instance(store: ExperimentStore, instance_id: str) -> dict[str, Any]:
    """Read a context and revalidate its pinned dataset before it can be used.

    Stale data are **not** silently reselected. A stale snapshot reports its
    original reference for display/repair but returns no trusted active dataset.
    """
    identifier = _instance_id(instance_id)
    with store._connection() as db:
        record = db.execute(
            "SELECT * FROM workbench_instances WHERE instance_id = ?", (identifier,)
        ).fetchone()
    if record is None:
        raise WorkbenchInstanceNotFound("Unknown workbench instance")
    response: dict[str, Any] = {
        "schema": WORKBENCH_SCHEMA,
        "instance_id": record["instance_id"],
        "generation": record["generation"],
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
        "experiment_id": record["experiment_id"],
        "experiment_revision": record["experiment_revision"],
        "status": "empty",
        "active_dataset": None,
        "pinned_dataset": None,
        "validation_error": None,
    }
    if record["experiment_id"] is not None:
        experiment = store.get_experiment(record["experiment_id"], revision=record["experiment_revision"])
        pinned = experiment["dataset_ref"]
        response["pinned_dataset"] = pinned
        try:
            response["active_dataset"] = verify_selection(store, pinned)
            response["status"] = "ready"
        except DatasetSelectionError as exc:
            response["status"] = "stale"
            response["validation_error"] = str(exc)
    return response


def _update_binding(
    store: ExperimentStore,
    instance_id: str,
    expected_generation: int,
    experiment_id: str | None,
    revision: int | None,
) -> dict[str, Any]:
    identifier = _instance_id(instance_id)
    generation = _expected_generation(expected_generation)
    now = _utcnow()
    with store._connection(write=True) as db:
        changed = db.execute(
            """UPDATE workbench_instances
               SET experiment_id = ?, experiment_revision = ?,
                   generation = generation + 1, updated_at = ?
               WHERE instance_id = ? AND generation = ?""",
            (experiment_id, revision, now, identifier, generation),
        )
        if changed.rowcount != 1:
            exists = db.execute(
                "SELECT generation FROM workbench_instances WHERE instance_id = ?", (identifier,)
            ).fetchone()
            if exists is None:
                raise WorkbenchInstanceNotFound("Unknown workbench instance")
            raise WorkbenchConflict("Workbench context changed; reload before switching datasets")
    return get_workbench_instance(store, identifier)


def activate_workbench_experiment(
    store: ExperimentStore,
    instance_id: str,
    experiment_id: str,
    *,
    expected_generation: int,
    revision: int | None = None,
) -> dict[str, Any]:
    """Bind an already-selected experiment revision without copying its data.

    A historical revision can be activated deliberately, but its CSV/provenance
    must still match the pinned identity. No other experiment is modified.
    """
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise WorkbenchInstanceError("experiment_id is required")
    if revision is not None and (type(revision) is not int or revision < 1):
        raise WorkbenchInstanceError("revision must be a positive integer")
    experiment = store.get_experiment(experiment_id, revision=revision)
    verify_selection(store, experiment["dataset_ref"])
    return _update_binding(
        store, instance_id, expected_generation, experiment["experiment_id"], experiment["revision"]
    )


def select_workbench_dataset(
    store: ExperimentStore,
    instance_id: str,
    path: str | Path,
    *,
    expected_generation: int,
    experiment_id: str | None = None,
    experiment_name: str | None = None,
    feature_columns: Sequence[str] | None = None,
    target_columns: Sequence[str] = (),
    expected_revision: int | None = None,
) -> dict[str, Any]:
    """Create/revise an experiment selection and activate it in this instance.

    The workbench generation is checked *before* creating the revision, as
    well as atomically at the final context switch. A concurrent change may
    leave an unbound experiment revision, but never overwrite an active
    dataset pointer or discard archived experiments/runs.
    """
    current = get_workbench_instance(store, instance_id)
    if current["generation"] != _expected_generation(expected_generation):
        raise WorkbenchConflict("Workbench context changed; reload before switching datasets")
    selected = select_experiment_dataset(
        store, path, experiment_id=experiment_id, experiment_name=experiment_name,
        feature_columns=feature_columns, target_columns=target_columns,
        expected_revision=expected_revision,
    )
    return activate_workbench_experiment(
        store, instance_id, selected["experiment_id"],
        expected_generation=expected_generation, revision=selected["revision"],
    )


def clear_workbench_dataset(
    store: ExperimentStore, instance_id: str, *, expected_generation: int
) -> dict[str, Any]:
    """Clear only the active pointer; historical experiments remain intact."""
    return _update_binding(store, instance_id, expected_generation, None, None)
