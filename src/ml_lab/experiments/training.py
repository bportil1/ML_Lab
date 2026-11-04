"""S4.1: opt-in execution of a pinned, configured ML_Lab experiment.

The six model families still run through the authoritative ``execute_task``
engine.  This module owns only the experiment/run lifecycle, input provenance,
result artifacts and coarse-grained observable stages.  It intentionally does
not claim epoch-level progress, cancellation, resume, or checkpoint allocation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Callable
from uuid import uuid4

from .models import MODEL_SCHEMA, prepare_configured_model_handoff
from .store import ExperimentStore, ExperimentStoreError
from .run_control import (RunCancelled, check_cancelled, cancellation_requested, write_owner)

RUN_PROGRESS_SCHEMA = "ml-lab.configured-run-progress@1"
RUN_RESULT_SCHEMA = "ml-lab.configured-run@1"
_RUN_ID = re.compile(r"^[0-9a-f]{32}$")


class ConfiguredRunError(ExperimentStoreError):
    """An experiment run was recorded as failed; its run ID is preserved."""

    def __init__(self, message: str, *, run_id: str) -> None:
        super().__init__(message)
        self.run_id = run_id


def _run_folder(store: ExperimentStore, run_id: str, *, create: bool) -> Path:
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise ExperimentStoreError("Invalid run identifier")
    current = store.project_root
    for part in ("analysis", "ml_lab", "training_runs"):
        current = current / part
        if current.is_symlink():
            raise ExperimentStoreError("Refusing symlinked training directory")
        if create:
            current.mkdir(exist_ok=True)
    folder = current / run_id
    if folder.is_symlink():
        raise ExperimentStoreError("Refusing symlinked run directory")
    if create:
        folder.mkdir(exist_ok=False)
    return folder


def _json_bytes(value: Any) -> bytes:
    from ml_lab.core.serialization import to_jsonable

    try:
        return (json.dumps(to_jsonable(value), sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
    except (TypeError, ValueError, RecursionError, OverflowError) as exc:
        raise ExperimentStoreError("Experiment result cannot be persisted as strict JSON") from exc


def _write_new(path: Path, value: Any) -> None:
    with path.open("xb") as output:
        output.write(_json_bytes(value))
        output.flush()
        os.fsync(output.fileno())


def _stage(store: ExperimentStore, run_id: str, label: str, step: int, *,
           status: str = "running", notify: Callable[[dict[str, Any]], None] | None = None) -> None:
    """Best-effort presentation stage; SQLite remains authoritative for status."""
    folder = _run_folder(store, run_id, create=False)
    record = {
        "schema": RUN_PROGRESS_SCHEMA, "run_id": run_id, "stage": label,
        "step": step, "steps": 4, "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary = folder / f".progress-{uuid4().hex}.tmp"
    try:
        _write_new(temporary, record)
        os.replace(temporary, folder / "progress.json")
    finally:
        temporary.unlink(missing_ok=True)
    if notify is not None:
        notify(dict(record))


def get_configured_run_status(store: ExperimentStore, run_id: str) -> dict[str, Any]:
    """A persistently queryable run record plus last known execution stage."""
    run = store.get_run(run_id)  # Validate project ownership before filesystem access.
    stage = None
    path = _run_folder(store, run_id, create=False) / "progress.json"
    if path.is_file() and not path.is_symlink():
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("schema") == RUN_PROGRESS_SCHEMA and record.get("run_id") == run_id:
                stage = record
        except (OSError, ValueError, TypeError):
            pass
    return {
        "schema": RUN_PROGRESS_SCHEMA,
        "run": run,
        "stage": stage["stage"] if stage else ("Queued" if run["status"] == "queued" else run["status"].title()),
        "step": stage["step"] if stage else 0,
        "steps": 4,
        "last_progress_at": stage.get("updated_at") if stage else None,
        "artifacts": store.artifacts.list_for_run(run_id),
        "cancellation_requested": cancellation_requested(store, run_id) if run["status"] == "running" else False,
    }


def queue_configured_run(
    store: ExperimentStore, experiment_id: str, *, revision: int | None = None,
    expected_revision: int | None = None, seed: int | None = None,
) -> dict[str, Any]:
    """Queue one run bound to a specific immutable experiment revision.

    Model execution is *not* started here; either a standalone caller or a UI
    worker must invoke :func:`execute_configured_run` with this run ID.
    """
    experiment = store.get_experiment(experiment_id, revision=revision)
    if expected_revision is not None:
        current = store.get_experiment(experiment_id)
        if current["revision"] != expected_revision or experiment["revision"] != expected_revision:
            raise ExperimentStoreError("Experiment revision changed; reload before starting training")
    if experiment["model_spec"].get("schema") != MODEL_SCHEMA:
        raise ExperimentStoreError("Select and save a stable model configuration before training")
    # Verify selected dataset identity, including its Data Lab lineage, without
    # materializing the entire model payload in the UI request thread.
    from .datasets import verify_selection
    verify_selection(store, experiment["dataset_ref"])
    return store.create_run(
        experiment_id, revision=experiment["revision"], seed=seed,
        environment={"executor": "ml_lab.experiments.training@1", "progress": "stage-only",
                     "seed_note": "recorded provenance, not an override of model random state"},
        expected_current_revision=expected_revision,
    )


def execute_configured_run(
    store: ExperimentStore, run_id: str, *,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Execute a queued run synchronously; suitable for a UI worker or CLI.

    An exception marks the recorded run as failed; successful runs link source,
    configuration and result artifacts inside the owning project.
    """
    run = store.get_run(run_id)
    if run["status"] != "queued":
        raise ExperimentStoreError("Only queued configured runs may start")
    experiment_id, revision = run["experiment_id"], run["revision"]
    experiment = store.get_experiment(experiment_id, revision=revision)
    if experiment["model_spec"].get("schema") != MODEL_SCHEMA:
        raise ExperimentStoreError("Run does not reference a configured model")

    # Claim the queued run first.  A competing worker must not launch another
    # execution for the same run ID (SQLite serializes state transitions).
    store.update_run_status(run_id, "running")
    artifacts: list[dict[str, Any]] = []
    try:
        folder = _run_folder(store, run_id, create=True)
        write_owner(store, run_id)
        check_cancelled(store, run_id)
        _stage(store, run_id, "Validating pinned dataset and model", 1, notify=progress)
        handoff = prepare_configured_model_handoff(store, experiment_id, revision=revision)
        # Never duplicate the entire feature matrix into the metadata catalog.
        manifest = {
            "schema": "ml-lab.configured-run-input@1", "experiment_id": experiment_id,
            "revision": revision, "run_id": run_id, "dataset_ref": experiment["dataset_ref"],
            "model_spec": experiment["model_spec"], "task": handoff["task"],
            "seed": run["seed"],
        }
        manifest_path = folder / "input.json"
        _write_new(manifest_path, manifest)
        selected = handoff["dataset_ref"]
        dataset_file = store.project_root / selected["path"]
        source = store.artifacts.register(
            run_id, dataset_file, role="input", kind="dataset",
            metadata={"dataset_id": selected["dataset_id"], "expected_sha256": selected["sha256"]},
        )
        if source["sha256"] != selected["sha256"]:
            raise ExperimentStoreError("Source dataset changed during input registration")
        artifacts.append(source)
        if selected.get("provenance"):
            provenance = store.artifacts.register(
                run_id, store.project_root / selected["provenance"]["path"],
                role="input", kind="provenance",
                metadata={"data_lab_event_id": selected["provenance"]["event_id"]},
            )
            if provenance["sha256"] != selected["provenance"]["sha256"]:
                raise ExperimentStoreError("Data Lab provenance changed during input registration")
            artifacts.append(provenance)
        manifest_artifact = store.artifacts.register(
            run_id, manifest_path, role="input", kind="manifest", media_type="application/json",
            source_artifact_ids=[item["artifact_id"] for item in artifacts],
        )
        artifacts.append(manifest_artifact)
        _stage(store, run_id, "Executing model (engine progress unavailable)", 2, notify=progress)
        from ml_lab.application import execute_task  # Import heavy model engines lazily.
        check_cancelled(store, run_id)
        response = execute_task(handoff["task"], handoff["payload"])
        check_cancelled(store, run_id)
        _stage(store, run_id, "Checking source and persisting results", 3, notify=progress)
        from .datasets import verify_selection
        verify_selection(store, selected)
        if store.artifacts.verify(source["artifact_id"])["status"] != "valid":
            raise ExperimentStoreError("Source dataset changed during model execution")
        result_path = folder / "result.json"
        _write_new(result_path, response)
        artifacts.append(store.artifacts.register(
            run_id, result_path, role="output", kind="result", media_type="application/json",
            source_artifact_ids=[item["artifact_id"] for item in artifacts if item["role"] == "input"],
        ))
        check_cancelled(store, run_id)
        _stage(store, run_id, "Model run completed", 4, status="succeeded", notify=progress)
        store.update_run_status(run_id, "succeeded")
    except RunCancelled as exc:
        store.update_run_status(run_id, "cancelled")
        _stage(store, run_id, "Run cancelled at a safe boundary", 4, status="cancelled", notify=progress)
        raise ConfiguredRunError(f"Run cancelled (run={run_id})", run_id=run_id) from exc
    except Exception as exc:
        # Never disguise a failed execution or incomplete artifact registration.
        store.update_run_status(run_id, "failed", error=f"{type(exc).__name__}: {exc}"[:2000])
        try:
            _stage(store, run_id, "Model run failed", 4, status="failed", notify=progress)
        except (ExperimentStoreError, OSError):
            pass  # Keep the authoritative SQLite failure regardless of presentation errors.
        raise ConfiguredRunError(f"Model run failed (run={run_id})", run_id=run_id) from exc
    return {
        "schema": RUN_RESULT_SCHEMA, "experiment_id": experiment_id,
        "revision": revision, "run_id": run_id, "status": "succeeded", "artifacts": artifacts,
    }


def run_configured_experiment(
    store: ExperimentStore, experiment_id: str, *, revision: int | None = None,
    seed: int | None = None, progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Convenience synchronous entry point for standalone callers."""
    queued = queue_configured_run(store, experiment_id, revision=revision, seed=seed)
    return execute_configured_run(store, queued["run_id"], progress=progress)


__all__ = [
    "ConfiguredRunError", "RUN_PROGRESS_SCHEMA", "RUN_RESULT_SCHEMA",
    "queue_configured_run", "execute_configured_run", "run_configured_experiment",
    "get_configured_run_status",
]
