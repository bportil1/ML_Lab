"""Opt-in recording adapter for existing ML_Lab application tasks (S1.3).

This module does not replace ``ml_lab.application.execute_task``. A recorded
call executes that function exactly once and persists JSON-shaped inputs and
results as project-local files, registered using the S1.2 artifact registry.
No PAH/Flask dependency, model runner modification, or implicit auto-tracking.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import _hash_file
from .store import ExperimentStore, ExperimentStoreError


class RecordedTaskError(ExperimentStoreError):
    """A recorded invocation failed after a run was created.

    ``run_id`` and ``experiment_id`` identify the persisted failed run. The
    original error is available through normal Python exception chaining.
    """

    def __init__(self, message: str, *, experiment_id: str, run_id: str) -> None:
        super().__init__(message)
        self.experiment_id = experiment_id
        self.run_id = run_id


def _json_default(value: Any) -> Any:
    # Legacy execute_task accepts NumPy arrays/scalars. Convert them to JSON
    # without relying on core.serialization's intentionally lossy fallback
    # (which stringifies unknown objects and replaces nonfinite floats).
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON-compatible: {type(value).__name__}")


def _json_bytes(value: Any) -> bytes:
    try:
        return (json.dumps(value, ensure_ascii=False, allow_nan=False,
                           sort_keys=True, separators=(",", ":"), default=_json_default) + "\n").encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ExperimentStoreError("Recorded task input/output must be strict JSON-compatible data") from exc


def _snapshot(
    store: ExperimentStore, task: str, payload: dict[str, Any], payload_bytes: bytes,
) -> dict[str, dict[str, Any]]:
    digest = hashlib.sha256(payload_bytes).hexdigest()
    # Experiment metadata stores a *reference* to the complete input snapshot,
    # never a duplicate of feature/target arrays inside SQLite.
    dataset_ref: dict[str, Any] = {"kind": "legacy_task_payload", "sha256": digest}
    if "X" in payload:
        data = payload["X"]
        dataset_ref["inline_features"] = True
        dataset_ref["row_count"] = len(data) if isinstance(data, list) else None
    elif "path" in payload or "paths" in payload:
        dataset_ref["inline_features"] = False
    # Pin known *project-local file inputs* by contents, not just by path.
    # Directory inputs and out-of-project paths are explicitly not guaranteed
    # by this S1.3 snapshot; dataset resolution will handle those in S2.
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in _input_paths(task, payload):
        candidate = _project_file_or_none(store, source)
        if candidate is None:
            continue
        relative = candidate.relative_to(store.project_root).as_posix()
        if relative not in seen:
            seen.add(relative)
            size, digest_file = _hash_file(candidate)
            sources.append({"path": relative, "size_bytes": size, "sha256": digest_file})
    if sources:
        dataset_ref["source_files"] = sources
    # Model configuration is a descriptive snapshot; task input.json remains
    # authoritative for all parameters, source paths, labels and arrays.
    settings = {k: v for k, v in payload.items() if k not in {"X", "y", "y_true"}}
    return {
        "dataset_ref": dataset_ref,
        "model_spec": {"task": task, "settings": settings},
        "evaluation_plan": {},
        "metadata": {"adapter": "ml-lab.recorded-legacy-task@1", "payload_sha256": digest},
    }


def _private_run_directory(store: ExperimentStore, run_id: str) -> Path:
    # A generated UUID run directory is the only write destination. Reject
    # symlinked components before creating files, including preexisting parent
    # paths. Registry registration independently enforces project containment.
    current = store.project_root
    for part in ("analysis", "ml_lab", "recorded_runs"):
        current = current / part
        if current.is_symlink():
            raise ExperimentStoreError("Refusing symlinked recorded-run directory")
    current.mkdir(parents=True, exist_ok=True)
    directory = current / run_id
    directory.mkdir(exist_ok=False)
    return directory


def _write_snapshot(path: Path, content: bytes) -> None:
    # Exclusive creation avoids silently replacing earlier run evidence.
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()


def _project_file_or_none(store: ExperimentStore, value: Any) -> Path | None:
    """Identify existing local files only; do not infer external provenance."""
    if not isinstance(value, (str, Path)) or not str(value).strip():
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = store.project_root / path
    # Do not index a path containing traversal or symlinks. Paths outside the
    # project (permitted by legacy Data Lab) remain untracked, not copied.
    if ".." in path.parts or not path.is_file():
        return None
    try:
        path.resolve(strict=True).relative_to(store.project_root)
        store.artifacts._project_file(path)
    except (ValueError, OSError, ExperimentStoreError):
        return None
    return path


def _input_paths(task: str, payload: Mapping[str, Any]) -> list[Any]:
    if not task.startswith("data."):
        return []
    if "path" in payload:
        return [payload["path"]]
    paths = payload.get("paths", [])
    if isinstance(paths, (str, Path)):
        return [paths]
    return list(paths) if isinstance(paths, (list, tuple)) else []


def _known_output_files(task: str, response: Mapping[str, Any]) -> list[tuple[Any, str]]:
    if task != "data.transform.apply":
        return []
    result = response.get("result", {})
    if not isinstance(result, dict):
        return []
    derived = result.get("derived", {})
    provenance = result.get("provenance", {})
    return [
        (derived.get("path") if isinstance(derived, dict) else None, "dataset"),
        (result.get("recipe_path"), "recipe"),
        (result.get("manifest_path"), "manifest"),
        (provenance.get("path") if isinstance(provenance, dict) else None, "provenance"),
    ]


def execute_recorded_task(
    task: str,
    payload: dict[str, Any],
    *,
    project_root: str | Path | None = None,
    experiment_id: str | None = None,
    revision: int | None = None,
    experiment_name: str | None = None,
    seed: int | None = None,
    source_artifact_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Run an existing application task while recording its inputs and result.

    This is deliberately opt-in. Existing ``execute_task(task, payload)``
    callers retain their exact envelope and side effects. A new experiment is
    created unless ``experiment_id`` identifies a previously recorded task with
    an *identical* input snapshot. Revisions must be explicitly prepared using
    the store if payload settings change; no silent mutations occur.

    The supplied ``seed`` is recorded as provenance only; it does not seed an
    underlying model (the task's own configuration determines that).
    """
    if not isinstance(task, str) or not task.strip():
        raise ExperimentStoreError("task must be a nonempty string")
    if not isinstance(payload, dict):
        raise ExperimentStoreError("payload must be a JSON object")
    if any(not isinstance(key, str) for key in payload):
        raise ExperimentStoreError("payload keys must be strings")
    if experiment_id is None and revision is not None:
        raise ExperimentStoreError("revision requires an experiment_id")
    payload_bytes = _json_bytes(payload)
    normalized = json.loads(payload_bytes)
    store = ExperimentStore(project_root)
    spec = _snapshot(store, task, normalized, payload_bytes)

    if experiment_id is None:
        experiment = store.create_experiment(experiment_name or f"Legacy task: {task}", **spec)
        experiment_id = experiment["experiment_id"]
        chosen_revision = experiment["revision"]
    else:
        experiment = store.get_experiment(experiment_id, revision=revision)
        chosen_revision = experiment["revision"]
        if experiment["metadata"].get("adapter") != spec["metadata"]["adapter"] or (
            experiment["metadata"].get("payload_sha256") != spec["metadata"]["payload_sha256"]
        ) or experiment["model_spec"].get("task") != task or experiment["dataset_ref"] != spec["dataset_ref"]:
            raise ExperimentStoreError(
                "Experiment revision does not match this task/input; create a new "
                "experiment or explicitly publish a matching revision"
            )

    run = store.create_run(experiment_id, revision=chosen_revision, seed=seed,
                           environment={"executor": "ml_lab.application.execute_task", "adapter": "legacy@1"})
    run_id = run["run_id"]
    artifacts: list[dict[str, Any]] = []
    try:
        folder = _private_run_directory(store, run_id)
        input_path = folder / "input.json"
        _write_snapshot(input_path, payload_bytes)
        input_artifact = store.artifacts.register(
            run_id, input_path, name="Original task payload", role="input", kind="payload",
            media_type="application/json", source_artifact_ids=source_artifact_ids,
            metadata={"task": task, "snapshot": True},
        )
        artifacts.append(input_artifact)

        # Legacy Data Lab paths may refer to files outside the selected project.
        # Register only explicit project-contained files, never auto-import them.
        local_inputs: list[str] = []
        for path in _input_paths(task, normalized):
            candidate = _project_file_or_none(store, path)
            if candidate is None or str(candidate) in local_inputs:
                continue
            local_inputs.append(str(candidate))
            artifacts.append(store.artifacts.register(
                run_id, candidate, role="input", kind="dataset",
                metadata={"task": task, "path_argument": True},
            ))

        store.update_run_status(run_id, "running")
        # Import only when needed: ml_lab.experiments remains lightweight.
        from ml_lab.application import execute_task

        response = execute_task(task, payload)  # Exact original caller payload and result.
        result_bytes = _json_bytes(response)
        output_path = folder / "result.json"
        _write_snapshot(output_path, result_bytes)
        sources = [a["artifact_id"] for a in artifacts if a["role"] == "input"]
        result_artifact = store.artifacts.register(
            run_id, output_path, name="Legacy task response", role="output", kind="result",
            media_type="application/json", source_artifact_ids=sources,
            metadata={"task": task, "snapshot": True},
        )
        artifacts.append(result_artifact)

        for path, kind in _known_output_files(task, response):
            candidate = _project_file_or_none(store, path)
            if candidate is None:
                continue
            artifacts.append(store.artifacts.register(
                run_id, candidate, role="output", kind=kind,
                source_artifact_ids=sources, metadata={"task": task, "legacy_derived_output": True},
            ))
        # Source file contents must still match the recorded pre-run hash.
        expected_sources = {item["path"]: item["sha256"] for item in
                            spec["dataset_ref"].get("source_files", [])}
        for artifact in artifacts:
            if artifact["role"] == "input" and artifact["kind"] == "dataset":
                if (artifact["sha256"] != expected_sources.get(artifact["path"])
                        or store.artifacts.verify(artifact["artifact_id"])["status"] != "valid"):
                    raise ExperimentStoreError("Project-local input dataset changed during recording")
        store.update_run_status(run_id, "succeeded")
    except Exception as exc:
        # The legacy task itself may have succeeded but persistence may have
        # failed. Record this as a failed *recorded invocation*, never hide it.
        store.update_run_status(run_id, "failed", error=f"{type(exc).__name__}: {exc}"[:2000])
        raise RecordedTaskError(
            f"Recorded task {task!r} failed (experiment={experiment_id}, run={run_id})",
            experiment_id=experiment_id, run_id=run_id,
        ) from exc

    return {
        "schema": "ml-lab.recorded-legacy-task@1",
        "experiment_id": experiment_id,
        "revision": chosen_revision,
        "run_id": run_id,
        "status": "succeeded",
        "response": response,
        "artifacts": artifacts,
    }
