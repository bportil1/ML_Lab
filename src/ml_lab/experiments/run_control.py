"""S4.2: project-local, cooperative run control (not process termination).

Only queued jobs can be cancelled immediately. A running task acknowledges a
request at its next safe boundary; in-process scientific kernels are NOT killed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
from uuid import uuid4

from .store import ExperimentStore, ExperimentStoreError, InvalidRunTransition


def _control_dir(store: ExperimentStore) -> Path:
    path = store.project_root
    for component in ("analysis", "ml_lab", "run_controls"):
        path = path / component
        if path.is_symlink():
            raise ExperimentStoreError("Symlinked run controls are not allowed")
        path.mkdir(exist_ok=True)
    return path


def _marker(store: ExperimentStore, run_id: str, suffix: str) -> Path:
    # Validate the run belongs to this project before constructing a path.
    store.get_run(run_id)
    if not (len(run_id) == 32 and all(c in "0123456789abcdef" for c in run_id)):
        raise ExperimentStoreError("Invalid run identifier")
    return _control_dir(store) / f"{run_id}.{suffix}"


def cancellation_requested(store: ExperimentStore, run_id: str) -> bool:
    path = _marker(store, run_id, "cancel")
    if path.is_symlink():
        raise ExperimentStoreError("Symlinked cancellation marker is not allowed")
    return path.is_file()


class RunCancelled(Exception):
    """Cooperative cancellation reached a safe execution boundary."""


def check_cancelled(store: ExperimentStore, run_id: str) -> None:
    if cancellation_requested(store, run_id):
        raise RunCancelled("Cancellation requested; stopped at a safe stage boundary")


def request_run_cancellation(store: ExperimentStore, run_id: str) -> dict:
    """Cancel queued work atomically, or record a cooperative running request."""
    state = store.get_run(run_id)
    if state["status"] == "queued":
        try:
            return {"status": store.update_run_status(run_id, "cancelled")["status"], "acknowledged": True}
        except InvalidRunTransition:
            # A worker claimed the job concurrently. Inspect its new state.
            state = store.get_run(run_id)
    if state["status"] != "running":
        return {"status": state["status"], "acknowledged": state["status"] == "cancelled"}
    marker = _marker(store, run_id, "cancel")
    if marker.is_symlink():
        raise ExperimentStoreError("Symlinked cancellation marker is not allowed")
    try:
        with marker.open("x", encoding="utf-8") as handle:
            handle.write("cancel requested\n")
    except FileExistsError:
        if not marker.is_file() or marker.is_symlink():
            raise ExperimentStoreError("Invalid cancellation marker")
    # Report request, never imply the running engine stopped immediately.
    return {"status": store.get_run(run_id)["status"], "acknowledged": False,
            "cancellation_requested": True}


def _linux_process_identity(pid: int) -> str | None:
    """Linux /proc PID start ticks: avoids mistaking a recycled PID for owner."""
    try:
        line = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        return line.rsplit(") ", 1)[1].split()[19]  # field 22, offset 3
    except (OSError, IndexError, ValueError):
        return None


def write_owner(store: ExperimentStore, run_id: str) -> None:
    marker = _marker(store, run_id, "owner")
    identity = _linux_process_identity(os.getpid())
    if identity is None:
        return  # Never fabricate a verifiable lease on unsupported platforms.
    payload = {"pid": os.getpid(), "start_ticks": identity,
               "host": socket.gethostname(), "token": uuid4().hex}
    with marker.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle)


def recover_interrupted_runs(store: ExperimentStore, experiment_id: str) -> list[str]:
    """Explicit conservative recovery when a verified Linux worker no longer exists.

    Unknown owners and different hosts are intentionally left untouched. This
    marks dead runs failed, not resumed; use a fresh run for any retry.
    """
    recovered: list[str] = []
    for run in store.list_runs(experiment_id, limit=500):
        if run["status"] != "running":
            continue
        marker = _marker(store, run["run_id"], "owner")
        if not marker.is_file() or marker.is_symlink():
            continue
        try:
            owner = json.loads(marker.read_text(encoding="utf-8"))
            if owner["host"] != socket.gethostname() or type(owner["pid"]) is not int:
                continue
            if not isinstance(owner["start_ticks"], str):
                continue
            # Linux only, and only if /proc is available; no PID-only guesswork.
            if not Path("/proc/self/stat").exists():
                continue
            current = _linux_process_identity(owner["pid"])
            if current == owner["start_ticks"] or Path(f"/proc/{owner['pid']}").exists():
                continue
            try:
                store.update_run_status(run["run_id"], "failed",
                                        error="Worker process ended before completion; rerun as a new job")
                recovered.append(run["run_id"])
            except InvalidRunTransition:
                continue
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return recovered
