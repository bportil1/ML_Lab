from __future__ import annotations

import threading
import time
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ml_lab import data

try:
    from pah_jobs.exchange import begin as _monitor_begin, note as _monitor_note
    from pah_jobs.exchange import heartbeat as _monitor_heartbeat, finish as _monitor_finish
    from pah_jobs.exchange import monitor_root as _monitor_root, monitor_directory as _monitor_directory
except ImportError:
    from contextlib import nullcontext
    def _monitor_root():
        return None
    def _monitor_directory(path):
        return nullcontext()
    def _monitor_begin(*args, **kwargs):
        return None
    def _monitor_note(*args, **kwargs):
        return None
    def _monitor_heartbeat(*args, **kwargs):
        return None
    def _monitor_finish(*args, **kwargs):
        return None


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProfileJobManager:
    """Small in-process profile runner for the optional UI.

    This deliberately avoids external workers. The core profiler remains
    synchronous; only the presentation layer uses a daemon thread so the UI can
    show visible starting/running/completed/failed state while work is active.
    """

    def __init__(self, *, output_root: str | Path = "analysis/data/profile_runs") -> None:
        self.output_root = Path(output_root).expanduser()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._monitor_ids: dict[str, str | None] = {}
        self._monitor_roots: dict[str, str | None] = {}

    def start(self, payload: dict[str, Any]) -> str:
        job_id = uuid.uuid4().hex[:16]
        now = _utcnow()
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "starting",
                "stage": "Preparing profile run",
                "current": 0,
                "total": 0,
                "current_path": None,
                "created_at": now,
                "started_at": None,
                "finished_at": None,
                "elapsed_seconds": 0.0,
                "error": None,
                "warnings": [],
                "payload": deepcopy(payload),
                "artifact_manifest": None,
            }
        self._monitor_roots[job_id] = _monitor_root()
        self._monitor_ids[job_id] = _monitor_begin("ml_lab", "Data Lab profile", run_id=job_id)
        thread = threading.Thread(target=self._run_monitored, args=(job_id,), name=f"ml-lab-profile-{job_id}", daemon=True)
        thread.start()
        return job_id

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            snapshot = deepcopy(job)
        started = snapshot.get("started_monotonic")
        if started is not None and snapshot["status"] in {"starting", "running"}:
            snapshot["elapsed_seconds"] = max(0.0, time.monotonic() - started)
        snapshot.pop("started_monotonic", None)
        return snapshot

    def _update(self, job_id: str, **values: Any) -> None:
        with self._lock:
            self._jobs[job_id].update(values)

    def _progress(self, job_id: str, event: dict[str, Any]) -> None:
        stage = event.get("stage")
        labels = {
            "inventory_complete": "Inventory complete",
            "profiling_file": "Profiling dataset",
            "profiled_file": "Dataset profile complete",
            "profile_failed": "Dataset profile failed",
        }
        current = int(event.get("current", 0))
        total = int(event.get("total", 0))
        monitor_id = self._monitor_ids.get(job_id)
        _monitor_heartbeat("ml_lab", monitor_id, progress=(current / total if total else None))
        _monitor_note("ml_lab", monitor_id, labels.get(str(stage), str(stage or "Profiling data")))
        self._update(
            job_id,
            status="running",
            stage=labels.get(str(stage), str(stage or "Profiling data")),
            current=int(event.get("current", 0)),
            total=int(event.get("total", 0)),
            current_path=event.get("path"),
        )

    def _run_monitored(self, job_id: str) -> None:
        root = self._monitor_roots.get(job_id)
        if root is None:
            return self._run(job_id)
        with _monitor_directory(root):
            return self._run(job_id)

    def _run(self, job_id: str) -> None:
        started = time.monotonic()
        job = self.get(job_id)
        if job is None:
            return
        payload = job["payload"]
        self._update(
            job_id,
            status="running",
            stage="Inspecting source files",
            started_at=_utcnow(),
            started_monotonic=started,
        )
        try:
            collection = data.profile_paths(
                payload["paths"],
                recursive=bool(payload.get("recursive", True)),
                include_hidden=bool(payload.get("include_hidden", False)),
                preview_rows=int(payload.get("preview_rows", 20)),
                max_rows=payload.get("max_rows", 100_000),
                relationship_rows=int(payload.get("relationship_rows", 5_000)),
                max_relationship_columns=int(payload.get("max_relationship_columns", 25)),
                max_relationship_pairs=int(payload.get("max_relationship_pairs", 200)),
                outlier_iqr_multiplier=float(payload.get("outlier_iqr_multiplier", 1.5)),
                random_state=int(payload.get("random_state", 42)),
                progress=lambda event: self._progress(job_id, event),
            )
            self._update(job_id, stage="Saving profile artifacts", current=len(collection.profiles), total=len(collection.profiles))
            manifest = data.persist_profile_run(collection, output_root=self.output_root, run_id=job_id)
            elapsed = time.monotonic() - started
            self._update(
                job_id,
                status="completed",
                stage="Profile complete",
                current=len(collection.profiles),
                total=len(collection.profiles),
                current_path=None,
                finished_at=_utcnow(),
                elapsed_seconds=elapsed,
                warnings=list(collection.warnings),
                artifact_manifest=manifest,
            )
            _monitor_finish("ml_lab", self._monitor_ids.get(job_id))
        except Exception as exc:  # UI job boundary: preserve diagnostics rather than killing the worker
            elapsed = time.monotonic() - started
            self._update(
                job_id,
                status="failed",
                stage="Profile failed",
                finished_at=_utcnow(),
                elapsed_seconds=elapsed,
                error=f"{type(exc).__name__}: {exc}",
            )
            _monitor_note("ml_lab", self._monitor_ids.get(job_id), f"{type(exc).__name__}: {exc}", stream="stderr")
            _monitor_finish("ml_lab", self._monitor_ids.get(job_id), state="failed", reason=type(exc).__name__)
