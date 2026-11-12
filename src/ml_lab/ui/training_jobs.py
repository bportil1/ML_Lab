"""S4.1: lightweight, opt-in model-run launcher for the ML_Lab UI.

Jobs are in-process daemon threads, like existing Data Lab profile jobs.
Persistent SQLite run states and on-disk stage snapshots remain accessible
while a UI worker is alive.  These are NOT durable external workers: a server
restart can interrupt a running job; S4.2 handles interruption recovery.
"""
from __future__ import annotations

import threading
from pathlib import Path

from ml_lab.experiments import (
    ConfiguredRunError, ExperimentStore, ExperimentStoreError,
    execute_configured_run, queue_configured_run,
)


class TrainingJobManager:
    def __init__(self, project_root: str | Path, *, max_active: int = 2) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        self.max_active = max_active
        self._active: set[str] = set()
        self._lock = threading.RLock()

    def start(self, experiment_id: str, *, expected_revision: int, seed: int | None = None) -> str:
        with self._lock:
            if len(self._active) >= self.max_active:
                raise ExperimentStoreError("Too many active model runs; try again after one finishes")
            store = ExperimentStore(self.project_root)
            queued = queue_configured_run(
                store, experiment_id, revision=expected_revision,
                expected_revision=expected_revision, seed=seed,
            )
            run_id = queued["run_id"]
            self._active.add(run_id)
            try:
                threading.Thread(
                    target=self._execute, args=(run_id,),
                    name=f"ml-lab-model-{run_id[:10]}", daemon=True,
                ).start()
            except RuntimeError as exc:
                self._active.discard(run_id)
                store.update_run_status(run_id, "failed", error=f"Could not start worker: {exc}"[:2000])
                raise ExperimentStoreError("Could not start training worker") from exc
        return run_id

    def start_optimization(self, experiment_id: str, *, expected_revision: int, plan: dict) -> str:
        """Launch a bounded, explicitly requested serial optimizer search."""
        from ml_lab.experiments import queue_optimization_run
        with self._lock:
            if len(self._active) >= self.max_active:
                raise ExperimentStoreError("Too many active model runs")
            store = ExperimentStore(self.project_root)
            queued = queue_optimization_run(store, experiment_id, plan, expected_revision=expected_revision)
            run_id = queued["run_id"]
            self._active.add(run_id)
            try:
                threading.Thread(target=self._execute_opt, args=(run_id,),
                                 name=f"ml-lab-optimization-{run_id[:10]}", daemon=True).start()
            except RuntimeError as exc:
                self._active.discard(run_id)
                store.update_run_status(run_id, "failed", error=f"Could not start optimizer worker: {exc}"[:2000])
                raise ExperimentStoreError("Could not start optimization worker") from exc
        return run_id

    def start_synthetic(self, experiment_id: str, *, expected_revision: int, plan: dict) -> str:
        """Launch a bounded synthetic-data run with the existing local job budget."""
        from ml_lab.experiments import queue_synthetic_run
        with self._lock:
            if len(self._active) >= self.max_active:
                raise ExperimentStoreError("Too many active model runs")
            store = ExperimentStore(self.project_root)
            queued = queue_synthetic_run(store, experiment_id, plan, expected_revision=expected_revision)
            run_id = queued["run_id"]
            self._active.add(run_id)
            try:
                threading.Thread(target=self._execute_synthetic, args=(run_id,),
                                 name=f"ml-lab-synthetic-{run_id[:10]}", daemon=True).start()
            except RuntimeError as exc:
                self._active.discard(run_id)
                store.update_run_status(run_id, "failed", error=f"Could not start worker: {exc}"[:2000])
                raise ExperimentStoreError("Could not start synthetic worker") from exc
        return run_id

    def _execute_synthetic(self, run_id: str) -> None:
        from ml_lab.experiments import execute_synthetic_run
        try:
            execute_synthetic_run(ExperimentStore(self.project_root), run_id)
        except ConfiguredRunError:
            pass
        finally:
            with self._lock:
                self._active.discard(run_id)

    def _execute_opt(self, run_id: str) -> None:
        from ml_lab.experiments import execute_optimization_run
        try:
            execute_optimization_run(ExperimentStore(self.project_root), run_id)
        except ConfiguredRunError:
            pass
        finally:
            with self._lock:
                self._active.discard(run_id)

    def _execute(self, run_id: str) -> None:
        try:
            execute_configured_run(ExperimentStore(self.project_root), run_id)
        except ConfiguredRunError:
            pass  # Failure has already been recorded in the project-local store.
        finally:
            with self._lock:
                self._active.discard(run_id)
