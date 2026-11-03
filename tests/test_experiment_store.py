"""S1.1 local metadata storage; no model execution or PAH dependency."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import sqlite3
from pathlib import Path

import pytest

from ml_lab.experiments import (
    ExperimentNotFound,
    ExperimentStore,
    ExperimentStoreError,
    InvalidRunTransition,
    SchemaVersionError,
)


def test_experiment_revisions_and_run_snapshots_survive_reopen(tmp_path: Path):
    store = ExperimentStore(tmp_path)
    experiment = store.create_experiment(
        "  classifier trial  ",
        dataset_ref={"dataset_id": "d1", "sha256": "abc"},
        model_spec={"family": "classification", "config": {"cv": 3}},
        evaluation_plan={"holdout": 0.2},
    )
    assert experiment["schema"] == "ml-lab.experiment@1"
    assert experiment["name"] == "classifier trial"
    assert experiment["revision"] == 1
    assert experiment["dataset_ref"]["dataset_id"] == "d1"

    first = store.create_run(experiment["experiment_id"], seed=42, environment={"python": "3.12"})
    assert first["status"] == "queued"
    assert first["revision"] == 1
    assert first["seed"] == 42
    running = store.update_run_status(first["run_id"], "running")
    assert running["started_at"] and not running["finished_at"]
    completed = store.update_run_status(first["run_id"], "succeeded")
    assert completed["started_at"] == running["started_at"]
    assert completed["finished_at"]

    next_revision = store.revise_experiment(
        experiment["experiment_id"],
        dataset_ref={"dataset_id": "d2"},
        model_spec={"family": "clustering"},
    )
    assert next_revision["revision"] == 2
    assert store.create_run(experiment["experiment_id"])["revision"] == 2
    reopened = ExperimentStore(tmp_path)
    assert reopened.get_experiment(experiment["experiment_id"])["dataset_ref"] == {"dataset_id": "d2"}
    assert reopened.get_experiment(experiment["experiment_id"], revision=1) == experiment
    assert reopened.get_run(first["run_id"]) == completed
    assert len(reopened.list_experiments()) == 1
    assert len(reopened.list_runs(experiment["experiment_id"])) == 2


def test_isolated_projects_and_working_directory_default(tmp_path: Path, monkeypatch):
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    first = ExperimentStore(alpha)
    record = first.create_experiment("alpha only")
    second = ExperimentStore(beta)
    assert first.db_path != second.db_path
    assert second.list_experiments() == []
    with pytest.raises(ExperimentNotFound):
        second.get_experiment(record["experiment_id"])
    monkeypatch.chdir(beta)
    assert ExperimentStore().project_root == beta.resolve()
    assert ExperimentStore().list_experiments() == []


def test_invalid_payloads_and_run_transitions_are_rejected(tmp_path: Path):
    store = ExperimentStore(tmp_path)
    with pytest.raises(ExperimentStoreError):
        store.create_experiment("")
    with pytest.raises(ExperimentStoreError):
        store.create_experiment("bad", metadata={"not-json": {1, 2}})
    with pytest.raises(ExperimentStoreError):
        store.create_experiment("bad", metadata={"nan": float("nan")})
    assert store.list_experiments() == []
    experiment = store.create_experiment("valid")
    identifier = experiment["experiment_id"]
    with pytest.raises(ExperimentNotFound):
        store.create_run(identifier, revision=77)
    with pytest.raises(ExperimentStoreError):
        store.create_run(identifier, seed=True)
    queued = store.create_run(identifier)
    with pytest.raises(InvalidRunTransition):
        store.update_run_status(queued["run_id"], "queued")
    with pytest.raises(ExperimentStoreError):
        store.update_run_status(queued["run_id"], "running", error="invalid")
    assert store.get_run(queued["run_id"])["status"] == "queued"
    failed = store.update_run_status(queued["run_id"], "failed", error="worker failed")
    assert failed["error"] == "worker failed"
    assert failed["finished_at"]
    with pytest.raises(InvalidRunTransition):
        store.update_run_status(queued["run_id"], "running")
    with pytest.raises(ExperimentNotFound):
        store.get_run("missing")
    with pytest.raises(ExperimentNotFound):
        store.revise_experiment("missing")
    with pytest.raises(ExperimentStoreError):
        store.list_experiments(limit=0)
    with pytest.raises(ExperimentStoreError):
        store.list_runs(identifier, offset=-1)


def test_failed_transaction_does_not_leave_partial_metadata(tmp_path: Path):
    store = ExperimentStore(tmp_path)
    experiment = store.create_experiment("safe")
    with pytest.raises(ExperimentStoreError):
        store.revise_experiment(experiment["experiment_id"], model_spec={"invalid": object()})
    assert store.get_experiment(experiment["experiment_id"])["revision"] == 1


def test_newer_and_unversioned_databases_are_not_overwritten(tmp_path: Path):
    project = tmp_path / "future"
    project.mkdir()
    db_path = project / "analysis" / "ml_lab" / "experiments.sqlite3"
    db_path.parent.mkdir(parents=True)
    with sqlite3.connect(db_path) as db:
        db.execute("PRAGMA user_version = 77")
    with pytest.raises(SchemaVersionError):
        ExperimentStore(project)
    with sqlite3.connect(db_path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 77

    other = tmp_path / "unversioned"
    other.mkdir()
    other_db_path = other / "analysis" / "ml_lab" / "experiments.sqlite3"
    other_db_path.parent.mkdir(parents=True)
    with sqlite3.connect(other_db_path) as db:
        db.execute("CREATE TABLE foreign_table (id INTEGER PRIMARY KEY)")
    with pytest.raises(SchemaVersionError):
        ExperimentStore(other)
    with sqlite3.connect(other_db_path) as db:
        assert db.execute("SELECT name FROM sqlite_master WHERE name = 'foreign_table'").fetchone()


def test_concurrent_revision_and_run_writes(tmp_path: Path):
    store = ExperimentStore(tmp_path)
    experiment = store.create_experiment("concurrent")
    identifier = experiment["experiment_id"]

    def work(index: int) -> tuple[int, str]:
        local_store = ExperimentStore(tmp_path)
        revision = local_store.revise_experiment(identifier, model_spec={"iteration": index})
        run = local_store.create_run(identifier, revision=revision["revision"])
        return revision["revision"], run["run_id"]

    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(workers.map(work, range(10)))
    assert {rev for rev, _ in results} == set(range(2, 12))
    assert len({run_id for _, run_id in results}) == 10
    assert store.get_experiment(identifier)["revision"] == 11
    assert len(store.list_runs(identifier)) == 10
