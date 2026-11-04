"""Sprint 4.1: pinned model runs, provenance, state transitions and failure cases."""
from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import pytest

from ml_lab.experiments import (
    ConfiguredRunError, ExperimentStore, ExperimentStoreError,
    configure_experiment_model, execute_configured_run,
    get_configured_run_status, queue_configured_run,
    run_configured_experiment, select_experiment_dataset,
)


def prepared(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    source = project / "source.csv"
    with source.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["a", "b"])
        writer.writerows([[i, i * 2] for i in range(12)])
    store = ExperimentStore(project)
    selected = select_experiment_dataset(store, source, feature_columns=["a", "b"])
    configured = configure_experiment_model(
        store, selected["experiment_id"], family="representation", method="pca",
        settings={"config": {"n_components": 1}}, expected_revision=selected["revision"],
    )
    return store, configured, source


def test_pca_run_is_pinned_and_registers_source_config_result(tmp_path):
    store, experiment, source = prepared(tmp_path)
    events = []
    result = run_configured_experiment(store, experiment["experiment_id"], seed=37, progress=events.append)
    assert result["status"] == "succeeded"
    run = store.get_run(result["run_id"])
    assert run["status"] == "succeeded"
    assert run["revision"] == experiment["revision"]
    assert run["seed"] == 37
    assert "recorded provenance" in run["environment"]["seed_note"]
    assert run["started_at"] and run["finished_at"]
    kinds = {item["kind"] for item in result["artifacts"]}
    assert kinds == {"dataset", "manifest", "result"}
    for item in result["artifacts"]:
        assert store.artifacts.verify(item["artifact_id"])["status"] == "valid"
        assert item["revision"] == experiment["revision"]
    manifest = next(item for item in result["artifacts"] if item["kind"] == "manifest")
    assert source.read_text()  # source was never modified by the run
    assert "source.csv" in (store.project_root / manifest["path"]).read_text()
    assert [(e["step"], e["status"]) for e in events] == [
        (1, "running"), (2, "running"), (3, "running"), (4, "succeeded")
    ]
    status = get_configured_run_status(store, result["run_id"])
    assert status["stage"] == "Model run completed"
    assert status["step"] == 4
    assert len(status["artifacts"]) == 3


def test_queue_does_not_execute_and_revision_cas_prevents_stale_submit(tmp_path):
    store, experiment, _ = prepared(tmp_path)
    queued = queue_configured_run(
        store, experiment["experiment_id"], expected_revision=experiment["revision"]
    )
    assert queued["status"] == "queued"
    assert get_configured_run_status(store, queued["run_id"])["stage"] == "Queued"
    assert not (store.project_root / "analysis" / "ml_lab" / "training_runs" / queued["run_id"]).exists()
    store.revise_experiment(
        experiment["experiment_id"], dataset_ref=experiment["dataset_ref"],
        model_spec=experiment["model_spec"], evaluation_plan=experiment["evaluation_plan"],
        metadata=experiment["metadata"], expected_revision=experiment["revision"],
    )
    with pytest.raises(ExperimentStoreError, match="revision changed"):
        queue_configured_run(store, experiment["experiment_id"],
                             revision=experiment["revision"], expected_revision=experiment["revision"])
    assert len(store.list_runs(experiment["experiment_id"])) == 1
    # A previously queued run still references its historical immutable revision.
    assert execute_configured_run(store, queued["run_id"])["status"] == "succeeded"


def test_reject_unconfigured_experiment_and_stale_source(tmp_path):
    store, experiment, source = prepared(tmp_path)
    with pytest.raises(ExperimentStoreError, match="configuration"):
        queue_configured_run(store, experiment["experiment_id"], revision=1)
    queued = queue_configured_run(store, experiment["experiment_id"])
    with source.open("a", encoding="utf-8") as output:
        output.write("44,45\n")
    with pytest.raises(ConfiguredRunError) as error:
        execute_configured_run(store, queued["run_id"])
    assert error.value.run_id == queued["run_id"]
    assert store.get_run(queued["run_id"])["status"] == "failed"
    with pytest.raises(ExperimentStoreError, match="changed"):
        queue_configured_run(store, experiment["experiment_id"])


def test_failure_is_recorded_and_run_cannot_be_replayed(tmp_path, monkeypatch):
    store, experiment, _ = prepared(tmp_path)
    def broken(*args, **kwargs):
        raise RuntimeError("deliberate engine failure")
    monkeypatch.setattr("ml_lab.application.execute_task", broken)
    queued = queue_configured_run(store, experiment["experiment_id"])
    with pytest.raises(ConfiguredRunError) as error:
        execute_configured_run(store, queued["run_id"])
    assert error.value.run_id == queued["run_id"]
    state = store.get_run(queued["run_id"])
    assert state["status"] == "failed"
    assert "deliberate engine failure" in state["error"]
    assert get_configured_run_status(store, queued["run_id"])["stage"] == "Model run failed"
    with pytest.raises(ExperimentStoreError, match="Only queued"):
        execute_configured_run(store, queued["run_id"])


def test_changed_source_during_execution_marks_failed(tmp_path, monkeypatch):
    store, experiment, source = prepared(tmp_path)
    def mutate(task, payload):
        with source.open("a", encoding="utf-8") as output:
            output.write("16,32\n")
        return {"task": task, "result": {"ok": True}}
    monkeypatch.setattr("ml_lab.application.execute_task", mutate)
    queued = queue_configured_run(store, experiment["experiment_id"])
    with pytest.raises(ConfiguredRunError):
        execute_configured_run(store, queued["run_id"])
    assert store.get_run(queued["run_id"])["status"] == "failed"
    assert not any(a["kind"] == "result" for a in store.artifacts.list_for_run(queued["run_id"]))


def test_store_rejects_cross_project_run_ids(tmp_path):
    store, experiment, _ = prepared(tmp_path)
    other = tmp_path / "different-project"
    other.mkdir()
    different_store = ExperimentStore(other)
    queued = queue_configured_run(store, experiment["experiment_id"])
    with pytest.raises(ExperimentStoreError, match="Unknown run"):
        get_configured_run_status(different_store, queued["run_id"])
    with pytest.raises(ExperimentStoreError, match="Unknown run"):
        execute_configured_run(different_store, queued["run_id"])


def test_symlinked_training_directory_fails_closed(tmp_path):
    store, experiment, _ = prepared(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    path = store.project_root / "analysis" / "ml_lab" / "training_runs"
    path.symlink_to(outside, target_is_directory=True)
    queued = queue_configured_run(store, experiment["experiment_id"])
    with pytest.raises(ConfiguredRunError):
        execute_configured_run(store, queued["run_id"])
    assert not list(outside.iterdir())
    assert store.get_run(queued["run_id"])["status"] == "failed"


def test_duplicate_worker_does_not_train_twice(tmp_path, monkeypatch):
    store, experiment, _ = prepared(tmp_path)
    queued = queue_configured_run(store, experiment["experiment_id"])
    calls = []
    def fake_execute(task, payload):
        calls.append(task)
        time.sleep(.08)
        return {"task": task, "result": {"ok": True}}
    monkeypatch.setattr("ml_lab.application.execute_task", fake_execute)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(execute_configured_run, store, queued["run_id"]) for _ in range(2)]
    successes = [f.result() for f in futures if f.exception() is None]
    failures = [f.exception() for f in futures if f.exception() is not None]
    assert len(successes) == 1
    assert len(failures) == 1
    assert "Only queued" in str(failures[0]) or "Cannot change run" in str(failures[0])
    assert calls == ["representation"]


def test_training_manager_starts_daemon_ui_worker(tmp_path):
    from ml_lab.ui.training_jobs import TrainingJobManager
    store, experiment, _ = prepared(tmp_path)
    manager = TrainingJobManager(store.project_root, max_active=1)
    run_id = manager.start(experiment["experiment_id"], expected_revision=experiment["revision"])
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        state = store.get_run(run_id)
        if state["status"] in {"succeeded", "failed"}:
            break
        time.sleep(.03)
    assert state["status"] == "succeeded", state
    assert run_id not in manager._active or store.get_run(run_id)["status"] == "succeeded"
