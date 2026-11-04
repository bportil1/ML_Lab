"""S4.2 opt-in optimizer and cooperative control regression tests."""
from __future__ import annotations

import json
from pathlib import Path
import socket

import pytest

from ml_lab.experiments import (
    ConfiguredRunError, ExperimentStore, ExperimentStoreError,
    configure_experiment_model, optimize_configured_experiment,
    queue_configured_run, queue_optimization_run, request_run_cancellation,
    execute_configured_run, execute_optimization_run, recover_interrupted_runs,
    select_experiment_dataset,
)
from ml_lab.experiments.run_control import _marker, check_cancelled, RunCancelled


def prepared(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    source = root / "matrix.csv"
    source.write_text("a,b,c\n1,2,3\n2,5,6\n3,6,2\n4,9,3\n5,10,6\n6,11,7\n", encoding="utf-8")
    store = ExperimentStore(root)
    e = select_experiment_dataset(store, source, feature_columns=["a", "b", "c"])
    e = configure_experiment_model(store, e["experiment_id"], family="representation", method="pca",
                                   settings={"config": {"n_components": 1}}, expected_revision=e["revision"])
    return store, e, source


PLAN = {"parameter": "config.n_components", "lower": 1, "upper": 2,
        "metric": "rmse", "goal": "minimize", "population_size": 2,
        "iterations": 0, "seed": 9}


def test_optimize_real_pca_records_trials_and_provenance(tmp_path):
    store, e, source = prepared(tmp_path)
    result = optimize_configured_experiment(store, e["experiment_id"], PLAN,
                                            expected_revision=e["revision"])
    assert result["status"] == "succeeded"
    assert result["evaluations"] == 2
    assert store.get_run(result["run_id"])["status"] == "succeeded"
    artifacts = store.artifacts.list_for_run(result["run_id"])
    assert [a["kind"] for a in artifacts].count("trial") == 2
    assert {"result", "history", "dataset", "manifest", "search_space", "trial"} <= {a["kind"] for a in artifacts}
    assert all(store.artifacts.verify(a["artifact_id"])["status"] == "valid" for a in artifacts)
    assert source.read_text().startswith("a,b,c")
    assert store.get_experiment(e["experiment_id"])["revision"] == e["revision"]
    trial = next(a for a in artifacts if a["kind"] == "trial")
    trial_data = json.loads((store.project_root / trial["path"]).read_text())
    assert trial_data["metric"] == "rmse" and trial_data["status"] == "ok"


def test_plan_validation_rejects_unsafe_controls_and_large_searches(tmp_path):
    store, e, _ = prepared(tmp_path)
    for updates in ({"parameter": "config.random_state"}, {"parameter": "config.missing"},
                    {"parameter": "dataset.X"}, {"population_size": 99},
                    {"iterations": 10}, {"metric": "unknown"},
                    {"lower": 1.5}, {"lower": 4, "upper": 1},
                    {"lower": float('nan')}, {"parameter": "config.n_components", "lower": 0.8, "upper": 2}):
        with pytest.raises(ExperimentStoreError):
            queue_optimization_run(store, e["experiment_id"], {**PLAN, **updates},
                                   expected_revision=e["revision"])
    assert not store.list_runs(e["experiment_id"])


def test_queue_revision_and_source_validation(tmp_path):
    store, e, source = prepared(tmp_path)
    with pytest.raises(ExperimentStoreError):
        queue_optimization_run(store, e["experiment_id"], PLAN, expected_revision=1)
    source.write_text(source.read_text() + "8,9,10\n")
    with pytest.raises(ExperimentStoreError):
        queue_optimization_run(store, e["experiment_id"], PLAN, expected_revision=e["revision"])


def test_queued_cancellation_is_terminal_and_prevents_claim(tmp_path):
    store, e, _ = prepared(tmp_path)
    queued = queue_optimization_run(store, e["experiment_id"], PLAN, expected_revision=e["revision"])
    assert request_run_cancellation(store, queued["run_id"])["acknowledged"]
    assert store.get_run(queued["run_id"])["status"] == "cancelled"
    with pytest.raises(ExperimentStoreError):
        execute_optimization_run(store, queued["run_id"])
    assert request_run_cancellation(store, queued["run_id"])["status"] == "cancelled"


def test_running_requests_do_not_claim_instant_cancellation(tmp_path):
    store, e, _ = prepared(tmp_path)
    queued = queue_configured_run(store, e["experiment_id"])
    store.update_run_status(queued["run_id"], "running")
    response = request_run_cancellation(store, queued["run_id"])
    assert response["cancellation_requested"] and not response["acknowledged"]
    assert store.get_run(queued["run_id"])["status"] == "running"
    with pytest.raises(RunCancelled):
        check_cancelled(store, queued["run_id"])


def test_optimization_cancel_request_is_checked_before_model_launch(tmp_path):
    store, e, _ = prepared(tmp_path)
    queued = queue_optimization_run(store, e["experiment_id"], PLAN, expected_revision=e["revision"])
    # Worker claims the job, then sees a request before the costly engine.
    store.update_run_status(queued["run_id"], "running")
    request_run_cancellation(store, queued["run_id"])
    with pytest.raises(RunCancelled):
        check_cancelled(store, queued["run_id"])


def test_explicit_recovery_only_when_owner_verified_dead(tmp_path):
    if not Path("/proc/self/stat").exists():
        pytest.skip("Conservative process identity recovery requires Linux procfs")
    store, e, _ = prepared(tmp_path)
    queued = queue_configured_run(store, e["experiment_id"])
    store.update_run_status(queued["run_id"], "running")
    owner = _marker(store, queued["run_id"], "owner")
    owner.write_text(json.dumps({"pid": 99999999, "start_ticks": "12345", "host": socket.gethostname()}))
    assert recover_interrupted_runs(store, e["experiment_id"]) == [queued["run_id"]]
    assert store.get_run(queued["run_id"])["status"] == "failed"
    assert recover_interrupted_runs(store, e["experiment_id"]) == []


def test_optimizer_failed_candidates_mark_run_failed(tmp_path):
    store, e, _ = prepared(tmp_path)
    # Fractional threshold is validated but sklearn may reject otherwise
    # valid candidates if data is pathological; simulate an invalid result.
    from unittest.mock import patch
    with patch("ml_lab.application.execute_task", side_effect=ValueError("synthetic model failure")):
        queued = queue_optimization_run(store, e["experiment_id"], PLAN, expected_revision=e["revision"])
        with pytest.raises(ConfiguredRunError):
            execute_optimization_run(store, queued["run_id"])
    assert store.get_run(queued["run_id"])["status"] == "failed"
    assert len([a for a in store.artifacts.list_for_run(queued["run_id"]) if a["kind"] == "trial"]) == 2


def test_cancel_during_candidate_is_acknowledged_after_engine_returns(tmp_path):
    store, e, _ = prepared(tmp_path)
    queued = queue_optimization_run(store, e["experiment_id"], PLAN, expected_revision=e["revision"])
    from ml_lab.application import execute_task as real_execute
    from unittest.mock import patch
    def model_with_request(task, payload):
        # A real engine cannot be preempted. Request cancellation during the
        # model invocation; once it returns the runner must not mark success.
        response = real_execute(task, payload)
        pending = request_run_cancellation(store, queued["run_id"])
        assert pending["cancellation_requested"]
        return response
    with patch("ml_lab.application.execute_task", side_effect=model_with_request):
        with pytest.raises(ConfiguredRunError):
            execute_optimization_run(store, queued["run_id"])
    assert store.get_run(queued["run_id"])["status"] == "cancelled"
    trials = [a for a in store.artifacts.list_for_run(queued["run_id"]) if a["kind"] == "trial"]
    assert len(trials) == 1
    trial_record = json.loads((store.project_root / trials[0]["path"]).read_text())
    assert trial_record["status"] == "cancelled"


def test_project_isolation_cannot_cancel_foreign_run(tmp_path):
    store, e, _ = prepared(tmp_path)
    queued = queue_optimization_run(store, e["experiment_id"], PLAN, expected_revision=e["revision"])
    foreign = tmp_path / "other_project"
    foreign.mkdir()
    other = ExperimentStore(foreign)
    with pytest.raises(ExperimentStoreError):
        request_run_cancellation(other, queued["run_id"])
    assert store.get_run(queued["run_id"])["status"] == "queued"
