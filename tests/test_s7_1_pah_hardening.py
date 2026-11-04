"""Sprint 7.1: local PAH project binding and integrity-checked artifact handoff."""
from __future__ import annotations

import json
import sys
import threading
import types
from pathlib import Path

from ml_lab.experiments import (ExperimentStore, configure_experiment_model,
                                run_configured_experiment, select_experiment_dataset)
from ml_lab.pah_integration import MLLabRuntimeAdapter, discover_artifacts


def _prepared_training(project: Path):
    project.mkdir(parents=True)
    source = project / "source.csv"
    source.write_text("a,b\n1,2\n2,4\n3,6\n4,8\n5,10\n6,12\n", encoding="utf-8")
    store = ExperimentStore(project)
    selection = select_experiment_dataset(store, source, feature_columns=["a", "b"])
    experiment = configure_experiment_model(
        store, selection["experiment_id"], family="representation", method="pca",
        settings={"config": {"n_components": 1}}, expected_revision=selection["revision"],
    )
    return store, experiment


def test_host_project_binding_and_switching_without_flask(monkeypatch, tmp_path):
    """Check the actual launch kwargs with a local-only fake WSGI server."""
    from ml_lab import ui

    configuration = []
    servers = []
    class FakeServer:
        def __init__(self):
            self.stopped = threading.Event()
            self.closed = False
        def serve_forever(self):
            self.stopped.wait(2)
        def shutdown(self):
            self.stopped.set()
        def server_close(self):
            self.closed = True

    def fake_make_server(host, port, app, **kwargs):
        server = FakeServer()
        servers.append(server)
        assert kwargs["threaded"] is True
        return server

    def fake_create_app(*, config):
        configuration.append(config)
        return types.SimpleNamespace(wsgi_app=lambda env, start: None)

    monkeypatch.setattr(ui, "create_app", fake_create_app)
    fake_werkzeug = types.ModuleType("werkzeug")
    fake_serving = types.ModuleType("werkzeug.serving")
    fake_serving.make_server = fake_make_server
    fake_werkzeug.serving = fake_serving
    monkeypatch.setitem(sys.modules, "werkzeug", fake_werkzeug)
    monkeypatch.setitem(sys.modules, "werkzeug.serving", fake_serving)

    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    adapter = MLLabRuntimeAdapter()
    base = {"host": "127.0.0.1", "ports": {"ml_lab": 8769}}
    context_a = {**base, "project_root": first, "results_root": first / "ml_lab_results"}
    context_b = {**base, "project_root": second, "results_root": second / "ml_lab_results"}
    try:
        adapter.launch(context=context_a)
        assert configuration[-1]["ML_LAB_EXPERIMENT_PROJECT_ROOT"] == str(first.resolve())
        assert configuration[-1]["ML_LAB_PATH_AUTOCOMPLETE_ROOTS"] == [str(first.resolve())]
        assert configuration[-1]["ML_LAB_DERIVED_OUTPUT_ROOT"] == str(first / "analysis" / "data" / "derived")
        assert adapter.status(context=context_a)["running"] is True
        inactive = adapter.status(context=context_b)
        assert inactive["running"] is False and inactive["url"] is None
        assert inactive["metadata"]["bound_project"] == str(first.resolve())

        adapter.launch(context=context_b)
        assert servers[0].closed is True
        assert configuration[-1]["ML_LAB_EXPERIMENT_PROJECT_ROOT"] == str(second.resolve())
        assert adapter.status(context=context_b)["running"] is True
        changed_root = {**context_b, "results_root": second / "custom-results"}
        assert adapter.status(context=changed_root)["running"] is False
        adapter.launch(context=changed_root)
        assert configuration[-1]["ML_LAB_DERIVED_OUTPUT_ROOT"] == str(second / "custom-results" / "data" / "derived")
        assert adapter.status(context=changed_root)["running"] is True
    finally:
        adapter.shutdown()
    assert all(s.closed for s in servers)


def test_discovery_exposes_only_verified_completed_local_experiment_artifacts(tmp_path):
    project = tmp_path / "project"
    store, exp = _prepared_training(project)
    run = run_configured_experiment(store, exp["experiment_id"])
    context = {"project_root": project, "results_root": project / "analysis"}
    records = discover_artifacts(context)
    selected = [item for item in records if item["schema_id"] == "ml-lab.registered-experiment-output"]
    assert len(selected) == 1
    result = selected[0]
    assert result["kind"] == "training_result"
    assert result["metadata"]["run_id"] == run["run_id"]
    assert result["metadata"]["revision"] == exp["revision"]
    assert result["validation_state"] == "valid"
    assert Path(result["location"]).is_relative_to(project)
    assert result["project_id"] == str(project.resolve())
    # A completed run is not permission to advertise a file modified later.
    Path(result["location"]).write_text('altered', encoding="utf-8")
    assert result["artifact_id"] not in {a["artifact_id"] for a in discover_artifacts(context)}
    other = tmp_path / "other"
    other.mkdir()
    assert not discover_artifacts({"project_root": other, "results_root": other / "analysis"})


def test_discovery_from_experiment_db_even_if_custom_results_path_missing(tmp_path):
    project = tmp_path / "project"
    store, exp = _prepared_training(project)
    run_configured_experiment(store, exp["experiment_id"])
    records = discover_artifacts({"project_root": project, "results_root": project / "unused-results"})
    assert any(row["kind"] == "training_result" for row in records)


def test_discovery_ignores_external_manifest_locations_and_symlinks(tmp_path):
    project = tmp_path / "project"
    analysis = project / "analysis"
    analysis.mkdir(parents=True)
    outside = tmp_path / "outside.csv"
    outside.write_text("private,content\n1,2\n", encoding="utf-8")
    forged = {"schema": "ml-lab.derived-dataset@1", "derived": {"path": str(outside)}}
    (analysis / "forged.json").write_text(json.dumps(forged), encoding="utf-8")
    (analysis / "linked.json").symlink_to(tmp_path / "outside.json")
    (tmp_path / "outside.json").write_text(json.dumps(forged), encoding="utf-8")
    assert not discover_artifacts({"project_root": project, "results_root": analysis})


def test_discovery_never_advertises_queued_cancelled_or_failed_runs(tmp_path):
    from ml_lab.experiments import queue_configured_run, request_run_cancellation
    project = tmp_path / "project"
    store, exp = _prepared_training(project)
    queued = queue_configured_run(store, exp["experiment_id"])
    context = {"project_root": project, "results_root": project / "analysis"}
    assert not [r for r in discover_artifacts(context) if r["schema_id"] == "ml-lab.registered-experiment-output"]
    request_run_cancellation(store, queued["run_id"])
    assert not [r for r in discover_artifacts(context) if r["schema_id"] == "ml-lab.registered-experiment-output"]


def test_discovery_includes_synthetic_csv_with_reproducible_run_metadata(tmp_path):
    from ml_lab.experiments import generate_synthetic_dataset
    project = tmp_path / "project"
    store, exp = _prepared_training(project)
    # Bootstrap does not require a model checkpoint or a second repository.
    generated = generate_synthetic_dataset(store, exp["experiment_id"],
                                          {"method": "bootstrap", "count": 9, "seed": 12},
                                          expected_revision=exp["revision"])
    context = {"project_root": project, "results_root": project / "analysis"}
    rows = [r for r in discover_artifacts(context) if r["kind"] == "synthetic_dataset"]
    assert len(rows) == 1
    assert rows[0]["metadata"]["run_id"] == generated["run_id"]
    assert rows[0]["media_type"] == "text/csv"
    assert Path(rows[0]["location"]).read_text(encoding="utf-8").startswith("a,b\n")
    assert rows[0]["metadata"]["sha256"] == generated["dataset"]["sha256"]
