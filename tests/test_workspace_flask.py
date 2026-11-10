"""Optional Flask integration checks; skip in the base/headless install."""
from __future__ import annotations

import pytest

pytest.importorskip("flask")
from flask import Flask
from ml_lab.experiments import ExperimentStore
from ml_lab.ui import create_ui_blueprint


def _app(project, *, name="research_lab"):
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(create_ui_blueprint(
        name=name, experiment_project_root=project,
        path_autocomplete_roots=[project],
    ), url_prefix="/lab")
    return app


def _experiment(store, name):
    return store.create_experiment(
        name, dataset_ref={"schema": "ml-lab.experiment-dataset@1",
                           "path": str(store.project_root / "source.csv"),
                           "feature_columns": ["feature"], "target_columns": []},
    )


def test_workspace_available_without_dataset_or_stage_completion(tmp_path):
    client = _app(tmp_path).test_client()
    page = client.get("/lab/workspace")
    assert page.status_code == 200
    for title in (b"Data", b"Models", b"Representation", b"EBM", b"Optimization",
                  b"Generative", b"XML", b"Results"):
        assert title in page.data
    assert b"No experiment selected" in page.data
    assert b"/lab/static/ml_lab_workspace.js" in page.data
    assert b"data-workspace-scope=" in page.data
    assert client.get("/lab/workspace/results").status_code == 200
    assert client.get("/lab/task/clustering").status_code == 200
    assert client.get("/lab/workspace/context").status_code == 400


def test_project_context_validates_experiment_and_run_ownership(tmp_path):
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    store_a = ExperimentStore(alpha)
    store_b = ExperimentStore(beta)
    first = _experiment(store_a, "first")
    second = _experiment(store_a, "second")
    foreign = _experiment(store_b, "foreign")
    run_first = store_a.create_run(first["experiment_id"], environment={
        "executor": "ml_lab.experiments.training@1"})
    run_second = store_a.create_run(second["experiment_id"], environment={
        "executor": "ml_lab.experiments.training@1"})
    client = _app(alpha).test_client()

    data = client.get("/lab/workspace/context", query_string={
        "experiment_id": first["experiment_id"], "run_id": run_first["run_id"]})
    assert data.status_code == 200
    assert data.headers["Cache-Control"] == "no-store"
    assert data.get_json()["run_id"] == run_first["run_id"]
    assert data.get_json()["dataset"].endswith("source.csv")
    assert len(data.get_json()["runs"]) == 1
    assert client.get("/lab/workspace/context", query_string={
        "experiment_id": first["experiment_id"], "run_id": run_second["run_id"]}).status_code == 404
    assert client.get("/lab/workspace/context", query_string={
        "experiment_id": foreign["experiment_id"]}).status_code == 404
    assert client.get("/lab/workspace/results", query_string={
        "experiment_id": foreign["experiment_id"]}).status_code == 404
    assert client.get("/lab/workspace/results", query_string={
        "experiment_id": first["experiment_id"]}).status_code == 200
    assert client.get("/lab/workspace/results", query_string={
        "experiment_id": first["experiment_id"]}).data.count(b"Inspect run") == 1
    assert len(store_a.list_runs(first["experiment_id"])) == 1


def test_legacy_tasks_and_host_mount_routes_remain_unchanged(tmp_path):
    client = _app(tmp_path, name="mounted_workspace").test_client()
    assert client.get("/lab/data").status_code == 200
    assert client.get("/lab/experiments/models").status_code == 200
    assert client.get("/lab/data/xml/collection").status_code == 200
    assert b'data-pah-tool-nav' in client.get("/lab/").data
    assert b'/lab/workspace' in client.get("/lab/").data
    assert client.get("/lab/task/representation").status_code == 200
