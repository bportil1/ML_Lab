"""Optional Flask route checks for Sprint 5.1 comparison."""
from __future__ import annotations

import pytest

flask = pytest.importorskip("flask")
from ml_lab.experiments import ExperimentStore, configure_experiment_model, select_experiment_dataset
from ml_lab.ui import create_ui_blueprint


def test_compare_routes_display_pinned_runs_and_export_csv(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    source = project / "source.csv"
    source.write_text("x,y\n1,2\n3,4\n5,6\n", encoding="utf-8")
    store = ExperimentStore(project)
    experiment = select_experiment_dataset(store, source, feature_columns=["x", "y"])
    experiment = configure_experiment_model(store, experiment["experiment_id"],
                                            family="representation", method="pca",
                                            settings={"config": {"n_components": 1}},
                                            expected_revision=experiment["revision"])
    from ml_lab.experiments import execute_configured_run, queue_configured_run
    queued = queue_configured_run(store, experiment["experiment_id"])
    execute_configured_run(store, queued["run_id"])
    app = flask.Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(create_ui_blueprint(name="s51", experiment_project_root=project),
                           url_prefix="/ml")
    client = app.test_client()
    query = f"?experiment_id={experiment['experiment_id']}&run_id={queued['run_id']}"
    page = client.get("/ml/experiments/compare" + query)
    assert page.status_code == 200 and b"Compare experiment runs" in page.data
    assert b"rmse" in page.data and queued["run_id"][:10].encode() in page.data
    output = client.get("/ml/experiments/compare.csv" + query)
    assert output.status_code == 200
    assert b"dataset_sha256" in output.data and queued["run_id"].encode() in output.data
    assert b"Compare experiment runs" in client.get(
        "/ml/experiments/models?experiment_id=" + experiment["experiment_id"]).data
    assert client.get("/ml/experiments/compare?experiment_id=unknown").status_code == 404
