"""Optional Flask checks for the S4.2 optimization workflow."""
from __future__ import annotations

import time
import pytest

flask = pytest.importorskip("flask")

from ml_lab.experiments import ExperimentStore, configure_experiment_model, select_experiment_dataset
from ml_lab.ui import create_ui_blueprint


def client_for(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "sample.csv").write_text("a,b,c\n1,2,3\n2,5,6\n3,6,2\n4,9,3\n5,10,6\n6,11,7\n")
    store = ExperimentStore(root)
    e = select_experiment_dataset(store, root / "sample.csv", feature_columns=["a", "b", "c"])
    e = configure_experiment_model(store, e["experiment_id"], family="representation", method="pca",
                                   settings={"config": {"n_components": 1}}, expected_revision=e["revision"])
    app = flask.Flask(__name__)
    app.register_blueprint(create_ui_blueprint(name="s42", experiment_project_root=root), url_prefix="/ml")
    return app.test_client(), e


def test_optimize_launch_status_and_verified_result(tmp_path):
    client, experiment = client_for(tmp_path)
    page = client.get(f"/ml/experiments/models?experiment_id={experiment['experiment_id']}")
    assert page.status_code == 200
    assert b"Start optimization trial" in page.data
    response = client.post("/ml/experiments/models/optimize", data={
        "experiment_id": experiment["experiment_id"], "expected_revision": str(experiment["revision"]),
        "parameter": "config.n_components", "lower": "1", "upper": "2", "metric": "rmse",
        "goal": "minimize", "population_size": "2", "iterations": "0", "seed": "9",
    })
    assert response.status_code == 302
    target = response.headers["Location"]
    assert "experiments/runs" in target
    status = {}
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        status = client.get(target + "/status").get_json()
        if status["run"]["status"] in {"succeeded", "failed"}:
            break
        time.sleep(.05)
    assert status["run"]["status"] == "succeeded", status
    assert status["run"]["environment"]["executor"].endswith("optimization@1")
    result = client.get(target + "/result")
    assert result.status_code == 200
    assert result.get_json()["algorithm"] == "firefly"
    assert client.get(target).status_code == 200


def test_bad_optimization_plan_and_unknown_run(tmp_path):
    client, experiment = client_for(tmp_path)
    data = {"experiment_id": experiment["experiment_id"], "expected_revision": str(experiment["revision"]),
            "parameter": "config.random_state", "lower": "1", "upper": "2", "metric": "rmse",
            "goal": "minimize", "population_size": "2", "iterations": "0", "seed": "9"}
    bad = client.post("/ml/experiments/models/optimize", data=data)
    assert bad.status_code == 400
    assert b"numeric" in bad.data.lower()
    assert client.post("/ml/experiments/runs/" + "f" * 32 + "/cancel").status_code == 404
