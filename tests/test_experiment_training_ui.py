"""Optional Flask integration for S4.1's configured model-run interface."""
from __future__ import annotations

import time

import pytest

flask = pytest.importorskip("flask")
from ml_lab.experiments import (  # noqa: E402
    ExperimentStore, configure_experiment_model, select_experiment_dataset,
)
from ml_lab.ui import create_ui_blueprint  # noqa: E402


def make_client(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "matrix.csv").write_text("a,b\n1,2\n2,4\n3,6\n4,8\n", encoding="utf-8")
    store = ExperimentStore(root)
    selected = select_experiment_dataset(store, root / "matrix.csv", feature_columns=["a", "b"])
    item = configure_experiment_model(
        store, selected["experiment_id"], expected_revision=selected["revision"],
        family="representation", method="pca", settings={"config": {"n_components": 1}},
    )
    app = flask.Flask(__name__)
    app.register_blueprint(create_ui_blueprint(name="s41", experiment_project_root=root), url_prefix="/ml")
    return app.test_client(), store, item


def test_launch_poll_and_verified_result(tmp_path):
    client, store, experiment = make_client(tmp_path)
    url = f"/ml/experiments/models?experiment_id={experiment['experiment_id']}"
    page = client.get(url)
    assert page.status_code == 200
    assert b"Run configured model" in page.data
    response = client.post("/ml/experiments/models/run", data={
        "experiment_id": experiment["experiment_id"], "expected_revision": str(experiment["revision"]),
    })
    assert response.status_code == 302
    view_url = response.headers["Location"]
    assert "/experiments/runs/" in view_url
    run_id = view_url.split("/")[-1]
    view = client.get(view_url)
    assert view.status_code == 200
    assert b"Model run" in view.data
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        status = client.get(view_url + "/status").get_json()
        if status["run"]["status"] in {"succeeded", "failed"}:
            break
        time.sleep(.04)
    assert status["run"]["status"] == "succeeded", status
    assert {a["kind"] for a in status["artifacts"]} == {"dataset", "manifest", "result"}
    result = client.get(view_url + "/result")
    assert result.status_code == 200
    assert result.get_json()["task"] == "representation"
    old = client.post("/ml/experiments/models/run", data={
        "experiment_id": experiment["experiment_id"], "expected_revision": "1",
    })
    assert old.status_code == 400
    assert b"revision changed" in old.data
    assert client.get("/ml/experiments/runs/" + "f" * 32).status_code == 404
    assert client.get(view_url + "/result").status_code == 200


def test_unconfigured_run_rejected_in_ui(tmp_path):
    client, store, configured = make_client(tmp_path)
    selected = select_experiment_dataset(store, store.project_root / "matrix.csv",
                                         feature_columns=["a", "b"])
    rejected = client.post("/ml/experiments/models/run", data={
        "experiment_id": selected["experiment_id"], "expected_revision": "1",
    })
    assert rejected.status_code == 400
    assert b"configuration" in rejected.data.lower()
