from __future__ import annotations

import pytest

pytest.importorskip("flask")

from flask import Flask

from ml_lab.ui import create_app, create_ui_blueprint


def test_standalone_ui_renders_dashboard_and_capability_api():
    app = create_app(config={"TESTING": True})
    client = app.test_client()

    response = client.get("/")
    assert response.status_code == 200
    assert b"Machine Learning Lab" in response.data
    assert b"Data Lab" in response.data

    payload = client.get("/api/capabilities").get_json()
    assert payload["version"]
    assert any(record["id"] == "classification" for record in payload["capabilities"])


def test_ui_blueprint_mounts_under_host_prefix_without_route_assumptions():
    host = Flask(__name__)
    host.config["TESTING"] = True
    host.register_blueprint(create_ui_blueprint(name="mounted_ml_lab"), url_prefix="/tools/ml-lab")
    client = host.test_client()

    response = client.get("/tools/ml-lab/")
    assert response.status_code == 200
    assert b"ML Lab" in response.data
    assert b"/tools/ml-lab/static/ml_lab.css" in response.data

    task = client.get("/tools/ml-lab/task/clustering")
    assert task.status_code == 200
    assert b"Application payload" in task.data


def test_ui_task_executes_through_shared_application_service():
    app = create_app(config={"TESTING": True})
    client = app.test_client()
    payload = '''{
      "X": [[0, 0], [0.1, 0.1], [5, 5], [5.1, 5.1]],
      "feature_names": ["x", "y"],
      "estimators": ["kmeans"],
      "config": {"repeats": 1, "random_state": 7}
    }'''
    response = client.post("/task/clustering", data={"payload": payload})
    assert response.status_code == 200
    assert b'&quot;task&quot;: &quot;clustering&quot;' in response.data


def test_experimental_ui_is_hidden_unless_enabled():
    base = create_app(config={"TESTING": True}).test_client()
    assert base.get("/task/experimental").status_code == 404

    enabled = create_app(enable_experimental=True, config={"TESTING": True}).test_client()
    assert enabled.get("/task/experimental").status_code == 200


def test_data_lab_mounts_and_inspects_host_local_path(tmp_path):
    source = tmp_path / "data.csv"
    source.write_text("index,value\n0,10\n1,20\n", encoding="utf-8")

    host = Flask(__name__)
    host.config["TESTING"] = True
    host.register_blueprint(create_ui_blueprint(name="data_ml_lab"), url_prefix="/ml")
    client = host.test_client()

    landing = client.get("/ml/data")
    assert landing.status_code == 200
    assert b"Data intake &amp; inventory" in landing.data

    response = client.post(
        "/ml/data",
        data={"paths": str(source), "recursive": "1", "preview_rows": "20"},
    )
    assert response.status_code == 200
    assert "1 × 2".encode("utf-8") in response.data
    assert b"likely exported index column" in response.data
    assert b"ml-lab.data-inventory@1" in response.data
