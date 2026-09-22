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
    assert b">Compare</a>" in response.data
    assert b">Transform</a>" in response.data

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
    assert b"/tools/ml-lab/static/pah-module-theme.css" in response.data
    assert b"/tools/ml-lab/static/ml_lab_pah_compat.css" in response.data
    assert response.data.index(b"ml_lab.css") < response.data.index(b"pah-module-theme.css") < response.data.index(b"ml_lab_pah_compat.css")
    assert b'data-pah-module-root' in response.data
    assert b'data-pah-tool-nav' in response.data

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
    assert b"Data intake &amp; profiling" in landing.data

    response = client.post(
        "/ml/data",
        data={"paths": str(source), "recursive": "1", "preview_rows": "20"},
    )
    assert response.status_code == 200
    assert "1 × 2".encode("utf-8") in response.data
    assert b"likely exported index column" in response.data
    assert b"ml-lab.data-inventory@1" in response.data


def test_data_lab_profiles_with_visible_job_state_and_persisted_profile(tmp_path):
    import time

    source = tmp_path / "profile.csv"
    source.write_text("x,y,label\n1,2,A\n2,4,A\n3,6,B\n", encoding="utf-8")

    app = create_app(config={"TESTING": True, "ML_LAB_PROFILE_OUTPUT_ROOT": str(tmp_path / "runs")})
    client = app.test_client()
    started = client.post(
        "/data/profile/start",
        data={
            "paths": str(source),
            "recursive": "1",
            "preview_rows": "20",
            "max_rows": "100000",
            "relationship_rows": "0",
            "max_relationship_columns": "25",
        },
    )
    assert started.status_code == 302
    job_url = started.headers["Location"]
    assert "/data/profile/job/" in job_url
    status_url = job_url + "/status"

    deadline = time.time() + 10
    payload = client.get(status_url).get_json()
    while payload["status"] not in {"completed", "failed"} and time.time() < deadline:
        time.sleep(0.02)
        payload = client.get(status_url).get_json()

    assert payload["status"] == "completed", payload.get("error")
    assert payload["profile_links"]
    profile = client.get(payload["profile_links"][0]["url"])
    assert profile.status_code == 200
    assert b"Persisted statistical profile" in profile.data
    assert b"Pearson/Spearman" in profile.data
    assert b"Open source" in profile.data
    assert b"Profile visual summary" in profile.data
    assert b"Missingness" in profile.data
    assert b"Numeric spread" in profile.data
    assert b"Relationship strength" in profile.data


def test_data_transform_workspace_previews_and_applies_typed_recipe(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("name,trial,label\n A ,1,aug_3\n B ,x,aug_6\n", encoding="utf-8")
    output = tmp_path / "derived.csv"
    app = create_app(config={"TESTING": True})
    client = app.test_client()

    page = client.get("/data/transform")
    assert page.status_code == 200
    assert b"Controlled transformation" in page.data

    form = {
        "source": str(source),
        "output": str(output),
        "recipe_name": "UI cleanup",
        "type_overrides": "trial=integer",
        "coerce_errors": "coerce",
        "clean_columns": "name",
        "derive_source": "label",
        "derive_target": "augmentation",
        "derive_pattern": r"aug_(\\d+)",
        "derive_group": "0",
        "preview_rows": "50",
        "mode": "preview",
    }
    preview = client.post("/data/transform", data=form)
    assert preview.status_code == 200
    assert b"Operation diagnostics" in preview.data
    assert b"coerce_types" in preview.data
    assert not output.exists()

    form["mode"] = "apply"
    applied = client.post("/data/transform", data=form)
    assert applied.status_code == 200
    assert b"Derived dataset written" in applied.data
    assert output.is_file()


def test_data_compare_workspace(tmp_path):
    app = create_app(config={"TESTING": True})
    client = app.test_client()
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    left.write_text("id,x\n1,10\n2,20\n", encoding="utf-8")
    right.write_text("id,x\n1,10\n2,21\n", encoding="utf-8")
    response = client.post(
        "/data/compare",
        data={"paths": f"{left}\n{right}", "recursive": "1", "max_pairs": "200"},
    )
    assert response.status_code == 200
    assert b"Dataset comparison" in response.data
    assert b"same_schema_distinct_data" in response.data or b"likely_version_or_derivative" in response.data
    assert b"Pair overview" in response.data
    assert b"Schema" in response.data
    assert b"Left rows" in response.data
