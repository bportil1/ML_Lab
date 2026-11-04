"""S3.2 UI integration when optional Flask is installed."""
from __future__ import annotations

from pathlib import Path

import pytest

flask = pytest.importorskip("flask")
from ml_lab.experiments import ExperimentStore, select_experiment_dataset
from ml_lab.ui import create_ui_blueprint


def _client(tmp_path: Path, target: bool = True):
    project = tmp_path / "project"
    project.mkdir()
    source = project / "sample.csv"
    source.write_text("a,b,label\n1,2,one\n3,4,two\n5,6,one\n", encoding="utf-8")
    store = ExperimentStore(project)
    selected = select_experiment_dataset(store, source, experiment_name="Demo",
                                         feature_columns=["a", "b"],
                                         target_columns=["label"] if target else [])
    app = flask.Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(create_ui_blueprint(
        name="s32", experiment_project_root=project,
    ), url_prefix="/ml")
    return app.test_client(), store, selected


def _model_url(record, **kwargs):
    from urllib.parse import urlencode
    return "/ml/experiments/models?" + urlencode({"experiment_id": record["experiment_id"], **kwargs})


def test_model_page_renders_typed_configuration_and_retains_legacy_routes(tmp_path):
    client, store, item = _client(tmp_path)
    page = client.get(_model_url(item, family="classification", method="ridge"))
    assert page.status_code == 200
    assert b"Model workspace" in page.data
    assert b'ctl.config.cv_folds' in page.data
    assert b"Save model configuration" in page.data
    assert "3 · Training".encode("utf-8") in page.data
    assert b"Experiments" in client.get("/ml/").data
    assert b"All capabilities" not in page.data  # This is a dedicated model workspace.
    assert store.get_experiment(item["experiment_id"])["revision"] == 1  # GET is read-only


def test_post_creates_immutable_revision_without_running(tmp_path):
    client, store, item = _client(tmp_path)
    url = "/ml/experiments/models"
    payload = {
        "experiment_id": item["experiment_id"], "expected_revision": "1",
        "family": "classification", "method": "ridge",
        "ctl.config.cv_folds": "3", "ctl.config.refit": "false",
        "ctl.task_options.test_size": "0.25",
    }
    response = client.post(url, data=payload)
    assert response.status_code == 302
    latest = store.get_experiment(item["experiment_id"])
    assert latest["revision"] == 2
    assert latest["model_spec"]["family"] == "classification"
    assert latest["model_spec"]["settings"]["config"] == {"cv_folds": 3, "refit": False}
    assert latest["model_spec"]["settings"]["test_size"] == .25
    assert store.get_experiment(item["experiment_id"], revision=1)["model_spec"] == {}
    assert store.list_runs(item["experiment_id"]) == []
    page = client.get(response.headers["Location"])
    assert b"Model configuration saved" in page.data
    assert b'ctl.config.cv_folds' in page.data
    assert b'value="3"' in page.data

    # An old tab must not overwrite revision 2.
    stale = client.post(url, data=payload)
    assert stale.status_code == 400
    assert b"revision changed" in stale.data.lower()
    assert store.get_experiment(item["experiment_id"])["revision"] == 2


def test_bad_payload_and_unsuitable_dataset_cannot_publish(tmp_path):
    client, store, item = _client(tmp_path, target=False)
    url = "/ml/experiments/models"
    invalid = client.post(url, data={
        "experiment_id": item["experiment_id"], "expected_revision": "1",
        "family": "classification", "method": "ridge", "ctl.config.cv_folds": "1",
    })
    assert invalid.status_code == 400
    assert b"target column" in invalid.data.lower()
    assert store.get_experiment(item["experiment_id"])["revision"] == 1

    correct = client.post(url, data={
        "experiment_id": item["experiment_id"], "expected_revision": "1",
        "family": "representation", "method": "pca", "ctl.config.n_components": "1",
    })
    assert correct.status_code == 302
    assert store.get_experiment(item["experiment_id"])["revision"] == 2
    assert store.get_experiment(item["experiment_id"])["model_spec"]["method"] == "pca"
