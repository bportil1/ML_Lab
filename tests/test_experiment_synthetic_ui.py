"""Optional Flask routes for S6.1's synthetic-data workflow."""
from __future__ import annotations

import time
import pytest

flask = pytest.importorskip("flask")
from ml_lab.experiments import ExperimentStore, select_experiment_dataset
from ml_lab.ui import create_ui_blueprint


def test_synthetic_launch_poll_and_verified_download(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "matrix.csv").write_text("a,b\n1,2\n2,4\n3,6\n4,8\n", encoding="utf-8")
    store = ExperimentStore(root)
    selected = select_experiment_dataset(store, root / "matrix.csv")
    app = flask.Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(create_ui_blueprint(name="s61", experiment_project_root=root), url_prefix="/ml")
    client = app.test_client()
    url = f"/ml/experiments/synthetic?experiment_id={selected['experiment_id']}"
    page = client.get(url)
    assert page.status_code == 200
    assert b"Correlated Gaussian" in page.data
    response = client.post("/ml/experiments/synthetic/run", data={
        "experiment_id": selected["experiment_id"], "expected_revision": "1",
        "method": "bootstrap", "count": "12", "seed": "17", "degrees_of_freedom": "5",
    })
    assert response.status_code == 302
    target = response.headers["Location"]
    run_id = target.rsplit("/", 1)[-1]
    for _ in range(100):
        status = client.get(target + "/status").get_json()
        if status["run"]["status"] in {"succeeded", "failed", "cancelled"}:
            break
        time.sleep(0.05)
    assert status["run"]["status"] == "succeeded", status
    complete = client.get(target)
    assert complete.status_code == 200
    assert b"Use generated data in an experiment" in complete.data
    downloaded = client.get(target + "/synthetic.csv")
    assert downloaded.status_code == 200
    assert downloaded.data.startswith(b"a,b")
    assert len(downloaded.data.splitlines()) == 13
    artifact = next(a for a in store.artifacts.list_for_run(run_id) if a["kind"] == "dataset" and a["role"] == "output")
    (root / artifact["path"]).write_text("changed", encoding="utf-8")
    assert client.get(target + "/synthetic.csv").status_code == 404
    assert client.post("/ml/experiments/synthetic/run", data={
        "experiment_id": selected["experiment_id"], "expected_revision": "1",
        "method": "tangent_manifold", "count": "8", "seed": "1",
    }).status_code == 400
