"""Optional Flask smoke test: chart routes remain local and read-only."""
from __future__ import annotations

from urllib.parse import urlencode

import pytest

flask = pytest.importorskip("flask")

from ml_lab.experiments import (  # noqa: E402
    ExperimentStore, configure_experiment_model, execute_configured_run,
    queue_configured_run, select_experiment_dataset,
)
from ml_lab.ui import create_ui_blueprint  # noqa: E402


def test_chart_page_svg_csv_json_and_invalid_selection(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    source = project / "source.csv"
    source.write_text("a,b,c\n1,2,3\n2,5,6\n3,6,2\n4,9,3\n5,10,6\n6,11,7\n", encoding="utf-8")
    store = ExperimentStore(project)
    exp = select_experiment_dataset(store, source, feature_columns=["a", "b", "c"])
    exp = configure_experiment_model(store, exp["experiment_id"], family="representation", method="pca",
                                     settings={"config": {"n_components": 1}}, expected_revision=exp["revision"])
    run_ids = []
    for _ in range(2):
        run = queue_configured_run(store, exp["experiment_id"])
        execute_configured_run(store, run["run_id"])
        run_ids.append(run["run_id"])
    app = flask.Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(create_ui_blueprint(name="s52", experiment_project_root=project), url_prefix="/ml")
    client = app.test_client()
    query = "?" + urlencode([("experiment_id", exp["experiment_id"])]
                            + [("run_id", run) for run in run_ids])
    page = client.get("/ml/experiments/compare" + query)
    assert page.status_code == 200
    assert b"Scientific visualizations" in page.data
    assert b"Export chart data + provenance" in page.data
    payload = client.get("/ml/experiments/visualizations.json" + query)
    assert payload.status_code == 200
    report = payload.get_json()
    assert report["charts"]
    assert report["selected_runs"] == run_ids
    chart_id = report["charts"][0]["id"]
    svg = client.get(f"/ml/experiments/charts/{chart_id}.svg" + query)
    assert svg.status_code == 200
    assert svg.mimetype == "image/svg+xml"
    assert b"<svg" in svg.data and b"<script" not in svg.data
    assert "no-store" in svg.headers["Cache-Control"]
    csv = client.get(f"/ml/experiments/charts/{chart_id}.csv" + query)
    assert csv.status_code == 200 and csv.data.startswith(b"x,label,y\r\n")
    assert client.get("/ml/experiments/charts/missing.svg" + query).status_code == 404
    assert client.get("/ml/experiments/visualizations.json?experiment_id=nope").status_code == 404
