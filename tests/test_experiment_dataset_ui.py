"""S2.3 optional Flask UI: explicit, signed dataset -> experiment revisions."""
from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

flask = pytest.importorskip("flask")
from ml_lab import data
from ml_lab.experiments import ExperimentStore
from ml_lab.ui import create_ui_blueprint


def _client(project: Path):
    app = flask.Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(create_ui_blueprint(
        name="s23", derived_output_root=project / "analysis/data/derived",
        experiment_project_root=project,
    ), url_prefix="/ml")
    return app.test_client()


def _token_from_source_page(client, source: Path) -> str:
    # Source tokens must be issued by the mounted UI, never fabricated.
    page = client.post("/ml/data", data={"paths": str(source)}).get_data(as_text=True)
    import re
    match = re.search(r'href="([^"]+/data/source/[^"/]+)"', page)
    assert match
    return match.group(1).rsplit("/", 1)[-1]


def test_matrix_save_to_experiment_with_immutable_revision(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    source = project / "data.csv"
    source.write_text("a,b,label\n1,2,cat\n3,4,dog\n")
    client = _client(project)
    token = _token_from_source_page(client, source)
    selection = f"/ml/data/experiment/selection?source_token={token}"
    assert b"Use in experiment" in client.get(f"/ml/data/source/{token}").data
    preview = client.get(selection)
    assert preview.status_code == 200
    assert b"Create experiment with dataset" in preview.data
    sha = data.inspect_file(source).sha256
    create = client.post("/ml/data/experiment/selection", data={
        "source_token": token, "source_sha256": sha, "provenance_sha256": "",
        "experiment_name": "A", "features": ["a", "b"], "targets": ["label"],
    })
    assert create.status_code == 200, create.get_data(as_text=True)
    store = ExperimentStore(project)
    first = store.list_experiments()[0]
    assert first["revision"] == 1
    assert first["dataset_ref"]["target_columns"] == ["label"]

    saved = client.post(f"/ml/data/source/{token}/matrix/save", json={
        "source_sha256": sha, "edits": [{"row": 1, "column": 0, "value": "11"}],
    })
    assert saved.status_code == 201, saved.get_data(as_text=True)
    new_url = saved.get_json()["experiment_url"]
    assert new_url
    selected_page = client.get(new_url + "&experiment_id=" + first["experiment_id"])
    assert selected_page.status_code == 200
    assert b"revision 1" in selected_page.data
    derived_token = parse_qs(urlsplit(new_url).query)["source_token"][0]
    derived_path = Path(saved.get_json()["path"])
    provenance_sha = data.describe_dataset(derived_path)  # verify created file
    assert provenance_sha["sha256"] != sha
    event_hash = client.get(new_url).data
    assert b"Data Lab provenance" in event_hash
    from ml_lab.experiments import describe_selection
    pinned = describe_selection(store, derived_path)
    revise = client.post("/ml/data/experiment/selection", data={
        "source_token": derived_token, "source_sha256": pinned["sha256"],
        "provenance_sha256": pinned["provenance"]["sha256"],
        "experiment_id": first["experiment_id"], "expected_revision": "1",
        "features": ["a", "b"], "targets": ["label"],
    })
    assert revise.status_code == 200, revise.get_data(as_text=True)
    assert store.get_experiment(first["experiment_id"])["revision"] == 2
    assert store.get_experiment(first["experiment_id"], revision=1)["dataset_ref"]["sha256"] == sha
    retry = client.post("/ml/data/experiment/selection", data={
        "source_token": token, "source_sha256": sha, "provenance_sha256": "",
        "experiment_id": first["experiment_id"], "expected_revision": "1",
        "features": ["a", "b"], "targets": ["label"],
    })
    assert retry.status_code == 400
    assert b"revision changed" in retry.data
    assert store.get_experiment(first["experiment_id"])["revision"] == 2


def test_selection_rejects_stale_source_and_external_file(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    source = project / "input.tsv"
    source.write_text("x\ty\n1\t2\n")
    client = _client(project)
    token = _token_from_source_page(client, source)
    original_hash = data.inspect_file(source).sha256
    source.write_text("x\ty\n7\t8\n")
    response = client.post("/ml/data/experiment/selection", data={
        "source_token": token, "source_sha256": original_hash,
        "experiment_name": "Stale", "features": ["x", "y"],
    })
    assert response.status_code == 400
    assert b"changed" in response.data
    assert ExperimentStore(project).list_experiments() == []
    outside = tmp_path / "outside.csv"
    outside.write_text("x\n1\n")
    outside_token = _token_from_source_page(client, outside)
    assert b"Use in experiment" not in client.get(f"/ml/data/source/{outside_token}").data
    assert client.get(f"/ml/data/experiment/selection?source_token={outside_token}").status_code == 400


def test_unknown_experiment_id_returns_form_error(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    source = project / "source.csv"
    source.write_text("x\n1\n")
    client = _client(project)
    token = _token_from_source_page(client, source)
    response = client.get(f"/ml/data/experiment/selection?source_token={token}&experiment_id=missing")
    assert response.status_code == 400
    assert b"Unknown experiment" in response.data
