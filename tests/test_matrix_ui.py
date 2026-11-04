from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("flask")
from flask import Flask
from ml_lab.ui import create_ui_blueprint
from ml_lab import data


def test_matrix_editor_mounts_and_saves_derived_dataset(tmp_path):
    original = tmp_path / "source.csv"
    original.write_text("first,second\n1,alpha\n2,beta\n", encoding="utf-8")
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(create_ui_blueprint(name="matrix_test", derived_output_root=tmp_path / "analysis/data/derived"), url_prefix="/module")
    client = app.test_client()
    discovered = client.post("/module/data", data={"paths": str(original)}).get_data(as_text=True)
    match = re.search(r'href="([^"]+/data/source/[^"/]+)"', discovered)
    assert match, "Source link should be discoverable in Data Lab"
    source_url = match.group(1).replace('&amp;', '&')
    page = client.get(source_url)
    assert page.status_code == 200
    assert b"Edit a copy as matrix" in page.data
    matrix_url = source_url + "/matrix"
    matrix = client.get(matrix_url)
    assert matrix.status_code == 200
    assert b"matrix-draft" in matrix.data
    assert b"ml_lab_matrix_editor.js" in matrix.data
    sha = data.inspect_file(original).sha256
    result = client.post(matrix_url + "/save", json={"source_sha256": sha,
                         "edits": [{"row": 2, "column": 1, "value": "gamma"}]})
    assert result.status_code == 201, result.get_data(as_text=True)
    payload = result.get_json()
    derived = Path(payload["path"])
    assert derived.read_text() == "first,second\n1,alpha\n2,gamma\n"
    assert original.read_text() == "first,second\n1,alpha\n2,beta\n"
    assert client.get(payload["source_url"]).status_code == 200
    assert client.get(payload["provenance_url"]).status_code == 200
    stale = client.post(matrix_url + "/save", json={"source_sha256": "0" * 64,
                        "edits": [{"row": 2, "column": 1, "value": "delta"}]})
    assert stale.status_code == 400


def test_matrix_editor_rejects_invalid_payload(tmp_path):
    original = tmp_path / "source.csv"
    original.write_text("x,y\n1,2\n", encoding="utf-8")
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(create_ui_blueprint(name="matrix_test", derived_output_root=tmp_path / "derived"), url_prefix="/module")
    client = app.test_client()
    inventory = client.post("/module/data", data={"paths": str(original)}).get_data(as_text=True)
    match = re.search(r'href="([^"]+/data/source/[^"/]+)"', inventory)
    url = match.group(1).replace('&amp;', '&') + "/matrix/save"
    assert client.post(url, data="[]", content_type="application/json").status_code == 400
    assert client.post(url, json={"edits": [], "source_sha256": "nope"}).status_code == 400
