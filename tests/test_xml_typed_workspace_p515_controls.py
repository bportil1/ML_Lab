"""P5-15 workspace publication contract and optional Flask end-to-end smoke test."""
from pathlib import Path

UI = Path(__file__).resolve().parents[1] / "src/ml_lab/ui"


def test_workspace_has_explicit_separate_persistence_controls():
    template = (UI / "templates/ml_lab_ui/xml_structure.html").read_text()
    script = (UI / "static/ml_lab_xml_typed_preview.js").read_text()
    blueprint = (UI / "blueprint.py").read_text()
    for name in ("data-xml-typed-save", "data-xml-typed-materialize", "data-typed-save-url", "data-typed-materialize-url"):
        assert name in template
    assert 'window.confirm("Materialize all matching records' in script
    assert 'max_records=5000, max_output_bytes=32 * 1024 * 1024' in blueprint
    assert 'persist_typed_preview(project_root, preview, recipe=recipe)' in blueprint
    assert 'materialize_typed_xml_records(' in blueprint
    assert 'registered_dataset' not in blueprint[blueprint.index('def xml_typed_save_preview'):blueprint.index('@blueprint.get("/data/xml/<token>/record-fields")')]


def test_typed_publication_http_if_flask_available(tmp_path):
    import pytest
    import html
    import re
    pytest.importorskip("flask")
    from ml_lab.ui import create_app

    source = tmp_path / "values.xml"
    source.write_text("<catalog><entry><name>A</name><tag>1</tag><tag>2</tag></entry><entry><name>B</name></entry></catalog>")
    project = tmp_path / "project"
    project.mkdir()
    client = create_app(config={"TESTING": True, "ML_LAB_EXPERIMENT_PROJECT_ROOT": str(project)}).test_client()
    response = client.post("/data", data={"paths": str(source), "preview_rows": "5"})
    assert response.status_code == 200
    match = re.search(rb'href="([^"]*/data/xml/[^\"]+)"[^>]*>Explore XML</a>', response.data)
    assert match is not None
    explorer = html.unescape(match.group(1).decode())
    payload = {
        "record_root_canonical_path": "/catalog/entry",
        "variables": [{"canonical_path": "/catalog/entry/name", "representation": "scalar"}],
        "max_records": 2,
    }
    saved = client.post(explorer + "/typed-save-preview", json=payload)
    assert saved.status_code == 200, saved.get_data(as_text=True)
    saved_result = saved.get_json()
    assert saved_result["registered_dataset"] is False
    assert saved_result["record_count"] == 2
    assert saved.headers["Cache-Control"] == "no-store"
    assert client.post(explorer + "/typed-save-preview", json={**payload, "variables": []}).status_code == 400
    full = client.post(explorer + "/typed-materialize", json={**payload, "max_records": 5000})
    assert full.status_code == 200, full.get_data(as_text=True)
    assert full.get_json()["complete"] is True
    assert full.get_json()["registered_dataset"] is False
    assert client.post(explorer + "/typed-materialize", json=payload).status_code == 400
