from __future__ import annotations

import json
import socket
from pathlib import Path
from urllib.request import urlopen

import pytest

from ml_lab.pah_integration import MLLabRuntimeAdapter, discover_artifacts, module_manifest


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_manifest_exposes_ml_lab_capabilities_without_pah_dependency():
    manifest = module_manifest()
    assert manifest["module_id"] == "ml_lab"
    assert manifest["version"] == "0.20.0"
    assert manifest["collections"] == ["ml_lab"]
    assert "data_provenance" in manifest["capabilities"]
    assert "embedded_ui" in manifest["interfaces"]


def test_artifact_discovery_only_registers_known_ml_lab_schema(tmp_path: Path):
    results = tmp_path / "ml_lab_results"
    results.mkdir()
    derived = results / "derived.csv"
    derived.write_text("x\n1\n", encoding="utf-8")
    manifest = {
        "schema": "ml-lab.derived-dataset@1",
        "derived": {"path": str(derived), "sha256": "abc", "dataset_id": "dataset-1"},
        "source": {"path": str(tmp_path / "source.csv")},
        "provenance": {"event_id": "event-1", "path": str(results / "derived.csv.provenance.json")},
    }
    (results / "derived.csv.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (results / "unrelated.json").write_text(json.dumps({"hello": "world"}), encoding="utf-8")

    artifacts = discover_artifacts({"project_root": tmp_path, "results_root": results})
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact["producer_module"] == "ml_lab"
    assert artifact["kind"] == "derived_dataset"
    assert artifact["schema_id"] == "ml-lab.derived-dataset"
    assert artifact["schema_version"] == "1"
    assert artifact["location"] == str(derived.resolve())
    assert artifact["project_id"] == str(tmp_path.resolve())


def test_runtime_adapter_launches_ml_lab_owned_ui(tmp_path: Path):
    pytest.importorskip("flask")
    port = _free_port()
    context = {
        "project_root": tmp_path,
        "results_root": tmp_path / "ml_lab_results",
        "host": "127.0.0.1",
        "ports": {"ml_lab": port},
    }
    adapter = MLLabRuntimeAdapter()
    try:
        status = adapter.status(context=context)
        assert status["available"] is True
        assert status["running"] is False

        launch = adapter.launch(context=context)
        assert launch["launched"] is True
        assert launch["surface"] == "ml_lab"
        with urlopen(launch["url"], timeout=3.0) as response:
            body = response.read().decode("utf-8")
        assert "ML Lab" in body

        running = adapter.status(context=context)
        assert running["running"] is True
        assert running["url"] == launch["url"]
    finally:
        adapter.shutdown(context=context)
    assert adapter.status(context=context)["running"] is False


def test_host_configured_ui_writes_default_derived_output_under_results_root(tmp_path: Path):
    pytest.importorskip("flask")
    from ml_lab.ui import create_app

    source = tmp_path / "source.csv"
    source.write_text("value\n1\n2\n", encoding="utf-8")
    derived_root = tmp_path / "host-results" / "data" / "derived"
    app = create_app(config={
        "TESTING": True,
        "ML_LAB_PROFILE_OUTPUT_ROOT": str(tmp_path / "host-results" / "data" / "profile_runs"),
        "ML_LAB_DERIVED_OUTPUT_ROOT": str(derived_root),
    })
    with app.test_client() as client:
        response = client.post("/data/transform", data={
            "source": str(source),
            "mode": "apply",
            "recipe_name": "identity",
            "preview_rows": "10",
        })
    assert response.status_code == 200
    assert (derived_root / "source-derived.csv").is_file()
