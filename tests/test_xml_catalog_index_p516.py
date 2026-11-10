"""P5-16 independent catalog smoke and regression tests (no Flask)."""
import json
from pathlib import Path

import pytest

from ml_lab.data.xml_catalog_index import (
    XmlCatalogError, list_xml_catalog, rebuild_xml_catalog,
)
from ml_lab.data.xml_typed_artifacts import _encoded


def test_empty_and_project_isolation(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir(); second.mkdir()
    assert list_xml_catalog(first)["total"] == 0
    assert rebuild_xml_catalog(first)["indexed"] == 0
    assert (first / "analysis/ml_lab/xml_catalog.sqlite3").exists()
    assert not (second / "analysis").exists()
    assert list_xml_catalog(second)["items"] == []


def test_tampering_skipped_and_pagination(tmp_path):
    from ml_lab.data.xml_typed_artifacts import persist_typed_preview
    root = tmp_path / "project"; root.mkdir()
    sha = "a" * 64
    recipe = {"schema": "ml-lab.xml-typed-recipe@1", "version": 1,
              "record_root_canonical_path": "/root/item", "encoding": "none",
              "legacy_extraction_modified": False}
    base = {"schema": "ml-lab.xml-typed-record-preview@1",
            "registered_dataset": False, "encoding": "none", "source_sha256": sha,
            "source_fingerprint": "fingerprint", "record_root_canonical_path": "/root/item",
            "records": [], "preview_record_count": 0, "truncated": True}
    from ml_lab.data.xml_typed_records import PREVIEW_SCHEMA
    base["schema"] = PREVIEW_SCHEMA
    ids = []
    for i in range(2):
        preview = {**base, "source_fingerprint": f"fingerprint-{i}"}
        ids.append(persist_typed_preview(root, preview, recipe=recipe)["artifact_id"])
    assert rebuild_xml_catalog(root)["indexed"] == 2
    page = list_xml_catalog(root, limit=1, offset=1)
    assert page["total"] == 2 and len(page["items"]) == 1
    assert list_xml_catalog(root, source_sha256="b" * 64)["total"] == 0
    artifact_file = root / "artifacts/xml_typed_previews" / (ids[0] + ".json")
    artifact_file.write_text("{}", encoding="utf-8")
    result = rebuild_xml_catalog(root)
    assert result["indexed"] == 1 and result["invalid_skipped"] == 1
    assert list_xml_catalog(root)["total"] == 1


def test_input_validation_and_future_schema(tmp_path):
    root = tmp_path / "p"; root.mkdir()
    with pytest.raises(XmlCatalogError):
        list_xml_catalog(root, limit=0)
    with pytest.raises(XmlCatalogError):
        list_xml_catalog(root, source_sha256="notahash")
    rebuild_xml_catalog(root)
    import sqlite3
    with sqlite3.connect(root / "analysis/ml_lab/xml_catalog.sqlite3") as db:
        db.execute("PRAGMA user_version=99")
    with pytest.raises(XmlCatalogError):
        list_xml_catalog(root)
