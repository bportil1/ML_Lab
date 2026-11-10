"""P5-16 XML source collection index acceptance, no Flask needed."""
import json
from pathlib import Path

import pytest

from ml_lab.data.xml_collection import inventory_xml_collection
from ml_lab.data.xml_catalog_index import rebuild_xml_catalog, list_xml_catalog
from ml_lab.data.xml_source_catalog import (
    save_xml_inventory_snapshot, read_xml_inventory_snapshot,
    rebuild_xml_source_catalog, list_xml_sources,
)


def _input(tmp_path: Path):
    folder = tmp_path / "inputs"
    folder.mkdir()
    (folder / "a.xml").write_text("<root><item>one</item></root>", encoding="utf-8")
    (folder / "bad.xml").write_text("<root>", encoding="utf-8")
    return inventory_xml_collection(folder)


def test_source_inventory_snapshot_and_rebuild(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    inventory = _input(tmp_path)
    saved = save_xml_inventory_snapshot(project, inventory)
    assert save_xml_inventory_snapshot(project, inventory) == saved
    assert read_xml_inventory_snapshot(project, saved["snapshot_id"])["inventory"]["summary"]["file_count"] == 2
    result = rebuild_xml_source_catalog(project)
    assert result["collections"] == 1 and result["sources"] == 2
    page = list_xml_sources(project, limit=1)
    assert page["total"] == 2 and len(page["items"]) == 1
    assert list_xml_sources(project, snapshot_id=saved["snapshot_id"])["total"] == 2
    assert list_xml_sources(project, sha256=inventory.files[0].sha256)["total"] == 1
    assert list_xml_sources(tmp_path / "inputs")["total"] == 0
    assert list_xml_catalog(project)["total"] == 0
    assert rebuild_xml_catalog(project)["indexed"] == 0
    assert list_xml_sources(project)["total"] == 2  # artifacts rebuild does not discard source rows


def test_mutation_and_index_rebuild_are_non_authoritative(tmp_path):
    project = tmp_path / "project"; project.mkdir()
    inv = _input(tmp_path)
    saved = save_xml_inventory_snapshot(project, inv)
    rebuild_xml_source_catalog(project)
    path = project / saved["relative_path"]
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        read_xml_inventory_snapshot(project, saved["snapshot_id"])
    assert rebuild_xml_source_catalog(project)["invalid_skipped"] == 1
    assert list_xml_sources(project)["total"] == 0
    assert (tmp_path / "inputs/a.xml").exists()


def test_no_implicit_indexing_and_invalid_input(tmp_path):
    project = tmp_path / "p"; project.mkdir()
    inv = _input(tmp_path)
    save_xml_inventory_snapshot(project, inv)
    assert list_xml_sources(project)["total"] == 0
    with pytest.raises(ValueError):
        list_xml_sources(project, limit=0)
    with pytest.raises(ValueError):
        list_xml_sources(project, snapshot_id="x")
    assert rebuild_xml_source_catalog(project)["sources"] == 2
