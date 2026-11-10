"""Offline P5-16 recipe provenance and catalog acceptance tests."""
import copy
from pathlib import Path

import pytest

from ml_lab.data.xml_collection import inventory_xml_collection
from ml_lab.data.xml_catalog_index import rebuild_xml_catalog, list_xml_catalog
from ml_lab.data.xml_source_catalog import save_xml_inventory_snapshot, rebuild_xml_source_catalog, list_xml_sources
from ml_lab.data.xml_recipe_catalog import (
    save_xml_recipe_snapshot, read_xml_recipe_snapshot,
    rebuild_xml_recipe_catalog, list_xml_recipes,
)


def fixture(tmp_path):
    project = tmp_path / "project"; project.mkdir()
    source = tmp_path / "input"; source.mkdir()
    (source / "a.xml").write_text("<root><entry><x>3</x></entry></root>")
    inventory = inventory_xml_collection(source)
    entry = inventory.files[0]
    snapshot = save_xml_inventory_snapshot(project, inventory)
    recipe = {"schema": "ml-lab.xml-collection-recipe@1", "status": "ready_for_collection_preview",
              "structural_recipe_validated": True, "merge_authorized": False, "csv_generated": False,
              "blocking_reasons": [], "root_path": inventory.root_path,
              "record_root_canonical_path": "/root/entry", "selected_field_ids": ["x"],
              "selected_sources": [{"relative_path": entry.relative_path,
                                    "source_fingerprint": "sha256:" + entry.sha256}]}
    return project, snapshot["snapshot_id"], entry.sha256, recipe


def test_snapshot_and_paged_provenance_lookup(tmp_path):
    project, inventory_id, source_sha, recipe = fixture(tmp_path)
    save = save_xml_recipe_snapshot(project, recipe, inventory_snapshot_id=inventory_id)
    assert save_xml_recipe_snapshot(project, recipe, inventory_snapshot_id=inventory_id) == save
    doc = read_xml_recipe_snapshot(project, save["snapshot_id"])
    assert doc["recipe"]["selected_field_ids"] == ["x"]
    assert list_xml_recipes(project)["total"] == 0
    assert rebuild_xml_recipe_catalog(project)["source_links"] == 1
    assert list_xml_recipes(project, source_sha256=source_sha)["total"] == 1
    assert list_xml_recipes(project, inventory_snapshot_id=inventory_id, limit=1)["total"] == 1
    assert list_xml_recipes(project, source_sha256="0" * 64)["total"] == 0
    assert rebuild_xml_source_catalog(project)["sources"] == 1
    assert rebuild_xml_catalog(project)["indexed"] == 0
    assert list_xml_sources(project)["total"] == 1
    assert list_xml_catalog(project)["total"] == 0
    assert list_xml_recipes(project)["total"] == 1


def test_reject_mismatched_inventory_and_bad_recipe(tmp_path):
    project, inventory_id, source_sha, recipe = fixture(tmp_path)
    changed = copy.deepcopy(recipe)
    changed["selected_sources"][0]["source_fingerprint"] = "sha256:" + "f" * 64
    with pytest.raises(ValueError, match="evidence"):
        save_xml_recipe_snapshot(project, changed, inventory_snapshot_id=inventory_id)
    changed = copy.deepcopy(recipe)
    changed["status"] = "blocked"
    with pytest.raises(ValueError, match="validated"):
        save_xml_recipe_snapshot(project, changed)
    changed = copy.deepcopy(recipe)
    changed["selected_sources"][0]["relative_path"] = "../escape.xml"
    with pytest.raises(ValueError, match="path"):
        save_xml_recipe_snapshot(project, changed)
    with pytest.raises(ValueError):
        list_xml_recipes(project, offset=-1)


def test_corruption_excluded_and_project_isolation(tmp_path):
    project, inventory_id, _, recipe = fixture(tmp_path)
    saved = save_xml_recipe_snapshot(project, recipe, inventory_snapshot_id=inventory_id)
    other = tmp_path / "other"; other.mkdir()
    assert rebuild_xml_recipe_catalog(other)["recipes"] == 0
    assert list_xml_recipes(other)["total"] == 0
    assert rebuild_xml_recipe_catalog(project)["recipes"] == 1
    path = project / saved["relative_path"]
    path.write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        read_xml_recipe_snapshot(project, saved["snapshot_id"])
    assert rebuild_xml_recipe_catalog(project)["invalid_skipped"] == 1
    assert list_xml_recipes(project)["total"] == 0
