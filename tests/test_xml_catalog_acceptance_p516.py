"""P5-16 offline acceptance: index authority, isolation, recovery and completeness.

These checks exercise the public catalog APIs without Flask or source mutation.
"""
from __future__ import annotations

import sqlite3

import pytest

from ml_lab.data.xml_catalog_browser import browse_xml_catalog
from ml_lab.data.xml_catalog_refresh import refresh_xml_catalog
from ml_lab.data.xml_collection import inventory_xml_collection
from ml_lab.data.xml_recipe_catalog import save_xml_recipe_snapshot
from ml_lab.data.xml_source_catalog import save_xml_inventory_snapshot
from ml_lab.data.xml_typed_artifacts import persist_typed_preview
from ml_lab.data.xml_typed_records import PREVIEW_SCHEMA


def _project_fixture(tmp_path, name="project"):
    project = tmp_path / name
    project.mkdir()
    inputs = tmp_path / (name + "_inputs")
    inputs.mkdir()
    source = inputs / "file.xml"
    source.write_text("<root><entry><n>42</n></entry></root>", encoding="utf-8")
    inventory = inventory_xml_collection(inputs)
    snapshot = save_xml_inventory_snapshot(project, inventory)
    item = inventory.files[0]
    recipe = {
        "schema": "ml-lab.xml-collection-recipe@1",
        "status": "ready_for_collection_preview",
        "structural_recipe_validated": True, "merge_authorized": False,
        "csv_generated": False, "blocking_reasons": [],
        "root_path": inventory.root_path,
        "record_root_canonical_path": "/root/entry",
        "selected_field_ids": ["n"],
        "selected_sources": [{"relative_path": item.relative_path,
                              "source_fingerprint": "sha256:" + item.sha256}],
    }
    recipe_snapshot = save_xml_recipe_snapshot(
        project, recipe, inventory_snapshot_id=snapshot["snapshot_id"])
    typed_recipe = {"schema": "ml-lab.xml-typed-recipe@1", "version": 1,
                    "record_root_canonical_path": "/root/entry", "encoding": "none",
                    "legacy_extraction_modified": False}
    typed_preview = {"schema": PREVIEW_SCHEMA, "registered_dataset": False,
                     "encoding": "none", "source_sha256": item.sha256,
                     "source_fingerprint": "sha256:" + item.sha256,
                     "record_root_canonical_path": "/root/entry", "records": [],
                     "preview_record_count": 0, "truncated": True}
    artifact = persist_typed_preview(project, typed_preview, recipe=typed_recipe)
    return project, source, snapshot, recipe_snapshot, artifact, item.sha256


def test_rebuild_and_cross_section_provenance(tmp_path):
    project, source, inventory, recipe, artifact, sha = _project_fixture(tmp_path)
    assert all(browse_xml_catalog(project, section=s)["total"] == 0
               for s in ("collections", "sources", "recipes", "artifacts"))
    report = refresh_xml_catalog(project)
    assert report["complete"] and not report["partial"]
    assert not report["failed_sections"]
    assert report["sections"]["sources"]["counts"]["sources"] == 1
    assert report["sections"]["recipes"]["counts"]["source_links"] == 1
    assert report["sections"]["artifacts"]["counts"]["indexed"] == 1
    assert browse_xml_catalog(project, section="collections")["total"] == 1
    assert browse_xml_catalog(project, section="sources", source_sha256=sha)["total"] == 1
    assert browse_xml_catalog(project, section="recipes", source_sha256=sha)["total"] == 1
    assert browse_xml_catalog(project, section="recipes", inventory_snapshot_id=inventory["snapshot_id"])["total"] == 1
    assert browse_xml_catalog(project, section="artifacts", source_sha256=sha,
                              artifact_kind="preview")["total"] == 1
    assert source.read_text(encoding="utf-8") == "<root><entry><n>42</n></entry></root>"
    assert not (project / "datasets").exists()
    assert recipe["registered_dataset"] is False
    assert artifact["registered_dataset"] is False


def test_catalog_is_observation_not_source_reverification(tmp_path):
    project, source, *_ = _project_fixture(tmp_path)
    assert refresh_xml_catalog(project)["complete"]
    before = browse_xml_catalog(project, section="sources")["items"]
    source.write_text("<root><entry><n>changed</n></entry></root>", encoding="utf-8")
    assert refresh_xml_catalog(project)["complete"]
    after = browse_xml_catalog(project, section="sources")["items"]
    assert before == after  # no XML re-scan or false 'current' evidence
    assert browse_xml_catalog(project, section="sources")["stale_until_rebuilt"]


def test_rebuild_skips_tampered_artifact_without_erasing_other_sections(tmp_path):
    project, source, inventory, recipe, artifact, _ = _project_fixture(tmp_path)
    assert refresh_xml_catalog(project)["complete"]
    (project / artifact["relative_path"]).write_text("{}", encoding="utf-8")
    result = refresh_xml_catalog(project, scope="artifacts")
    assert result["complete"]
    assert result["sections"]["artifacts"]["counts"]["invalid_skipped"] == 1
    assert browse_xml_catalog(project, section="artifacts")["total"] == 0
    for section in ("sources", "collections", "recipes"):
        assert browse_xml_catalog(project, section=section)["total"] == 1
    assert (project / recipe["relative_path"]).is_file()
    assert source.is_file()


def test_database_failure_rolls_back_previous_source_index(tmp_path):
    project, *_ = _project_fixture(tmp_path)
    assert refresh_xml_catalog(project)["complete"]
    database = project / "analysis/ml_lab/xml_catalog.sqlite3"
    prior_sources = browse_xml_catalog(project, section="sources")["items"]
    with sqlite3.connect(database) as db:
        db.execute("""CREATE TRIGGER prevent_source_update BEFORE INSERT ON xml_sources
                      BEGIN SELECT RAISE(ABORT, 'injected catalog failure'); END""")
    report = refresh_xml_catalog(project, scope="sources")
    assert not report["complete"] and not report["partial"]
    assert report["failed_sections"] == ["sources"]
    assert "injected catalog failure" in report["sections"]["sources"]["error"]
    assert browse_xml_catalog(project, section="sources")["items"] == prior_sources
    with sqlite3.connect(database) as db:
        db.execute("DROP TRIGGER prevent_source_update")
    assert refresh_xml_catalog(project, scope="sources")["complete"]
    assert browse_xml_catalog(project, section="sources")["items"] == prior_sources


def test_read_only_pagination_and_independent_projects(tmp_path):
    one, *_ = _project_fixture(tmp_path, "one")
    two, *_ = _project_fixture(tmp_path, "two")
    refresh_xml_catalog(one)
    for section in ("collections", "sources", "recipes", "artifacts"):
        assert browse_xml_catalog(one, section=section, limit=1, offset=0)["total"] == 1
        assert browse_xml_catalog(one, section=section, limit=1, offset=1)["items"] == []
        assert browse_xml_catalog(two, section=section)["total"] == 0
    assert not (two / "analysis").exists()
    assert refresh_xml_catalog(two)["complete"]
    assert browse_xml_catalog(two, section="sources")["total"] == 1


@pytest.mark.parametrize("scope", ["sources", "recipes", "artifacts"])
def test_single_section_refresh_does_not_erase_other_sections(tmp_path, scope):
    project, *_ = _project_fixture(tmp_path)
    refresh_xml_catalog(project)
    result = refresh_xml_catalog(project, scope=scope)
    assert result["complete"] and list(result["sections"]) == [scope]
    assert all(browse_xml_catalog(project, section=section)["total"] == 1
               for section in ("collections", "sources", "recipes", "artifacts"))
