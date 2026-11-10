"""P5-16 explicit index-refresh integration, runnable without Flask."""
import pytest
from ml_lab.data.xml_catalog_refresh import refresh_xml_catalog
from ml_lab.data.xml_catalog_browser import browse_xml_catalog
from ml_lab.data.xml_source_catalog import save_xml_inventory_snapshot
from ml_lab.data.xml_collection import inventory_xml_collection


def test_no_implicit_rebuild_and_explicit_refresh(tmp_path):
    project = tmp_path / "project"; project.mkdir()
    inputs = tmp_path / "xml"; inputs.mkdir()
    (inputs / "file.xml").write_text("<r><n>1</n></r>")
    save_xml_inventory_snapshot(project, inventory_xml_collection(inputs))
    assert browse_xml_catalog(project, section="sources")["total"] == 0
    assert not (project / "analysis/ml_lab/xml_catalog.sqlite3").exists()
    result = refresh_xml_catalog(project)
    assert result["complete"] and not result["partial"]
    assert result["sections"]["sources"]["counts"]["sources"] == 1
    assert browse_xml_catalog(project, section="sources")["total"] == 1
    assert (inputs / "file.xml").exists()


def test_reject_scope_and_project_isolation(tmp_path):
    one = tmp_path / "one"; one.mkdir()
    two = tmp_path / "two"; two.mkdir()
    with pytest.raises(ValueError):
        refresh_xml_catalog(one, scope="invalid")
    refresh_xml_catalog(one, scope="artifacts")
    assert not (two / "analysis/ml_lab/xml_catalog.sqlite3").exists()


def test_invalid_snapshot_count_visible_and_one_section_refresh(tmp_path):
    project = tmp_path / "project"; project.mkdir()
    folder = project / "artifacts/xml_collection_inventories"
    folder.mkdir(parents=True)
    (folder / ("a" * 64 + ".json")).write_text("tampered")
    report = refresh_xml_catalog(project, scope="sources")
    assert report["complete"]
    assert report["sections"]["sources"]["counts"]["invalid_skipped"] == 1
    assert "recipes" not in report["sections"]


def test_partial_failure_explicit_and_no_false_success(tmp_path, monkeypatch):
    from ml_lab.data import xml_catalog_refresh as module
    project = tmp_path / "project"; project.mkdir()
    def broken(_):
        raise ValueError("fixture failure")
    monkeypatch.setattr(module, "rebuild_xml_recipe_catalog", broken)
    report = module.refresh_xml_catalog(project)
    assert not report["complete"] and report["partial"]
    assert report["sections"]["recipes"]["status"] == "failed"
    assert report["sections"]["sources"]["status"] == "rebuilt"
    assert report["sections"]["artifacts"]["status"] == "rebuilt"
