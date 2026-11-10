"""Read-only XML catalog workspace query tests (no Flask required)."""
import pytest
from ml_lab.data.xml_catalog_browser import browse_xml_catalog
from ml_lab.data.xml_source_catalog import save_xml_inventory_snapshot, rebuild_xml_source_catalog
from ml_lab.data.xml_collection import inventory_xml_collection


def test_empty_catalog_is_read_only(tmp_path):
    root = tmp_path / "project"; root.mkdir()
    for section in ("collections", "sources", "recipes", "artifacts"):
        page = browse_xml_catalog(root, section=section)
        assert page["total"] == 0 and page["items"] == []
    assert not (root / "analysis/ml_lab/xml_catalog.sqlite3").exists()


def test_bounded_collection_source_pagination_and_project_isolation(tmp_path):
    root = tmp_path / "project"; root.mkdir()
    sources = tmp_path / "xml"; sources.mkdir()
    for i in range(4):
        (sources / f"{i}.xml").write_text(f"<root><value>{i}</value></root>")
    inventory = inventory_xml_collection(sources)
    snap = save_xml_inventory_snapshot(root, inventory)
    rebuild_xml_source_catalog(root)
    collection = browse_xml_catalog(root, section="collections")
    assert collection["total"] == 1
    first = browse_xml_catalog(root, section="sources", limit=2)
    second = browse_xml_catalog(root, section="sources", limit=2, offset=2)
    assert first["total"] == 4 and len(first["items"]) == len(second["items"]) == 2
    assert first["items"][0]["relative_path"] != second["items"][0]["relative_path"]
    assert browse_xml_catalog(root, section="sources", inventory_snapshot_id=snap["snapshot_id"])["total"] == 4
    other = tmp_path / "other"; other.mkdir()
    assert browse_xml_catalog(other, section="sources")["total"] == 0


@pytest.mark.parametrize("params", [
    {"section": "other"}, {"limit": 0}, {"limit": 101},
    {"offset": -1}, {"limit": True},
    {"section": "collections", "source_sha256": "a" * 64},
    {"section": "sources", "source_sha256": "unsafe"},
    {"section": "artifacts", "artifact_kind": "bad"},
])
def test_reject_invalid_requests(tmp_path, params):
    with pytest.raises(ValueError):
        browse_xml_catalog(tmp_path, **params)
