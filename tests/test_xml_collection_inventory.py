"""XML-C1 inventory tests: local discovery, isolation, limits and CLI/UI entrypoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml_lab import data
from ml_lab.cli import main


def _xml(path: Path, *, namespace: str = "urn:a") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'<rows xmlns="{namespace}"><record id="1"><value>42</value></record>'
                    '<record id="2"><value>43</value></record></rows>', encoding="utf-8")


def test_nested_collection_inventory_and_record_candidates(tmp_path: Path) -> None:
    _xml(tmp_path / "a.xml")
    _xml(tmp_path / "nested" / "b.XML", namespace="urn:b")
    (tmp_path / "readme.txt").write_text("ignored", encoding="utf-8")
    result = data.inventory_xml_collection(tmp_path)
    payload = result.to_record()
    assert payload["schema"] == "ml-lab.xml-collection-inventory@1"
    assert [item["relative_path"] for item in payload["files"]] == ["a.xml", "nested/b.XML"]
    assert payload["summary"]["parsed_file_count"] == 2
    assert payload["summary"]["similarity_status"] == "not_computed"
    assert payload["summary"]["consolidation_status"] == "not_available"
    assert payload["files"][0]["root"]["local_name"] == "rows"
    assert payload["files"][1]["root"]["namespace_uri"] == "urn:b"
    assert payload["files"][0]["sha256"]
    assert payload["files"][0]["record_candidates"]


def test_hidden_files_symlinks_globs_and_nonrecursive(tmp_path: Path) -> None:
    _xml(tmp_path / "a.xml")
    _xml(tmp_path / "b.xml")
    _xml(tmp_path / ".secret.xml")
    _xml(tmp_path / "nested" / "c.xml")
    _xml(tmp_path / ".hidden" / "d.xml")
    (tmp_path / "link.xml").symlink_to(tmp_path / "a.xml")
    (tmp_path / "link_dir").symlink_to(tmp_path / "nested", target_is_directory=True)
    result = data.inventory_xml_collection(tmp_path, recursive=False, exclude_globs=["b.*"])
    assert [item.relative_path for item in result.files] == ["a.xml"]
    result = data.inventory_xml_collection(tmp_path, include_hidden=True, include_globs=["nested/*"])
    assert [item.relative_path for item in result.files] == ["nested/c.xml"]
    result = data.inventory_xml_collection(tmp_path, include_hidden=True)
    assert len(result.files) == 5  # .secret.xml, .hidden/d.xml, a.xml, b.xml, nested/c.xml


def test_parse_failure_is_per_source_not_silent(tmp_path: Path) -> None:
    _xml(tmp_path / "good.xml")
    (tmp_path / "bad.xml").write_text("<rows>", encoding="utf-8")
    (tmp_path / "dtd.xml").write_text('<!DOCTYPE x [<!ENTITY x "bad">]><x>&x;</x>', encoding="utf-8")
    result = data.inventory_xml_collection(tmp_path)
    status = {item.relative_path: item for item in result.files}
    assert status["good.xml"].status == "parsed"
    assert status["bad.xml"].status == "failed"
    assert status["dtd.xml"].status == "failed"
    assert "not allowed" in status["dtd.xml"].error
    assert status["bad.xml"].sha256 == ""


def test_file_and_collection_limits_fail_closed(tmp_path: Path) -> None:
    _xml(tmp_path / "first.xml")
    _xml(tmp_path / "second.xml")
    with pytest.raises(ValueError, match="file limit"):
        data.inventory_xml_collection(tmp_path, max_files=1)
    with pytest.raises(ValueError, match="total size limit"):
        data.inventory_xml_collection(tmp_path, max_total_bytes=1)
    with pytest.raises(ValueError, match="entry limit"):
        data.inventory_xml_collection(tmp_path, max_entries=1)
    result = data.inventory_xml_collection(tmp_path, max_file_bytes=1)
    assert all(item.status == "failed" for item in result.files)
    assert all("size limit" in item.error for item in result.files)


def test_missing_or_symlink_root_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="directory"):
        data.inventory_xml_collection(tmp_path / "missing")
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="directory"):
        data.inventory_xml_collection(alias)
    with pytest.raises(ValueError, match="positive"):
        data.inventory_xml_collection(tmp_path, max_files=0)


def test_xml_collection_cli_writes_explicit_inventory(tmp_path: Path, capsys) -> None:
    _xml(tmp_path / "a.xml")
    output = tmp_path / "results" / "collection.json"
    assert main(["data", "xml-collection-inventory", str(tmp_path), "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["summary"]["file_count"] == 1
    assert "Similarity and consolidation are not performed" in capsys.readouterr().out


def test_xml_collection_ui_is_separate_from_single_file_processing(tmp_path: Path) -> None:
    pytest.importorskip("flask")
    from ml_lab.ui import create_app

    _xml(tmp_path / "one.xml")
    client = create_app(config={"TESTING": True}).test_client()
    data_page = client.get("/data")
    assert b"XML collection analysis" in data_page.data
    assert b"Analyze an XML directory" in data_page.data
    response = client.post("/data/xml/collection", data={"directory": str(tmp_path), "recursive": "1"})
    assert response.status_code == 200
    assert b"one.xml" in response.data
    assert b"Similarity grouping and consolidated CSV export" in response.data
    assert client.get("/data/xml/collection").status_code == 200
