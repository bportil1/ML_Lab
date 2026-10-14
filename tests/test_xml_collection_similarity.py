"""XML-C2: structural similarity, provenance guards, bounded grouping, CLI/UI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml_lab import data
from ml_lab.cli import main


def _write(path: Path, *, uri: str = "urn:inventory", prefix: str = "", extra: str = "", n: int = 2) -> None:
    ns = f'xmlns:{prefix}="{uri}"' if prefix else f'xmlns="{uri}"'
    name = lambda tag: f'{prefix}:{tag}' if prefix else tag
    records = "".join(
        f'<{name("record")} id="{i}"><{name("value")}>{i}</{name("value")}>{extra}</{name("record")}>'
        for i in range(n)
    )
    path.write_text(f'<{name("rows")} {ns}>{records}</{name("rows")}>', encoding="utf-8")


def _pairs(result) -> dict[tuple[str, str], dict]:
    return {(p["left"], p["right"]): p for p in result.to_record()["pairs"]}


def test_identical_shape_ignores_namespace_prefix_and_values(tmp_path: Path) -> None:
    _write(tmp_path / "a.xml", prefix="a")
    _write(tmp_path / "b.xml", prefix="b")
    report = data.compare_xml_collection(data.inventory_xml_collection(tmp_path)).to_record()
    pair = report["pairs"][0]
    assert pair["classification"] == "exact"
    assert pair["similarity_score"] == 1.0
    assert report["groups"][0]["members"] == ["a.xml", "b.xml"]
    assert report["groups"][0]["merge_authorized"] is False
    assert report["summary"]["consolidation_status"] == "not_available"
    assert report["sources"][0]["structure_fingerprint"] == report["sources"][1]["structure_fingerprint"]


def test_optional_extra_field_is_advisory_compatible_variant(tmp_path: Path) -> None:
    _write(tmp_path / "a.xml")
    _write(tmp_path / "b.xml", extra="<note xmlns='urn:inventory'>optional</note>")
    pair = data.compare_xml_collection(data.inventory_xml_collection(tmp_path)).to_record()["pairs"][0]
    assert pair["classification"] == "compatible_variant_candidate"
    assert pair["differences"]["right_only_path_count"] == 1
    assert pair["shared_record_root_candidates"]
    assert pair["merge_authorized"] is False


def test_namespace_and_record_layout_conflicts_are_not_merged(tmp_path: Path) -> None:
    _write(tmp_path / "a.xml", uri="urn:original")
    _write(tmp_path / "b.xml", uri="urn:different")
    pair = data.compare_xml_collection(data.inventory_xml_collection(tmp_path)).to_record()["pairs"][0]
    assert pair["classification"] == "incompatible"
    assert pair["reason"] == "different_expanded_root_qname"


def test_record_cardinality_requires_review(tmp_path: Path) -> None:
    _write(tmp_path / "a.xml", n=2)
    _write(tmp_path / "b.xml", n=1)
    pair = data.compare_xml_collection(data.inventory_xml_collection(tmp_path)).to_record()["pairs"][0]
    assert pair["classification"] == "review_required"
    assert pair["reason"] == "observed_cardinality_differs"


def test_failed_and_stale_sources_excluded_with_reasons(tmp_path: Path) -> None:
    _write(tmp_path / "a.xml")
    _write(tmp_path / "b.xml")
    (tmp_path / "broken.xml").write_text("<broken>", encoding="utf-8")
    inventory = data.inventory_xml_collection(tmp_path)
    (tmp_path / "b.xml").write_text("<changed />", encoding="utf-8")
    report = data.compare_xml_collection(inventory).to_record()
    assert report["summary"]["compared_source_count"] == 1
    assert report["summary"]["excluded_source_count"] == 2
    reasons = {s["relative_path"]: s.get("reason", "") for s in report["sources"]}
    assert "fingerprint" in reasons["b.xml"] or "metadata" in reasons["b.xml"]
    assert "ParseError" in reasons["broken.xml"]
    assert report["pairs"] == []


def test_comparison_limits_and_nonmutating_output(tmp_path: Path) -> None:
    for name in ("a.xml", "b.xml", "c.xml"):
        _write(tmp_path / name)
    inventory = data.inventory_xml_collection(tmp_path)
    with pytest.raises(ValueError, match="source limit"):
        data.compare_xml_collection(inventory, max_sources=2)
    with pytest.raises(ValueError, match="hard comparison safety"):
        data.compare_xml_collection(inventory, max_sources=1000)
    with pytest.raises(ValueError, match="minimum_similarity"):
        data.compare_xml_collection(inventory, minimum_similarity=1.2)
    before = (tmp_path / "a.xml").read_bytes()
    report = data.compare_xml_collection(inventory)
    destination = data.save_xml_collection_comparison(report, tmp_path / "report.json")
    assert json.loads(destination.read_text())["schema"] == "ml-lab.xml-collection-similarity@1"
    assert (tmp_path / "a.xml").read_bytes() == before


def test_cli_comparison_writes_report(tmp_path: Path, capsys) -> None:
    _write(tmp_path / "a.xml")
    _write(tmp_path / "b.xml")
    output = tmp_path / "report.json"
    assert main(["data", "xml-collection-compare", str(tmp_path), "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["pair_count"] == 1
    assert report["pairs"][0]["classification"] == "exact"
    assert "no files were merged" in capsys.readouterr().out


def test_ui_opt_in_similarity_versus_inventory_only(tmp_path: Path) -> None:
    pytest.importorskip("flask")
    from ml_lab.ui import create_app
    _write(tmp_path / "a.xml")
    _write(tmp_path / "b.xml")
    client = create_app(config={"TESTING": True}).test_client()
    base = client.post("/data/xml/collection", data={"directory": str(tmp_path), "recursive": "1"})
    assert base.status_code == 200
    assert b"Machine-readable XML-C2 similarity report" not in base.data
    response = client.post("/data/xml/collection", data={"directory": str(tmp_path), "recursive": "1", "compare": "1"})
    assert response.status_code == 200
    assert b"Machine-readable XML-C2 similarity report" in response.data
    assert b"Candidate structure groups" in response.data
    assert b"exact" in response.data
