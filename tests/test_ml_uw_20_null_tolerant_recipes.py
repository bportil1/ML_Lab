"""ML-UW-20: null-tolerant recipe, source alignment and provenance acceptance."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ml_lab.data.xml_collection import inventory_xml_collection
from ml_lab.data.xml_collection_export import materialize_xml_collection, preview_xml_collection_export
from ml_lab.data.xml_collection_recipe import build_xml_collection_recipe
from ml_lab.data.xml_similarity import compare_xml_collection


FIELDS = ["text:/rows/item/name", "text:/rows/item/value", "attribute:/rows/item@code"]


def _sources(folder: Path):
    (folder / "a.xml").write_text(
        '<rows xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        '<item code="1"><name>A</name><value xsi:nil="true"/></item>'
        '<item><name>B</name><value/></item></rows>', encoding="utf-8")
    (folder / "b.xml").write_text(
        '<rows><item><name>C</name></item><item><name>D</name></item></rows>', encoding="utf-8")
    (folder / "c.xml").write_text(
        '<rows><item><other>X</other></item><item><other>Y</other></item></rows>', encoding="utf-8")
    inventory = inventory_xml_collection(folder)
    report = compare_xml_collection(inventory)
    return inventory, report


def _recipe(folder: Path, **overrides):
    inventory, report = _sources(folder)
    return build_xml_collection_recipe(
        inventory, report, ["a.xml", "b.xml", "c.xml"], "/rows/item",
        FIELDS, acknowledge_variants=True, **overrides,
    )


def _cell(lookup: dict, source: str, row: int, field_id: str) -> dict:
    return next(cell for cell in lookup[source][row - 1]["cells"] if cell["field_id"] == field_id)


def test_null_is_default_and_strict_mode_remains_explicit(tmp_path):
    recipe = _recipe(tmp_path)
    assert recipe["structural_recipe_validated"]
    assert recipe["allow_missing_fields"] is True
    assert recipe["status"] == "ready_for_collection_preview"
    assert recipe["selected_sources"][2]["present_field_ids"] == []
    assert recipe["selected_sources"][2]["ready_for_preview"] is True
    contract = recipe["null_alignment"]
    assert contract["schema"] == "ml-lab.xml-null-alignment@1"
    assert contract["policy"] == "missing_as_null"
    fields = {entry["field_id"]: entry for entry in contract["fields"]}
    assert fields["text:/rows/item/value"]["missing_in"] == ["b.xml", "c.xml"]
    assert fields["text:/rows/item/value"]["absence_reason_by_source"] == {
        "b.xml": "path_absent", "c.xml": "path_absent",
    }
    assert fields["attribute:/rows/item@code"]["absence_reason_by_source"] == {
        "b.xml": "attribute_absent", "c.xml": "attribute_absent",
    }
    assert fields["attribute:/rows/item@code"]["missing_in"] == ["b.xml", "c.xml"]
    assert [field["column_key"] for field in contract["fields"]] == FIELDS
    json.loads(json.dumps(recipe))
    strict = _recipe(tmp_path, allow_missing_fields=False)
    assert strict["status"] == "blocked"
    assert any("explicitly allow missing" in text for text in strict["blocking_reasons"])
    assert strict["null_alignment"]["policy"] == "reject_missing"


def test_record_null_observations_and_rectangular_main_rows(tmp_path):
    recipe = _recipe(tmp_path)
    preview = preview_xml_collection_export(recipe)
    assert preview["no_files_written"] and preview["null_alignment"] == recipe["null_alignment"]
    assert preview["columns"] == ["__source_file", "__source_row", "name", "value", "@code"]
    assert len(preview["rows"]) == 6
    assert all(set(row) == set(preview["columns"]) for row in preview["rows"])
    a1, a2, b1, b2, c1, c2 = preview["rows"]
    assert a1["value"] is None  # explicit xsi:nil
    assert a2["value"] == ""  # element exists, but is empty
    assert b1["value"] is None and b2["value"] is None  # path absent
    assert c1["name"] is None and c1["value"] is None and c1["@code"] is None
    assert c2["name"] is None and c2["value"] is None
    states = preview["null_observations_by_source"]
    assert _cell(states, "a.xml", 1, "text:/rows/item/value")["status"] == "explicit_xsi_nil"
    assert _cell(states, "a.xml", 2, "text:/rows/item/value")["status"] == "empty_element"
    assert _cell(states, "b.xml", 1, "text:/rows/item/value")["status"] == "path_absent"
    assert _cell(states, "a.xml", 2, "attribute:/rows/item@code")["status"] == "attribute_absent"
    assert _cell(states, "c.xml", 1, "text:/rows/item/name")["status"] == "path_absent"
    assert all("null_reason" in obs for source in states.values() for record in source
               for cell in record["cells"] for obs in cell["observations"])
    assert not (tmp_path / "combined.csv").exists()


def test_materialization_preserves_source_rows_and_contract(tmp_path):
    recipe = _recipe(tmp_path)
    destination = tmp_path / "combined.tsv"
    manifest = materialize_xml_collection(recipe, destination, confirm=True)
    assert manifest["main_dataset"]["row_count"] == 6
    with destination.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 6
    assert rows[-1]["__source_file"] == "c.xml"
    assert rows[-1]["name"] == rows[-1]["value"] == ""
    # Until ML-UW-21, CSV/TSV cannot distinguish an empty string from None:
    # the canonical recipe is the authoritative null provenance contract.
    assert manifest["recipe"]["null_alignment"]["policy"] == "missing_as_null"
    assert set(manifest["recipe"]["null_alignment"]["fields"][0]) >= {"field_id", "canonical_path", "missing_in"}


def test_revalidation_rejects_changes_in_a_null_only_source(tmp_path):
    recipe = _recipe(tmp_path)
    (tmp_path / "c.xml").write_text("<rows><item><name>replaced</name></item></rows>")
    with pytest.raises(ValueError, match="changed"):
        preview_xml_collection_export(recipe)


def test_collected_empty_leaf_selection_does_not_change_legacy_single_file(tmp_path):
    recipe = _recipe(tmp_path)
    from ml_lab.data.xml_ingest import inspect_xml_structure
    from ml_lab.data.xml_selection import build_xml_record_selection
    artifact = inspect_xml_structure(tmp_path / "a.xml")
    legacy = {field.field_id for field in build_xml_record_selection(artifact, "/rows/item").fields}
    collection = {field.field_id for field in build_xml_record_selection(
        artifact, "/rows/item", include_empty_leaf_fields=True).fields}
    assert "text:/rows/item/value" not in legacy
    assert "text:/rows/item/value" in collection
    assert "text:/rows/item/value" in recipe["selected_field_ids"]
