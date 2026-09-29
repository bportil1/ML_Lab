from __future__ import annotations

import json
from pathlib import Path

from ml_lab import data


def test_csv_inventory_detects_shape_header_delimiter_and_exported_index(tmp_path: Path):
    source = tmp_path / "sample.csv"
    source.write_text(
        "Unnamed: 0,feature_a,feature_b\n"
        "0,1.5,alpha\n"
        "1,2.5,beta\n"
        "2,3.5,gamma\n",
        encoding="utf-8",
    )

    inventory = data.inspect_paths([tmp_path])
    assert inventory.discovered_file_count == 1
    record = inventory.files[0]
    assert record.parse_status == "parsed"
    assert record.delimiter == ","
    assert record.has_header is True
    assert record.row_count == 3
    assert record.column_count == 3
    assert record.columns == ("Unnamed: 0", "feature_a", "feature_b")
    assert record.likely_exported_index_columns == ("Unnamed: 0",)
    assert len(record.sha256) == 64


def test_tsv_and_malformed_width_rows_are_reported_without_mutation(tmp_path: Path):
    source = tmp_path / "messy.tsv"
    original = "name\tvalue\nalpha\t1\nbeta\t2\textra\ngamma\t3\n"
    source.write_text(original, encoding="utf-8")

    record = data.inspect_file(source)
    assert record.delimiter == "\t"
    assert record.row_count == 3
    assert record.malformed_row_count == 1
    assert record.parse_status == "partial"
    assert record.malformed_rows[0].reason == "column_count_mismatch"
    assert source.read_text(encoding="utf-8") == original


def test_inventory_recurses_skips_hidden_and_records_unsupported_files(tmp_path: Path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "data.csv").write_text("a,b\n1,2\n2,3\n", encoding="utf-8")
    (nested / "notes.md").write_text("hello", encoding="utf-8")
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    (hidden / "secret.csv").write_text("a,b\n9,9\n", encoding="utf-8")

    inventory = data.inspect_paths([tmp_path])
    assert inventory.discovered_file_count == 2
    assert inventory.supported_file_count == 1
    assert inventory.unsupported_file_count == 1
    unsupported = next(record for record in inventory.files if not record.supported)
    assert unsupported.parse_status == "unsupported"
    assert unsupported.sha256 == ""

    with_hidden = data.inspect_paths([tmp_path], include_hidden=True)
    assert with_hidden.discovered_file_count == 3


def test_inventory_records_missing_path_as_top_level_warning(tmp_path: Path):
    inventory = data.inspect_paths([tmp_path / "does-not-exist"])
    assert inventory.discovered_file_count == 0
    assert "path does not exist" in inventory.warnings[0]


def test_inventory_reporting_writes_schema_json(tmp_path: Path):
    source = tmp_path / "sample.csv"
    source.write_text("x,y\n1,2\n2,3\n", encoding="utf-8")
    inventory = data.inspect_paths([source])

    output = data.save_inventory(inventory, tmp_path / "report")
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert output.name == "inventory.json"
    assert payload["schema"] == "ml-lab.data-inventory@1"
    assert payload["summary"]["parsed_file_count"] == 1


def test_xml_inventory_loads_structural_artifact_and_namespaces(tmp_path: Path):
    source = tmp_path / "sample.xml"
    source.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<catalog xmlns="urn:catalog" xmlns:m="urn:metrics" id="root">\n'
        '  <project id="a"><m:metric name="x">1</m:metric></project>\n'
        '  <project id="b"><m:metric name="y">2</m:metric></project>\n'
        '</catalog>\n',
        encoding="utf-8",
    )

    inventory = data.inspect_paths([source])
    record = inventory.files[0]
    assert record.supported is True
    assert record.format == "xml"
    assert record.parse_status == "parsed"
    assert record.tabular_ready is False
    assert len(record.sha256) == 64
    assert record.xml_structure is not None
    artifact = record.xml_structure
    assert artifact.schema == "ml-lab.xml-structure@1"
    assert artifact.source_fingerprint == f"sha256:{record.sha256}"
    assert artifact.root_local_name == "catalog"
    assert artifact.root_namespace_uri == "urn:catalog"
    assert artifact.root_attribute_names == ("id",)
    assert artifact.top_level_child_count == 2
    assert artifact.top_level_element_tags == ("{urn:catalog}project",)
    assert {(item.prefix, item.uri) for item in artifact.namespaces} == {
        ("", "urn:catalog"),
        ("m", "urn:metrics"),
    }
    assert source.read_text(encoding="utf-8").startswith("<?xml")


def test_xml_direct_loader_computes_source_fingerprint(tmp_path: Path):
    source = tmp_path / "plain.xml"
    source.write_text("<root><child /></root>", encoding="utf-8")

    artifact = data.load_xml_structure(source)
    assert artifact.source_sha256
    assert artifact.source_fingerprint == f"sha256:{artifact.source_sha256}"
    assert artifact.root_tag == "root"
    assert artifact.element_count == 2
    assert artifact.observed_max_depth == 2


def test_xml_ingestion_rejects_doctype_and_entity_declarations(tmp_path: Path):
    source = tmp_path / "unsafe.xml"
    source.write_text(
        '<!DOCTYPE root [<!ENTITY local "expanded">]><root>&local;</root>',
        encoding="utf-8",
    )

    record = data.inspect_file(source)
    assert record.supported is True
    assert record.parse_status == "failed"
    assert record.tabular_ready is False
    assert "DTD/entity declarations are not allowed" in (record.error or "")


def test_malformed_xml_is_supported_but_reported_failed(tmp_path: Path):
    source = tmp_path / "broken.xml"
    source.write_text("<root><child></root>", encoding="utf-8")

    record = data.inspect_file(source)
    assert record.supported is True
    assert record.format == "xml"
    assert record.parse_status == "failed"
    assert record.xml_structure is None
    assert "ParseError" in (record.error or "")


def test_raw_xml_is_not_profiled_as_a_tabular_dataset(tmp_path: Path):
    csv_source = tmp_path / "sample.csv"
    csv_source.write_text("x,y\n1,2\n", encoding="utf-8")
    xml_source = tmp_path / "sample.xml"
    xml_source.write_text("<root><row><x>1</x></row></root>", encoding="utf-8")

    collection = data.profile_paths([tmp_path], max_rows=0)
    assert collection.inventory.supported_file_count == 2
    assert len(collection.profiles) == 1
    assert collection.profiles[0].relative_path == "sample.csv"

    comparison = data.compare_paths([tmp_path])
    assert comparison["summary"]["dataset_count"] == 1


def test_xml_structure_artifact_can_be_persisted_as_json(tmp_path: Path):
    source = tmp_path / "sample.xml"
    source.write_text('<root xmlns="urn:test"><row /></root>', encoding="utf-8")
    artifact = data.load_xml_structure(source)

    output = data.save_xml_structure(artifact, tmp_path / "artifacts")
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert output.name == "xml_structure.json"
    assert payload["schema"] == "ml-lab.xml-structure@1"
    assert payload["source_sha256"] == artifact.source_sha256
    assert payload["root"]["local_name"] == "root"
