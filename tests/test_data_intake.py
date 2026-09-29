from __future__ import annotations

import json
from pathlib import Path

import pytest

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
    assert artifact.schema == "ml-lab.xml-structure@2"
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
    assert payload["schema"] == "ml-lab.xml-structure@2"
    assert payload["source_sha256"] == artifact.source_sha256
    assert payload["root"]["local_name"] == "root"


def test_xml_structure_analysis_builds_paths_cardinality_and_optional_branches(tmp_path: Path):
    source = tmp_path / "structure.xml"
    source.write_text(
        "<catalog>"
        '<project id="p1"><name>A</name><metric>1</metric><metric>2</metric><note>x</note></project>'
        '<project id="p2"><name>B</name><metric>3</metric></project>'
        '<project id="p3"><name>C</name><metric>4</metric><metric>5</metric><metric>6</metric></project>'
        "</catalog>",
        encoding="utf-8",
    )

    artifact = data.analyze_xml_structure(source)
    assert artifact.schema == "ml-lab.xml-structure@2"
    assert artifact.unique_element_path_count == 5
    assert artifact.leaf_path_count == 3
    assert artifact.repeated_path_count == 2
    assert artifact.optional_path_count == 1

    profiles = {item.path: item for item in artifact.element_profiles}
    project = profiles["/catalog/project"]
    assert project.occurrence_count == 3
    assert project.parent_occurrence_count == 1
    assert project.parents_with_element == 1
    assert project.min_per_parent == 3
    assert project.max_per_parent == 3
    assert project.mean_per_parent == 3.0
    assert project.repeated is True
    assert project.optional is False
    assert project.child_paths == (
        "/catalog/project/name",
        "/catalog/project/metric",
        "/catalog/project/note",
    )

    metric = profiles["/catalog/project/metric"]
    assert metric.occurrence_count == 6
    assert metric.parent_occurrence_count == 3
    assert metric.parents_with_element == 3
    assert metric.min_per_parent == 1
    assert metric.max_per_parent == 3
    assert metric.mean_per_parent == 2.0
    assert metric.repeated is True
    assert metric.optional is False
    assert metric.text_occurrence_count == 6
    assert metric.text_presence_rate == 1.0

    note = profiles["/catalog/project/note"]
    assert note.occurrence_count == 1
    assert note.parent_occurrence_count == 3
    assert note.parents_with_element == 1
    assert note.min_per_parent == 0
    assert note.max_per_parent == 1
    assert note.mean_per_parent == pytest.approx(1 / 3, abs=1e-6)
    assert note.repeated is False
    assert note.optional is True


def test_xml_structure_analysis_profiles_attributes_and_likely_identifiers(tmp_path: Path):
    source = tmp_path / "ids.xml"
    source.write_text(
        "<root>"
        '<record id="r1" kind="a"><value>1</value></record>'
        '<record id="r2" kind="a"><value>2</value></record>'
        '<record id="r3"><value>3</value></record>'
        "</root>",
        encoding="utf-8",
    )

    artifact = data.analyze_xml_structure(source)
    record = next(item for item in artifact.element_profiles if item.path == "/root/record")
    attributes = {item.local_name: item for item in record.attributes}

    identifier = attributes["id"]
    assert identifier.occurrence_count == 3
    assert identifier.presence_rate == 1.0
    assert identifier.distinct_sample_count == 3
    assert identifier.distinct_sample_rate == 1.0
    assert identifier.likely_identifier is True
    assert record.likely_identifier_attributes == ("id",)

    kind = attributes["kind"]
    assert kind.occurrence_count == 2
    assert kind.presence_rate == pytest.approx(2 / 3, abs=1e-6)
    assert kind.distinct_sample_count == 1
    assert kind.likely_identifier is False

    candidate = next(item for item in artifact.record_candidates if item.path == "/root/record")
    assert candidate.score >= 0.9
    assert "repeats_within_parent" in candidate.reasons
    assert "has_likely_identifier" in candidate.reasons


def test_xml_structure_analysis_preserves_namespace_aware_paths(tmp_path: Path):
    source = tmp_path / "namespaced.xml"
    source.write_text(
        '<c:catalog xmlns:c="urn:catalog" xmlns:m="urn:metrics">'
        '<c:project c:id="p1"><m:metric m:id="m1">1</m:metric></c:project>'
        '<c:project c:id="p2"><m:metric m:id="m2">2</m:metric></c:project>'
        "</c:catalog>",
        encoding="utf-8",
    )

    artifact = data.analyze_xml_structure(source)
    paths = {item.path: item for item in artifact.element_profiles}
    assert "/c:catalog/c:project" in paths
    assert "/c:catalog/c:project/m:metric" in paths
    metric = paths["/c:catalog/c:project/m:metric"]
    assert metric.canonical_path == "/{urn:catalog}catalog/{urn:catalog}project/{urn:metrics}metric"
    assert metric.namespace_uri == "urn:metrics"
    assert metric.prefix == "m"
    assert metric.attributes[0].prefix == "m"
    assert metric.attributes[0].likely_identifier is True


def test_xml_attribute_identifier_sampling_is_bounded(tmp_path: Path):
    source = tmp_path / "bounded.xml"
    source.write_text(
        "<root>" + "".join(f'<row id="{index}" />' for index in range(20)) + "</root>",
        encoding="utf-8",
    )

    artifact = data.analyze_xml_structure(source, max_attribute_value_samples=5)
    row = next(item for item in artifact.element_profiles if item.path == "/root/row")
    identifier = row.attributes[0]
    assert identifier.occurrence_count == 20
    assert identifier.sampled_value_count == 5
    assert identifier.distinct_sample_count == 5
    assert identifier.likely_identifier is True


def test_xml_structure_json_persists_sprint2_analysis(tmp_path: Path):
    source = tmp_path / "sample.xml"
    source.write_text('<root><row id="1"/><row id="2"/></root>', encoding="utf-8")
    artifact = data.analyze_xml_structure(source)

    output = data.save_xml_structure(artifact, tmp_path / "artifacts")
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "ml-lab.xml-structure@2"
    assert payload["analysis"]["unique_element_path_count"] == 2
    assert payload["analysis"]["repeated_path_count"] == 1
    assert payload["element_profiles"][1]["path"] == "/root/row"
    assert payload["element_profiles"][1]["attributes"][0]["likely_identifier"] is True
    assert payload["record_candidates"][0]["path"] == "/root/row"


def test_xml_structure_analysis_detects_identifier_text_children(tmp_path: Path):
    source = tmp_path / "child_ids.xml"
    source.write_text(
        "<root>"
        "<record><id>r1</id><value>A</value></record>"
        "<record><id>r2</id><value>B</value></record>"
        "<record><id>r3</id><value>C</value></record>"
        "</root>",
        encoding="utf-8",
    )

    artifact = data.analyze_xml_structure(source)
    profiles = {item.path: item for item in artifact.element_profiles}
    identifier = profiles["/root/record/id"]
    assert identifier.likely_identifier_text is True
    assert identifier.text_sampled_value_count == 3
    assert identifier.text_distinct_sample_count == 3
    assert identifier.text_distinct_sample_rate == 1.0

    record = profiles["/root/record"]
    assert "has_likely_identifier" in record.record_candidate_reasons


def test_xml_canonical_paths_escape_namespace_uri_separators(tmp_path: Path):
    source = tmp_path / "uri.xml"
    source.write_text(
        '<root xmlns:x="https://example.test/ns/v1"><x:row /></root>',
        encoding="utf-8",
    )

    artifact = data.analyze_xml_structure(source)
    row = next(item for item in artifact.element_profiles if item.local_name == "row")
    assert row.path == "/root/x:row"
    assert row.canonical_path == "/root/{https:~1~1example.test~1ns~1v1}row"
