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


def test_xml_record_root_selection_derives_relative_scalar_fields(tmp_path: Path):
    source = tmp_path / "records.xml"
    source.write_text(
        "<catalog>"
        "<project id='a'><name>A</name><metrics><cwe id='190' count='2'/><cwe id='191' count='1'/></metrics></project>"
        "<project id='b'><name>B</name><metrics><cwe id='190' count='3'/></metrics></project>"
        "</catalog>",
        encoding="utf-8",
    )
    artifact = data.analyze_xml_structure(source)
    selection = data.build_xml_record_selection(artifact, "/catalog/project")

    assert selection.schema == "ml-lab.xml-record-selection@1"
    assert selection.record_root_path == "/catalog/project"
    assert selection.record_root_occurrence_count == 2
    fields = {field.relative_path: field for field in selection.fields}
    assert fields["@id"].kind == "attribute"
    assert fields["@id"].likely_identifier is True
    assert fields["name"].kind == "element_text"
    assert fields["name"].repeated is False
    assert fields["metrics/cwe/@id"].repeated is True
    assert fields["metrics/cwe/@count"].repeated is True
    assert all(field.source_element_canonical_path.startswith("/catalog/project") for field in selection.fields)


def test_xml_record_root_selection_rejects_unknown_path(tmp_path: Path):
    source = tmp_path / "records.xml"
    source.write_text("<root><row><value>1</value></row></root>", encoding="utf-8")
    artifact = data.analyze_xml_structure(source)
    with pytest.raises(data.XmlSelectionError, match="not present"):
        data.build_xml_record_selection(artifact, "/root/missing")


def test_xml_collection_plan_identifies_repeated_branches_and_requires_explicit_rules(tmp_path: Path):
    source = tmp_path / "collections.xml"
    source.write_text(
        "<catalog>"
        "<project id='a'><metrics><cwe id='190' count='2'/><cwe id='191' count='1'/></metrics></project>"
        "<project id='b'><metrics><cwe id='190' count='3'/></metrics></project>"
        "</catalog>",
        encoding="utf-8",
    )
    artifact = data.analyze_xml_structure(source)
    selection = data.build_xml_record_selection(artifact, "/catalog/project")
    selected = [field.field_id for field in selection.fields if field.relative_path in {"@id", "metrics/cwe/@id", "metrics/cwe/@count"}]

    plan = data.build_xml_collection_plan(artifact, "/catalog/project", selected)

    assert plan.schema == "ml-lab.xml-collection-plan@1"
    assert plan.unresolved_branch_count == 1
    assert plan.invalid_rule_count == 0
    assert plan.ready_for_preview is False
    assert len(plan.repeated_branches) == 1
    branch = plan.repeated_branches[0]
    assert branch.path == "/catalog/project/metrics/cwe"
    assert branch.relative_path == "metrics/cwe"
    assert branch.max_per_parent == 2
    assert {field.relative_path for field in branch.selected_fields} == {"metrics/cwe/@id", "metrics/cwe/@count"}
    assert branch.rule is not None
    assert branch.rule.strategy is None
    assert branch.rule.valid is False

    strategy_ids = {item["id"] for item in plan.strategy_catalog}
    assert strategy_ids == {
        "keep_nested", "first", "last", "count", "join", "aggregate", "pivot", "explode_rows", "separate_table"
    }


def test_xml_collection_plan_validates_strategy_specific_options(tmp_path: Path):
    source = tmp_path / "rules.xml"
    source.write_text(
        "<root><row><item key='a' value='1'/><item key='b' value='2'/></row>"
        "<row><item key='c' value='3'/></row></root>",
        encoding="utf-8",
    )
    artifact = data.analyze_xml_structure(source)
    selection = data.build_xml_record_selection(artifact, "/root/row")
    selected = [field.field_id for field in selection.fields]
    initial = data.build_xml_collection_plan(artifact, "/root/row", selected)
    branch = initial.repeated_branches[0]
    branch_path = branch.canonical_path
    key_id = next(field.field_id for field in branch.selected_fields if field.relative_path.endswith("@key"))
    value_id = next(field.field_id for field in branch.selected_fields if field.relative_path.endswith("@value"))

    def plan_for(strategy: str, **options):
        return data.build_xml_collection_plan(
            artifact,
            "/root/row",
            selected,
            rules=[{"branch_canonical_path": branch_path, "strategy": strategy, "options": options}],
        )

    for strategy in ("keep_nested", "first", "last", "count", "explode_rows", "separate_table"):
        assert plan_for(strategy).ready_for_preview is True
    assert plan_for("join", join_delimiter=" | ").ready_for_preview is True
    assert plan_for("aggregate", aggregate_operation="mean").ready_for_preview is True
    assert plan_for("pivot", pivot_key_field_id=key_id, pivot_value_field_id=value_id).ready_for_preview is True

    bad_aggregate = plan_for("aggregate")
    assert bad_aggregate.invalid_rule_count == 1
    assert "sum, mean, min, max" in bad_aggregate.repeated_branches[0].rule.errors[0]

    bad_pivot = plan_for("pivot", pivot_key_field_id=key_id, pivot_value_field_id=key_id)
    assert bad_pivot.invalid_rule_count == 1
    assert any("different" in error for error in bad_pivot.repeated_branches[0].rule.errors)


def test_xml_collection_plan_tracks_nested_repeated_branch_ancestry(tmp_path: Path):
    source = tmp_path / "nested.xml"
    source.write_text(
        "<root><row>"
        "<group><item code='a'/><item code='b'/></group>"
        "<group><item code='c'/><item code='d'/></group>"
        "</row></root>",
        encoding="utf-8",
    )
    artifact = data.analyze_xml_structure(source)
    selection = data.build_xml_record_selection(artifact, "/root/row")
    selected = [field.field_id for field in selection.fields if field.relative_path == "group/item/@code"]
    plan = data.build_xml_collection_plan(artifact, "/root/row", selected)

    assert [branch.relative_path for branch in plan.repeated_branches] == ["group", "group/item"]
    assert plan.repeated_branches[0].parent_repeated_branch_canonical_path is None
    assert plan.repeated_branches[1].parent_repeated_branch_canonical_path == plan.repeated_branches[0].canonical_path


def test_xml_collection_plan_rejects_unknown_selected_fields_and_rule_branches(tmp_path: Path):
    source = tmp_path / "invalid-rules.xml"
    source.write_text("<root><row><item x='1'/><item x='2'/></row></root>", encoding="utf-8")
    artifact = data.analyze_xml_structure(source)

    with pytest.raises(data.XmlCollectionRuleError, match="selected field is not available"):
        data.build_xml_collection_plan(artifact, "/root/row", ["attribute:/not/a/field@x"])

    selection = data.build_xml_record_selection(artifact, "/root/row")
    selected = [field.field_id for field in selection.fields]
    with pytest.raises(data.XmlCollectionRuleError, match="does not match a repeated branch"):
        data.build_xml_collection_plan(
            artifact,
            "/root/row",
            selected,
            rules=[{"branch_canonical_path": "/root/row/missing", "strategy": "first"}],
        )


def test_xml_tabular_preview_materializes_pivot_columns_without_mutating_source(tmp_path: Path):
    source = tmp_path / "preview.xml"
    original = (
        "<catalog>"
        "<project id='a'><name>A</name><cwe id='190' count='2'/><cwe id='191' count='1'/></project>"
        "<project id='b'><name>B</name><cwe id='190' count='3'/></project>"
        "</catalog>"
    )
    source.write_text(original, encoding="utf-8")
    artifact = data.analyze_xml_structure(source)
    selection = data.build_xml_record_selection(artifact, "/catalog/project")
    fields = {field.relative_path: field for field in selection.fields}
    selected = [fields[name].field_id for name in ("@id", "name", "cwe/@id", "cwe/@count")]
    rules = [{
        "branch_canonical_path": "/catalog/project/cwe",
        "strategy": "pivot",
        "options": {
            "pivot_key_field_id": fields["cwe/@id"].field_id,
            "pivot_value_field_id": fields["cwe/@count"].field_id,
        },
    }]

    preview = data.preview_xml_tabularization(
        source,
        artifact,
        "/catalog/project",
        selected,
        rules=rules,
        max_rows=25,
    )

    payload = preview.to_record()
    assert payload["schema"] == "ml-lab.xml-tabular-preview@1"
    assert payload["preview_signature"].startswith("sha256:")
    assert payload["source_fingerprint"] == artifact.source_fingerprint
    main = payload["main_table"]
    assert [column["name"] for column in main["columns"]] == ["@id", "name", "cwe[190]", "cwe[191]"]
    assert main["rows"] == [
        {"@id": "a", "name": "A", "cwe[190]": "2", "cwe[191]": "1"},
        {"@id": "b", "name": "B", "cwe[190]": "3", "cwe[191]": None},
    ]
    pivot_columns = [column for column in main["columns"] if column["strategy"] == "pivot"]
    assert all(column["dynamic"] is True for column in pivot_columns)
    assert all(column["source_element_canonical_paths"] == ["/catalog/project/cwe"] for column in pivot_columns)
    assert source.read_text(encoding="utf-8") == original


def test_xml_tabular_preview_supports_explode_and_separate_table_rules(tmp_path: Path):
    source = tmp_path / "collections.xml"
    source.write_text(
        "<root>"
        "<row id='a'><item code='x' value='1'/><item code='y' value='2'/></row>"
        "<row id='b'><item code='z' value='3'/></row>"
        "</root>",
        encoding="utf-8",
    )
    artifact = data.analyze_xml_structure(source)
    selection = data.build_xml_record_selection(artifact, "/root/row")
    fields = {field.relative_path: field for field in selection.fields}
    selected = [fields[name].field_id for name in ("@id", "item/@code", "item/@value")]

    exploded = data.preview_xml_tabularization(
        source,
        artifact,
        "/root/row",
        selected,
        rules=[{"branch_canonical_path": "/root/row/item", "strategy": "explode_rows"}],
    ).to_record()
    assert exploded["main_table"]["preview_row_count"] == 3
    assert exploded["main_table"]["rows"][0]["@id"] == "a"
    assert [row["item/@code"] for row in exploded["main_table"]["rows"]] == ["x", "y", "z"]

    separated = data.preview_xml_tabularization(
        source,
        artifact,
        "/root/row",
        selected,
        rules=[{
            "branch_canonical_path": "/root/row/item",
            "strategy": "separate_table",
            "options": {"separate_table_name": "items"},
        }],
    ).to_record()
    assert separated["main_table"]["rows"] == [{"@id": "a"}, {"@id": "b"}]
    assert len(separated["child_tables"]) == 1
    child = separated["child_tables"][0]
    assert child["name"] == "items"
    assert child["rows"] == [
        {"__parent_record": 1, "@code": "x", "@value": "1"},
        {"__parent_record": 1, "@code": "y", "@value": "2"},
        {"__parent_record": 2, "@code": "z", "@value": "3"},
    ]


def test_xml_tabular_preview_requires_ready_rules_and_rejects_stale_source(tmp_path: Path):
    source = tmp_path / "guarded.xml"
    source.write_text("<root><row><item x='1'/><item x='2'/></row></root>", encoding="utf-8")
    artifact = data.analyze_xml_structure(source)
    selection = data.build_xml_record_selection(artifact, "/root/row")
    selected = [field.field_id for field in selection.fields if field.relative_path == "item/@x"]

    with pytest.raises(data.XmlPreviewError, match="rules are not ready"):
        data.preview_xml_tabularization(source, artifact, "/root/row", selected, rules=[])

    source.write_text("<root><row><item x='changed'/></row></root>", encoding="utf-8")
    with pytest.raises(data.XmlPreviewError, match="source changed"):
        data.preview_xml_tabularization(
            source,
            artifact,
            "/root/row",
            selected,
            rules=[{"branch_canonical_path": "/root/row/item", "strategy": "first"}],
        )


def test_xml_confirmed_extraction_materializes_normal_dataset_and_structure_artifact(tmp_path: Path):
    source = tmp_path / "projects.xml"
    source.write_text(
        '<catalog>'
        '<project id="p1"><name>A</name><metric key="x">1</metric><metric key="y">2</metric></project>'
        '<project id="p2"><name>B</name><metric key="x">3</metric></project>'
        '</catalog>',
        encoding="utf-8",
    )
    artifact = data.analyze_xml_structure(source)
    selection = data.build_xml_record_selection(artifact, "/catalog/project")
    field_by_relative = {field.relative_path: field for field in selection.fields}
    selected = [
        field_by_relative["@id"].field_id,
        field_by_relative["name"].field_id,
        field_by_relative["metric/@key"].field_id,
        field_by_relative["metric"].field_id,
    ]
    rules = [{
        "branch_canonical_path": "/catalog/project/metric",
        "strategy": "pivot",
        "options": {
            "pivot_key_field_id": field_by_relative["metric/@key"].field_id,
            "pivot_value_field_id": field_by_relative["metric"].field_id,
        },
    }]
    preview = data.preview_xml_tabularization(
        source,
        artifact,
        "/catalog/project",
        selected,
        rules=rules,
        max_rows=2,
    )

    output = tmp_path / "derived" / "projects.csv"
    result = data.materialize_xml_tabularization(
        source,
        artifact,
        "/catalog/project",
        selected,
        rules=rules,
        confirmed_preview_signature=preview.preview_signature,
        output=output,
    )

    assert result.dataset_path == str(output.resolve())
    assert result.row_count == 2
    assert result.column_count == 4
    assert result.columns == ("@id", "name", "metric[x]", "metric[y]")
    assert output.is_file()
    record = data.inspect_file(output)
    assert record.format == "delimited_text"
    assert record.tabular_ready is True
    assert record.parse_status == "parsed"
    assert record.row_count == 2
    assert record.columns == result.columns
    profile = data.profile_file(output, max_rows=0, relationship_rows=0)
    assert profile.profiled_row_count == 2
    assert tuple(item.name for item in profile.columns) == result.columns

    structure_path = Path(result.structure_artifact_path)
    structure_payload = json.loads(structure_path.read_text(encoding="utf-8"))
    assert structure_payload["schema"] == "ml-lab.xml-structure@2"
    assert structure_payload["source_fingerprint"] == artifact.source_fingerprint

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["schema"] == "ml-lab.xml-materialization@1"
    assert manifest["source"]["fingerprint"] == artifact.source_fingerprint
    assert manifest["extraction"]["preview_signature"] == preview.preview_signature
    assert manifest["dataset"]["path"] == str(output.resolve())
    assert manifest["provenance"]["status"] == "recorded"
    provenance_path = Path(result.provenance_path)
    assert provenance_path.is_file()
    event = data.load_provenance_event(provenance_path)
    assert event["operation_kind"] == "xml_tabularization"
    assert event["source"]["source_type"] == "xml"
    assert event["source"]["sha256"] == artifact.source_sha256
    assert event["derived"]["logical_sha256"] == data.describe_dataset(output)["logical_sha256"]
    assert event["recipe"]["snapshot"]["record_root_canonical_path"] == "/catalog/project"
    assert event["recipe"]["snapshot"]["preview_signature"] == preview.preview_signature
    assert event["structure_artifact"]["sha256"]

    lineage = data.trace_lineage(output)
    assert lineage["authoritative"] is True
    assert lineage["event_count"] == 1
    assert lineage["root"]["source_type"] == "xml"
    assert lineage["root"]["path"] == str(source.resolve())


def test_xml_materialization_writes_separate_table_as_supplemental_dataset(tmp_path: Path):
    source = tmp_path / "orders.xml"
    source.write_text(
        '<orders>'
        '<order id="o1"><item sku="a">2</item><item sku="b">3</item></order>'
        '<order id="o2"><item sku="c">1</item></order>'
        '</orders>',
        encoding="utf-8",
    )
    artifact = data.analyze_xml_structure(source)
    selection = data.build_xml_record_selection(artifact, "/orders/order")
    fields = {field.relative_path: field for field in selection.fields}
    selected = [fields["@id"].field_id, fields["item/@sku"].field_id, fields["item"].field_id]
    rules = [{
        "branch_canonical_path": "/orders/order/item",
        "strategy": "separate_table",
        "options": {"separate_table_name": "items"},
    }]
    preview = data.preview_xml_tabularization(source, artifact, "/orders/order", selected, rules=rules)
    result = data.materialize_xml_tabularization(
        source,
        artifact,
        "/orders/order",
        selected,
        rules=rules,
        confirmed_preview_signature=preview.preview_signature,
        output=tmp_path / "orders.csv",
    )

    assert result.row_count == 2
    assert result.columns == ("@id",)
    assert len(result.child_table_paths) == 1
    child = Path(result.child_table_paths[0])
    assert child.name == "orders__items.csv"
    child_record = data.inspect_file(child)
    assert child_record.row_count == 3
    assert child_record.columns == ("__parent_record", "@sku", "item")


def test_xml_materialization_rejects_unconfirmed_or_stale_preview_signature_without_writing(tmp_path: Path):
    source = tmp_path / "simple.xml"
    source.write_text('<root><row id="1"><value>A</value></row></root>', encoding="utf-8")
    artifact = data.analyze_xml_structure(source)
    selection = data.build_xml_record_selection(artifact, "/root/row")
    selected = [field.field_id for field in selection.fields]
    output = tmp_path / "out.csv"

    with pytest.raises(data.XmlMaterializationError, match="confirmed preview"):
        data.materialize_xml_tabularization(
            source,
            artifact,
            "/root/row",
            selected,
            confirmed_preview_signature="sha256:not-the-current-configuration",
            output=output,
        )

    assert not output.exists()
    assert source.read_text(encoding="utf-8") == '<root><row id="1"><value>A</value></row></root>'


def test_xml_provenance_regenerates_recorded_extraction_and_rejects_changed_source(tmp_path: Path):
    source = tmp_path / "records.xml"
    source.write_text(
        '<root><row id="1"><value>A</value></row><row id="2"><value>B</value></row></root>',
        encoding="utf-8",
    )
    artifact = data.analyze_xml_structure(source)
    selection = data.build_xml_record_selection(artifact, "/root/row")
    selected = [field.field_id for field in selection.fields if field.relative_path in {"@id", "value"}]
    preview = data.preview_xml_tabularization(source, artifact, "/root/row", selected)
    original = data.materialize_xml_tabularization(
        source, artifact, "/root/row", selected,
        confirmed_preview_signature=preview.preview_signature,
        output=tmp_path / "original.csv",
    )

    regenerated = data.regenerate_xml_extraction(
        original.provenance_path,
        output=tmp_path / "regenerated.csv",
    )
    assert Path(regenerated.dataset_path).read_text(encoding="utf-8") == Path(original.dataset_path).read_text(encoding="utf-8")
    original_desc = data.describe_dataset(original.dataset_path)
    regenerated_desc = data.describe_dataset(regenerated.dataset_path)
    assert regenerated_desc["logical_sha256"] == original_desc["logical_sha256"]
    assert Path(regenerated.provenance_path).is_file()

    source.write_text(
        '<root><row id="1"><value>CHANGED</value></row><row id="2"><value>B</value></row></root>',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source hash"):
        data.regenerate_xml_extraction(original.provenance_path, output=tmp_path / "should-not-exist.csv")
    assert not (tmp_path / "should-not-exist.csv").exists()
