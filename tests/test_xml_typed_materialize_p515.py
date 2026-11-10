"""P5-15 full extraction guarantees and immutable lineage checks."""
import json
import pytest
from ml_lab.data.xml_ingest import inspect_xml_structure
from ml_lab.data.xml_typed_representation import build_typed_recipe, XmlTypedRepresentationError
from ml_lab.data.xml_typed_materialize import materialize_typed_xml_records, verify_typed_xml_materialization


def _fixture(tmp_path):
    source = tmp_path / "source.xml"
    source.write_text('<doc xmlns:n="urn:n"><record><n:entry>1</n:entry><n:entry>2</n:entry><empty/></record>'
                      '<record><n:entry>3</n:entry><mixed>head<b>x</b>tail</mixed></record></doc>', encoding="utf-8")
    structure = inspect_xml_structure(source)
    root = next(x.canonical_path for x in structure.element_profiles if x.local_name == "record")
    entries = next(x.canonical_path for x in structure.element_profiles if x.local_name == "entry")
    empty = next(x.canonical_path for x in structure.element_profiles if x.local_name == "empty")
    mixed = next(x.canonical_path for x in structure.element_profiles if x.local_name == "mixed")
    recipe = build_typed_recipe(record_root_canonical_path=root, variables={entries: "list", empty: "scalar", mixed: "content_tokens"}, linked_child_paths=(entries,))
    return source, structure, recipe


def test_complete_roundtrip_and_lineage(tmp_path):
    source, structure, recipe = _fixture(tmp_path)
    output = materialize_typed_xml_records(tmp_path / "project", source, structure, recipe)
    assert output["record_count"] == 2 and output["complete"] is True
    manifest = verify_typed_xml_materialization(tmp_path / "project", output["artifact_id"])
    assert manifest["source_sha256"] == structure.source_sha256
    assert manifest["record_count"] == manifest["source_record_count"] == 2
    lines = (tmp_path / "project" / output["relative_records_path"]).read_text().splitlines()
    first, second = [json.loads(line) for line in lines]
    assert [x["value"]["element"]["text"]["value"] for x in first["variables"][0]["items"]] == ["1", "2"]
    assert first["variables"][1]["present"] is True
    assert first["variables"][1]["items"][0]["value"]["present"] is False
    assert second["variables"][1]["present"] is False
    assert second["variables"][2]["items"][0]["value"]["tokens"][1]["kind"] == "child"
    assert second["record_id"].startswith(structure.source_sha256)
    assert materialize_typed_xml_records(tmp_path / "project", source, structure, recipe) == output


def test_limits_do_not_publish_manifest(tmp_path):
    source, structure, recipe = _fixture(tmp_path)
    project = tmp_path / "project"
    with pytest.raises(XmlTypedRepresentationError, match="record limit"):
        materialize_typed_xml_records(project, source, structure, recipe, max_records=1)
    with pytest.raises(XmlTypedRepresentationError, match="byte limit"):
        materialize_typed_xml_records(project, source, structure, recipe, max_output_bytes=10)
    assert not list((project / "artifacts" / "xml_typed_records").glob("*.json"))
    assert not list((project / "artifacts" / "xml_typed_records").glob("*.jsonl"))


def test_source_change_rejected(tmp_path):
    source, structure, recipe = _fixture(tmp_path)
    source.write_text('<doc/>', encoding='utf-8')
    with pytest.raises(XmlTypedRepresentationError, match="differs"):
        materialize_typed_xml_records(tmp_path / "p", source, structure, recipe)


def test_tamper_rejected_and_cross_project_isolated(tmp_path):
    source, structure, recipe = _fixture(tmp_path)
    result = materialize_typed_xml_records(tmp_path / "p", source, structure, recipe)
    with pytest.raises(FileNotFoundError):
        verify_typed_xml_materialization(tmp_path / "other", result["artifact_id"])
    path = tmp_path / "p" / result["relative_records_path"]
    path.write_text("{}\n")
    with pytest.raises(ValueError, match="integrity"):
        verify_typed_xml_materialization(tmp_path / "p", result["artifact_id"])
