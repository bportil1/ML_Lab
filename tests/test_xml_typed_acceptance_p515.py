"""P5-15 offline acceptance: typed preview -> durable representations, no dataset side effects.

These tests exercise real XML parsing and local artifacts without Flask/browser/runtime
services. HTTP/browser checks remain a separate gate before closing the sprint.
"""
import json
from pathlib import Path

import pytest

from ml_lab.data.xml_ingest import inspect_xml_structure
from ml_lab.data.xml_typed_representation import build_typed_recipe, XmlTypedRepresentationError
from ml_lab.data.xml_typed_records import preview_typed_xml_records
from ml_lab.data.xml_typed_artifacts import persist_typed_preview, read_typed_preview
from ml_lab.data.xml_typed_materialize import (
    materialize_typed_xml_records, verify_typed_xml_materialization,
)


@pytest.fixture
def workflow(tmp_path):
    source = tmp_path / 'sample.xml'
    source.write_text(
        '<catalog xmlns:p="urn:part">'
        '<item key="A"><p:tag>x</p:tag><p:tag>y</p:tag><note></note>'
        '<mixed>head<em>middle</em>tail</mixed></item>'
        '<item key="B"><p:tag>z</p:tag></item>'
        '<item key="C"><note/></item>'
        '</catalog>', encoding='utf-8',
    )
    structure = inspect_xml_structure(source)
    paths = {profile.local_name: profile.canonical_path for profile in structure.element_profiles}
    recipe = build_typed_recipe(
        record_root_canonical_path=paths['item'],
        variables={paths['tag']: 'list', paths['note']: 'scalar', paths['mixed']: 'content_tokens'},
    )
    return source, structure, recipe


def test_preview_save_complete_materialize_and_verified_lineage(workflow, tmp_path):
    source, structure, recipe = workflow
    project = tmp_path / 'project'
    preview = preview_typed_xml_records(source, structure, recipe, max_records=2)
    assert preview['preview_record_count'] == 2
    assert preview['source_record_count'] == 3
    assert preview['truncated'] is True
    saved = persist_typed_preview(project, preview, recipe=recipe)
    assert read_typed_preview(project, saved['artifact_id'])['preview'] == preview

    completed = materialize_typed_xml_records(project, source, structure, recipe)
    manifest = verify_typed_xml_materialization(project, completed['artifact_id'])
    assert manifest['complete'] is True
    assert manifest['truncated'] is False
    assert manifest['record_count'] == manifest['source_record_count'] == 3
    assert manifest['source_sha256'] == structure.source_sha256
    assert manifest['recipe'] == recipe
    assert manifest['encoding'] == 'none' and manifest['registered_dataset'] is False

    records = [json.loads(line) for line in (project / completed['relative_records_path']).read_text().splitlines()]
    assert len(records) == 3
    for ordinal, (partial, full) in enumerate(zip(preview['records'], records)):
        assert partial['variables'] == full['variables']
        assert partial['linked_children'] == full['linked_children']
        assert full['record_index'] == ordinal
        assert full['record_id'] == f"{structure.source_sha256}:{recipe['record_root_canonical_path']}:{ordinal}"
    assert records[0]['variables'][0]['count'] == 2
    assert records[1]['variables'][1]['present'] is False
    assert records[0]['variables'][1]['present'] is True
    assert records[0]['variables'][1]['items'][0]['value']['value'] is None
    assert records[0]['variables'][2]['items'][0]['value']['tokens'][1]['kind'] == 'child'
    assert records[2]['variables'][0]['present'] is False
    assert (project / 'artifacts' / 'xml_typed_records').is_dir()
    assert not (project / 'datasets').exists()


def test_project_isolation_and_repeated_publication(workflow, tmp_path):
    source, structure, recipe = workflow
    project_a, project_b = tmp_path / 'a', tmp_path / 'b'
    preview = preview_typed_xml_records(source, structure, recipe, max_records=1)
    snapshot = persist_typed_preview(project_a, preview, recipe=recipe)
    full = materialize_typed_xml_records(project_a, source, structure, recipe)
    assert persist_typed_preview(project_a, preview, recipe=recipe) == snapshot
    assert materialize_typed_xml_records(project_a, source, structure, recipe) == full
    with pytest.raises(FileNotFoundError):
        read_typed_preview(project_b, snapshot['artifact_id'])
    with pytest.raises(FileNotFoundError):
        verify_typed_xml_materialization(project_b, full['artifact_id'])


def test_changed_source_cannot_publish_with_stale_structure(workflow, tmp_path):
    source, structure, recipe = workflow
    source.write_text(source.read_text().replace('>z<', '>changed<'), encoding='utf-8')
    project = tmp_path / 'project'
    with pytest.raises(XmlTypedRepresentationError, match='differs'):
        preview_typed_xml_records(source, structure, recipe, max_records=2)
    with pytest.raises(XmlTypedRepresentationError, match='differs'):
        materialize_typed_xml_records(project, source, structure, recipe)
    assert not list(project.rglob('*.json'))


def test_no_partial_complete_artifact_when_output_limit_hits(workflow, tmp_path):
    source, structure, recipe = workflow
    project = tmp_path / 'project'
    with pytest.raises(XmlTypedRepresentationError, match='byte limit'):
        materialize_typed_xml_records(project, source, structure, recipe, max_output_bytes=20)
    directory = project / 'artifacts' / 'xml_typed_records'
    assert not list(directory.glob('*.json'))
    assert not list(directory.glob('*.jsonl'))
    assert not list(directory.glob('*.tmp'))
