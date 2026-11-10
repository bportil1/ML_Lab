import xml.etree.ElementTree as ET

import pytest

from ml_lab.data.xml_ingest import inspect_xml_structure
from ml_lab.data.xml_typed_representation import build_typed_recipe, XmlTypedRepresentationError
from ml_lab.data.xml_typed_records import preview_typed_xml_records, PREVIEW_SCHEMA


def _fixture(tmp_path):
    path = tmp_path / 'source.xml'
    path.write_text('<doc xmlns:n="urn:n"><record><n:entry>1</n:entry><n:entry>2</n:entry>'
                    '<mixed>head<b>hi</b>tail</mixed><empty/></record>'
                    '<record><n:entry>3</n:entry><empty></empty></record></doc>', encoding='utf-8')
    artifact = inspect_xml_structure(path)
    root = next(x.canonical_path for x in artifact.element_profiles if x.local_name == 'record')
    entries = next(x.canonical_path for x in artifact.element_profiles if x.local_name == 'entry')
    mixed = next(x.canonical_path for x in artifact.element_profiles if x.local_name == 'mixed')
    empty = next(x.canonical_path for x in artifact.element_profiles if x.local_name == 'empty')
    recipe = build_typed_recipe(record_root_canonical_path=root,
                                variables={entries: 'list', mixed: 'content_tokens', empty: 'scalar'},
                                linked_child_paths=(entries,))
    return path, artifact, recipe


def test_record_values_order_links_and_missingness(tmp_path):
    path, artifact, recipe = _fixture(tmp_path)
    result = preview_typed_xml_records(path, artifact, recipe, max_records=1)
    assert result['schema'] == PREVIEW_SCHEMA
    assert result['source_sha256'] == artifact.source_sha256
    assert result['truncated'] is True
    assert result['registered_dataset'] is False
    record = result['records'][0]
    items = record['variables'][0]['items']
    assert [x['value']['element']['text']['value'] for x in items] == ['1', '2']
    assert [x['lineage']['occurrence_address'] for x in items] == [[0, 0], [0, 1]]
    assert record['variables'][1]['items'][0]['value']['tokens'][1]['kind'] == 'child'
    assert record['variables'][2]['items'][0]['value'] == {'kind': 'scalar', 'present': False, 'value': None}
    assert len(record['linked_children'][0]['records']) == 2
    assert all(x['parent_record_index'] == 0 for x in record['linked_children'][0]['records'])


def test_missing_path_distinct_from_empty_value(tmp_path):
    path, artifact, recipe = _fixture(tmp_path)
    result = preview_typed_xml_records(path, artifact, recipe)
    variables = result['records'][1]['variables']
    assert variables[1]['present'] is False and variables[1]['count'] == 0 and variables[1]['items'] == []
    assert variables[2]['present'] is True and variables[2]['items'][0]['value']['present'] is False


def test_stale_source_and_invalid_recipe_rejected(tmp_path):
    path, artifact, recipe = _fixture(tmp_path)
    path.write_text('<doc/>', encoding='utf-8')
    with pytest.raises(XmlTypedRepresentationError, match='differs'):
        preview_typed_xml_records(path, artifact, recipe)
    path, artifact, recipe = _fixture(tmp_path)
    recipe['variables'][0]['canonical_path'] = '/unknown'
    with pytest.raises(XmlTypedRepresentationError, match='unknown'):
        preview_typed_xml_records(path, artifact, recipe)


def test_safe_ingestion_policy_and_limits(tmp_path):
    path, artifact, recipe = _fixture(tmp_path)
    with pytest.raises(XmlTypedRepresentationError, match='limits'):
        preview_typed_xml_records(path, artifact, recipe, max_records=201)
    path.write_text('<!DOCTYPE x [<!ENTITY x "bad">]><doc/>', encoding='utf-8')
    with pytest.raises(ValueError):
        preview_typed_xml_records(path, artifact, recipe)
