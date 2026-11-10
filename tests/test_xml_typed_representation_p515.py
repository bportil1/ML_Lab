import xml.etree.ElementTree as ET

import pytest

from ml_lab.data.xml_typed_representation import (
    RECIPE_SCHEMA, SCHEMA, XmlTypedRepresentationError,
    build_typed_recipe, represent_xml_element,
)


def test_nested_repetition_namespace_mixed_text_and_order():
    root = ET.fromstring('<r xmlns:x="urn:test" a="1">start<x:item>first</x:item>gap<x:item/>end</r>')
    data = represent_xml_element(root)
    node = data['root']
    assert data['schema'] == SCHEMA
    assert [v['value'] for v in node['content_tokens'] if v['kind'] in {'text', 'tail'}] == ['start', 'gap', 'end']
    assert [x['occurrence_index'] for x in node['children']] == [0, 1]
    assert node['children'][0]['name']['namespace_uri'] == 'urn:test'
    assert node['children'][1]['text']['present'] is False
    assert node['child_cardinality'] == [{'expanded_name': '{urn:test}item', 'count': 2}]
    assert node['attributes'][0]['value'] == '1'


def test_missing_vs_empty_and_no_numeric_conversion():
    root = ET.Element('record')
    ET.SubElement(root, 'missing')
    ET.SubElement(root, 'empty').text = ''
    ET.SubElement(root, 'numeric').text = '001.20'
    children = represent_xml_element(root)['root']['children']
    assert children[0]['text'] == {'kind': 'scalar', 'value': None, 'present': False}
    assert children[1]['text'] == {'kind': 'scalar', 'value': '', 'present': True}
    assert children[2]['text']['value'] == '001.20'


def test_recipe_explicit_version_no_default_changes():
    recipe = build_typed_recipe(record_root_canonical_path='/records/record',
                                variables={'/records/record/a': 'list', '/records/record': 'object'},
                                linked_child_paths=('/records/record/a',))
    assert recipe['schema'] == RECIPE_SCHEMA
    assert recipe['version'] == 1
    assert recipe['encoding'] == 'none'
    assert recipe['legacy_extraction_modified'] is False
    with pytest.raises(XmlTypedRepresentationError):
        build_typed_recipe(record_root_canonical_path='/records/record', variables={'/other': 'scalar'})
    with pytest.raises(XmlTypedRepresentationError):
        build_typed_recipe(record_root_canonical_path='/records/record', variables={'/records/record/a': 'float'})


def test_bounded_representation():
    root = ET.fromstring('<r><a><b/></a></r>')
    with pytest.raises(XmlTypedRepresentationError):
        represent_xml_element(root, max_depth=1)
    with pytest.raises(XmlTypedRepresentationError):
        represent_xml_element(root, max_nodes=2)
