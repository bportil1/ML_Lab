"""Read-only projection acceptance. No PAH_Shared runtime is required."""
from dataclasses import replace
import pytest
from ml_lab.data.xml_ingest import inspect_xml_structure
from ml_lab.data.xml_shared_hierarchy import (
    project_xml_structure_hierarchy, XmlSharedHierarchyError,
)


def _fixture(tmp_path):
    source = tmp_path / "structure.xml"
    source.write_text('<doc xmlns:x="urn:lab/test"><record><x:item>1</x:item>'
                      '<x:item>2</x:item><empty/></record></doc>', encoding='utf-8')
    return inspect_xml_structure(source)


def test_source_namespace_cardinality_and_parent_identity(tmp_path):
    a = _fixture(tmp_path)
    result = project_xml_structure_hierarchy(a)
    model, prov = result['model'], result['provenance']
    assert model['schema'] == 'pah-shared.hierarchy@1'
    assert prov['source_sha256'] == a.source_sha256
    assert len(model['nodes']) == len(a.element_profiles)
    assert [n['id'] for n in model['nodes']] == [p.canonical_path for p in a.element_profiles]
    assert model['nodes'][0]['parent_id'] is None
    for node in model['nodes'][1:]:
        assert node['parent_id'] in prov['nodes']
    item = next(p for p in a.element_profiles if p.local_name == 'item')
    assert '~1' in item.canonical_path  # escaped slash in namespace URI
    assert prov['nodes'][item.canonical_path]['repeated'] is True
    assert prov['nodes'][item.canonical_path]['occurrence_count'] == 2
    assert model['nodes'][a.element_profiles.index(item)]['kind'] == 'xml-element'


def test_no_mutation_or_reparsing(tmp_path):
    a = _fixture(tmp_path)
    original = a.to_record()
    one = project_xml_structure_hierarchy(a)
    two = project_xml_structure_hierarchy(a)
    assert one == two
    assert a.to_record() == original
    assert set(one['model']) == {'schema','title','nodes'}


def test_rejects_missing_parent_duplicate_and_overlong_id(tmp_path):
    a = _fixture(tmp_path)
    p = a.element_profiles[-1]
    with pytest.raises(XmlSharedHierarchyError, match='Duplicate'):
        project_xml_structure_hierarchy(replace(a, element_profiles=a.element_profiles + (p,)))
    with pytest.raises(XmlSharedHierarchyError, match='Parent'):
        project_xml_structure_hierarchy(replace(a, element_profiles=(p,)))
    with pytest.raises(XmlSharedHierarchyError, match='node ID'):
        project_xml_structure_hierarchy(replace(a, element_profiles=(replace(a.element_profiles[0], canonical_path='/'+'x'*96),)))


def test_rejects_depth_limit_and_excess_nodes(tmp_path):
    a = _fixture(tmp_path)
    first = a.element_profiles[0]
    with pytest.raises(XmlSharedHierarchyError, match='depth'):
        project_xml_structure_hierarchy(replace(a, element_profiles=(replace(first,depth=65),)))
    with pytest.raises(XmlSharedHierarchyError, match='5,000'):
        project_xml_structure_hierarchy(replace(a, element_profiles=(first,)*5001))
