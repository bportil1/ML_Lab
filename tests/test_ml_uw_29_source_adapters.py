"""ML-UW-29: source adapters preserve structure, provenance and capability gates."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from ml_lab.data.xml_ingest import inspect_xml_structure
from ml_lab.structured import CONTRACT_SCHEMA, StructuredContractError, StructuredObject
from ml_lab.structured.adapters import (
    AdaptationResult, AdapterRegistry, GraphAdapter, JSONAdapter,
    XMLStructureAdapter, default_adapters,
)


FIXTURES = Path(__file__).parent / 'fixtures' / 'structured_uncertainty'


def fixture(name):
    return json.loads((FIXTURES / (name + '.json')).read_text())


def test_default_registry_is_explicit_local_and_extensible():
    a, b = default_adapters(), default_adapters()
    assert a.names == ('json', 'node_edge', 'xml_structure')
    assert a is not b
    a.register('custom', JSONAdapter())
    assert 'custom' in a.names and 'custom' not in b.names
    with pytest.raises(StructuredContractError, match='unique'):
        a.register('json', JSONAdapter())
    with pytest.raises(StructuredContractError, match='unregistered'):
        b.adapt('custom', {})
    with pytest.raises(StructuredContractError, match='must implement'):
        b.register('invalid', object())
    with pytest.raises(StructuredContractError, match='AdaptationResult'):
        bad = AdapterRegistry()
        bad.register('bad', type('Bad', (), {'adapt': lambda self, value: None})())
        bad.adapt('bad', {})


def test_json_adapter_preserves_nested_values_and_array_order():
    source = {'a/b': {'nested': [None, '', False, 2, 3.5, {'~k': 'v'}]}}
    result = default_adapters().adapt('json', source)
    assert isinstance(result, AdaptationResult)
    assert result.lossless
    obj = result.require_lossless()
    assert isinstance(obj, StructuredObject)
    assert obj.topology['kind'] == 'tree'
    assert 'aligned_observations' not in obj.capabilities
    assert 'ordered_children' not in obj.capabilities  # object key order is not semantic
    nodes = {n['id']: n for n in obj.nodes}
    assert nodes['#/a~1b/nested/0']['attributes']['value_state'] == 'explicit_null'
    assert nodes['#/a~1b/nested/1']['attributes']['value_state'] == 'empty'
    assert nodes['#/a~1b/nested/2']['attributes']['json_type'] == 'boolean'
    assert nodes['#/a~1b/nested/3']['attributes']['value'] == 2
    assert nodes['#/a~1b/nested/5/~0k']['attributes']['value'] == 'v'
    by_target = {e['target_id']: e for e in obj.edges}
    assert [by_target[f'#/a~1b/nested/{i}']['ordinal'] for i in range(6)] == list(range(6))
    assert StructuredObject.from_json(obj.to_json()).to_dict() == obj.to_dict()


def test_json_object_key_order_does_not_affect_fingerprint_or_object_identity():
    first = JSONAdapter().adapt({'b': [1, 2], 'a': 'x'}).object
    second = JSONAdapter().adapt({'a': 'x', 'b': [1, 2]}).object
    assert first.object_id == second.object_id
    assert first.to_json() == second.to_json()
    changed = JSONAdapter().adapt({'a': 'x', 'b': [2, 1]}).object
    assert changed.object_id != first.object_id


@pytest.mark.parametrize('value', [None, '', 3, True, [], {}])
def test_json_root_scalar_and_empty_containers(value):
    obj = JSONAdapter().adapt(value).object
    assert len(obj.nodes) == 1
    assert len(obj.edges) == 0
    obj.require('hierarchy_summary')


def test_json_text_strict_duplicate_and_nonfinite_rejection():
    obj = JSONAdapter().from_text('{"a":1,"b":[null,2]}').object
    assert len(obj.nodes) == 5
    with pytest.raises(StructuredContractError, match='duplicate JSON field'):
        JSONAdapter().from_text('{"a":1,"a":2}')
    with pytest.raises(StructuredContractError):
        JSONAdapter().from_text('{"x":NaN}')
    with pytest.raises(StructuredContractError):
        JSONAdapter().adapt({'x': float('inf')})
    with pytest.raises(StructuredContractError):
        JSONAdapter().adapt({12: 'bad'})
    with pytest.raises(StructuredContractError):
        JSONAdapter().adapt({'wrong': {1, 2}})


def test_json_input_depth_and_count_limits_are_enforced():
    with pytest.raises(StructuredContractError, match='node count'):
        JSONAdapter(max_nodes=2).adapt([1, 2])
    with pytest.raises(StructuredContractError, match='depth'):
        JSONAdapter(max_depth=2).adapt([[[['leaf']]]])


def test_xml_adapter_uses_existing_inspected_structure_and_preserves_namespace(tmp_path):
    path = tmp_path / 'namespaced.xml'
    path.write_text('''<a:root xmlns:a="urn:example/one" xmlns:b="urn:example/two">
        <b:item id="one"><b:amount>3</b:amount></b:item>
        <b:item><b:amount/></b:item>
    </a:root>''')
    inspected = inspect_xml_structure(path)
    adapter = XMLStructureAdapter()
    from_dataclass = adapter.adapt(inspected)
    from_record = adapter.adapt(inspected.to_record())
    assert from_dataclass.object.to_json() == from_record.object.to_json()
    assert from_dataclass.object.topology['kind'] == 'tree'
    assert from_dataclass.lossless is False
    with pytest.raises(StructuredContractError, match='lossy'):
        from_dataclass.require_lossless()
    obj = from_dataclass.object
    assert 'aligned_observations' not in obj.capabilities
    assert len(obj.observations) == 0
    element_ids = {n['id'] for n in obj.nodes if n['kind'] == 'element'}
    assert '/{urn:example~1one}root' in element_ids
    assert '/{urn:example~1one}root/{urn:example~1two}item' in element_ids
    item = next(n for n in obj.nodes if n['kind'] == 'element' and n['attributes']['canonical_path'].endswith('item'))
    assert item['attributes']['occurrence_count'] == 2
    assert item['metadata']['source_profile']['cardinality']['repeated'] is True
    assert any(n['kind'] == 'attribute' and n['attributes']['xml_name'] == 'id' for n in obj.nodes)
    assert obj.provenance['source_sha256_claim'] == inspected.source_sha256
    assert obj.provenance['source_fingerprint_claim'] == inspected.source_fingerprint


def test_xml_namespace_prefix_aliases_have_identical_node_ids(tmp_path):
    a, b = tmp_path / 'one.xml', tmp_path / 'two.xml'
    a.write_text('<p:root xmlns:p="urn:shared"><p:child>one</p:child></p:root>')
    b.write_text('<q:root xmlns:q="urn:shared"><q:child>two</q:child></q:root>')
    first = XMLStructureAdapter().adapt(inspect_xml_structure(a)).object
    second = XMLStructureAdapter().adapt(inspect_xml_structure(b)).object
    assert {n['id'] for n in first.nodes} == {n['id'] for n in second.nodes}
    # Different source content should *not* be advertised as an identical object.
    assert first.object_id != second.object_id


def test_xml_profile_is_not_misrepresented_as_raw_instances(tmp_path):
    path = tmp_path / 'x.xml'
    path.write_text('<root><item>one</item><item>two</item></root>')
    result = XMLStructureAdapter().adapt(inspect_xml_structure(path))
    assert len(result.object.nodes) == 2  # only canonical profiles, not two 'item' instances
    assert 'individual_occurrence_topology' in result.losses
    assert 'record_level_text_and_attribute_values' in result.losses
    assert result.warnings and 'does not re-read' in result.warnings[0]


@pytest.mark.parametrize('mutation', [
    lambda record: record['element_profiles'][0].__setitem__('canonical_path', '/forged'),
    lambda record: record['element_profiles'][1].__setitem__('canonical_path', '/missing/child'),
    lambda record: record['element_profiles'].append(deepcopy(record['element_profiles'][0])),
    lambda record: record.__setitem__('source_sha256', 'not-a-sha'),
])
def test_xml_rejects_broken_profiles(tmp_path, mutation):
    path = tmp_path / 'x.xml'
    path.write_text('<root><child/></root>')
    record = inspect_xml_structure(path).to_record()
    mutation(record)
    with pytest.raises(StructuredContractError):
        XMLStructureAdapter().adapt(record)


@pytest.mark.parametrize('name', ('xml_tree','ast_tree','shared_child_dag','cyclic_network'))
def test_generic_graph_accepts_existing_validated_fixtures_without_domain_branch(name):
    fixture_data = fixture(name)
    result = GraphAdapter().adapt(fixture_data)
    assert result.lossless
    assert result.object.to_dict()['nodes'] == fixture_data['nodes']
    assert result.object.to_dict()['edges'] == fixture_data['edges']
    assert GraphAdapter().adapt(result.object).object is result.object


def test_generic_graph_accepts_typed_exports_and_rejects_forged_capabilities():
    source = {'schema': 'ml-lab.node-edge@1', 'topology': 'dag',
              'nodes': [{'id': n, 'kind': 'uninterpreted', 'attributes': {}}
                        for n in ('a', 'b', 'c')],
              'edges': [{'id': 'ab', 'kind': 'links', 'source_id': 'a', 'target_id': 'b'},
                        {'id': 'ac', 'kind': 'links', 'source_id': 'a', 'target_id': 'c'},
                        {'id': 'bc', 'kind': 'links', 'source_id': 'b', 'target_id': 'c'}],
              'source_id': 'export-123'}
    obj = GraphAdapter().adapt(source).object
    assert obj.topology['kind'] == 'dag'
    assert obj.provenance['source_id'] == 'export-123'
    assert 'acyclic' in obj.capabilities
    obj.require('graph_summary')
    with pytest.raises(StructuredContractError, match='rooted_hierarchy'):
        obj.require('hierarchy_summary')
    source['topology'] = {'kind': 'dag', 'capabilities': ['graph_edges']}
    with pytest.raises(StructuredContractError, match='mismatch'):
        GraphAdapter().adapt(source)
    source['topology'] = 'dag'
    source['edges'].append({'id': 'ca', 'kind': 'links', 'source_id': 'c', 'target_id': 'a'})
    with pytest.raises(StructuredContractError, match='cycles'):
        GraphAdapter().adapt(source)


def test_generic_graph_refuses_unmapped_fields_and_unaligned_samples():
    src = {'topology':'tree', 'nodes':[{'id':'root','kind':'uninterpreted','attributes':{}}],
           'edges':[], 'unknown_vendor_metadata':'lost'}
    with pytest.raises(StructuredContractError, match='unrecognized'):
        GraphAdapter().adapt(src)
    del src['unknown_vendor_metadata']
    src['observations'] = [{'observation_id':'obs-1', 'values':{'x':{'status':'present','value':1}}}]
    with pytest.raises(StructuredContractError, match='observation unit'):
        GraphAdapter().adapt(src)
    src['observation_unit'] = 'node'
    assert 'aligned_observations' in GraphAdapter().adapt(src).object.capabilities
    src['observations'].append({'observation_id':'obs-2', 'values': {'y': {'status':'present','value':2}}})
    with pytest.raises(StructuredContractError, match='aligned variables'):
        GraphAdapter().adapt(src)
