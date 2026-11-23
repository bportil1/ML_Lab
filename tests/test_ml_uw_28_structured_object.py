"""ML-UW-28: source-neutral immutable graph / observation object tests."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest

from ml_lab.structured import CONTRACT_SCHEMA, StructuredContractError, StructuredObject

_FIXTURES = Path(__file__).parent / 'fixtures' / 'structured_uncertainty'


def example(name: str = 'xml_tree') -> dict:
    data = json.loads((_FIXTURES / f'{name}.json').read_text())
    # Fixture labels live in provenance rather than the scientific object schema.
    return data


@pytest.mark.parametrize('case', ('xml_tree', 'ast_tree', 'shared_child_dag', 'cyclic_network'))
def test_all_prior_fixtures_validate_and_roundtrip(case):
    raw = example(case)
    obj = StructuredObject.from_dict(raw)
    assert obj.schema == CONTRACT_SCHEMA
    assert obj.to_dict()['topology']['kind'] == raw['topology']['kind']
    assert set(obj.topology['capabilities']) == set(raw['topology']['capabilities'])
    assert obj.to_dict()['nodes'] == raw['nodes']
    assert obj.to_dict()['edges'] == raw['edges']
    assert obj.to_dict()['observations'] == raw['observations']
    assert obj.to_dict()['provenance'] == raw['provenance']
    assert obj.object_id.startswith('sha256:')
    assert StructuredObject.from_json(obj.to_json()).to_dict() == obj.to_dict()
    assert StructuredObject.from_json(obj.to_json(indent=2)).to_json() == obj.to_json()
    assert StructuredObject.from_dict(raw).object_id == obj.object_id


def test_immutability_and_detached_copies():
    original = example()
    obj = StructuredObject.from_dict(original)
    original['nodes'][0]['kind'] = 'changed'
    original['observations'][1]['values']['value']['source_reason'] = 'changed'
    assert obj.nodes[0]['kind'] != 'changed'
    assert obj.observations[1]['values']['value']['source_reason'] == 'path_absent'
    with pytest.raises((TypeError, FrozenInstanceError, AttributeError)):
        obj.topology['kind'] = 'dag'
    with pytest.raises((TypeError, FrozenInstanceError, AttributeError)):
        obj.observations[0]['values']['value']['value'] = 99
    with pytest.raises((TypeError, FrozenInstanceError, AttributeError)):
        obj.observations[0]['values'] = {}
    with pytest.raises((TypeError, FrozenInstanceError, AttributeError)):
        obj.nodes[0]['kind'] = 'tampered'
    with pytest.raises((TypeError, FrozenInstanceError, AttributeError)):
        obj.object_id = 'injected'
    detached = obj.to_dict()
    detached['nodes'][0]['kind'] = 'tampered'
    assert obj.nodes[0]['kind'] == 'uninterpreted'


def test_topology_gates_from_actual_edges():
    tree = StructuredObject.from_dict(example('xml_tree'))
    tree.require('hierarchy_summary')
    tree.require('conditional_information')
    with pytest.raises(StructuredContractError, match='explicit_probability_model'):
        tree.require('hierarchical_probability_inference')
    dag = StructuredObject.from_dict(example('shared_child_dag'))
    dag.require('graph_summary')
    with pytest.raises(StructuredContractError, match='rooted_hierarchy'):
        dag.require('depth_reduction')
    cycle = StructuredObject.from_dict(example('cyclic_network'))
    assert 'acyclic' not in cycle.capabilities
    assert 'unique_parent' not in cycle.capabilities
    cycle.require('graph_summary')
    with pytest.raises(StructuredContractError, match='aligned_observations'):
        cycle.require('conditional_information')


def test_derived_capabilities_when_declarations_missing():
    raw = example('xml_tree')
    del raw['topology']['capabilities']
    obj = StructuredObject.from_dict(raw)
    assert 'rooted_hierarchy' in obj.capabilities
    assert 'aligned_observations' in obj.capabilities
    assert 'explicit_probability_model' not in obj.capabilities


@pytest.mark.parametrize('mutation,message', [
    (lambda d: d['topology']['capabilities'].append('explicit_probability_model'), 'mismatch'),
    (lambda d: d['topology']['capabilities'].remove('acyclic'), 'mismatch'),
    (lambda d: d['topology']['capabilities'].append('xml_compatible'), 'unsupported capability'),
    (lambda d: d['topology']['capabilities'].append('acyclic'), 'duplicate'),
    (lambda d: d['topology'].__setitem__('kind', 'dag'), 'mismatch'),
    (lambda d: d['topology'].__setitem__('kind', 'undirected_graph'), 'mismatch'),
    (lambda d: d['topology'].__setitem__('kind', 'unsupported'), 'unsupported topology'),
])
def test_false_or_invalid_capability_claims_rejected(mutation, message):
    raw = example()
    mutation(raw)
    with pytest.raises(StructuredContractError, match=message):
        StructuredObject.from_dict(raw)


@pytest.mark.parametrize('mutation,message', [
    (lambda d: d['nodes'].append(deepcopy(d['nodes'][0])), 'duplicate node'),
    (lambda d: d['edges'].append(deepcopy(d['edges'][0])), 'duplicate edge'),
    (lambda d: d['edges'][0].__setitem__('target_id', 'nonexistent'), 'unknown target_id'),
    (lambda d: d['edges'][0].__setitem__('ordinal', -1), 'ordinal'),
    (lambda d: d['edges'][0].__setitem__('ordinal', True), 'ordinal'),
    (lambda d: d['edges'][0].__setitem__('directed', False), 'direction'),
    (lambda d: d['nodes'][0].__setitem__('id', ''), 'nonempty'),
    (lambda d: d['nodes'][0]['attributes'].__setitem__('not-json', {1, 2}), 'JSON-compatible'),
    (lambda d: d['nodes'][0].__setitem__('fictitious_field', 1), 'unsupported fields'),
])
def test_node_and_edge_integrity(mutation, message):
    raw = example()
    mutation(raw)
    with pytest.raises(StructuredContractError, match=message):
        StructuredObject.from_dict(raw)


def test_tree_rejects_multiple_parents_and_disconnected_roots():
    raw = example('shared_child_dag')
    raw['topology']['kind'] = 'tree'
    raw['topology'].pop('capabilities')
    with pytest.raises(StructuredContractError, match='unique parents'):
        StructuredObject.from_dict(raw)
    raw = example('xml_tree')
    raw['topology'].pop('capabilities')
    raw['edges'] = raw['edges'][1:]
    with pytest.raises(StructuredContractError, match='exactly one root'):
        StructuredObject.from_dict(raw)


def test_forest_and_undirected_graph_capabilities():
    raw = example('xml_tree')
    raw['topology'] = {'kind': 'forest'}
    raw['edges'] = raw['edges'][1:]
    forest = StructuredObject.from_dict(raw)
    assert 'rooted_hierarchy' in forest.capabilities
    assert 'unique_parent' in forest.capabilities
    forest.require('depth_reduction')
    raw = example('cyclic_network')
    raw['topology'] = {'kind': 'undirected_graph'}
    raw['edges'] = raw['edges'][:2]
    undirected = StructuredObject.from_dict(raw)
    assert 'acyclic' in undirected.capabilities
    assert 'directed_edges' not in undirected.capabilities
    raw['edges'].append({'id': 'parallel', 'kind': 'structural',
                         'source_id': 'alpha', 'target_id': 'beta'})
    assert 'acyclic' not in StructuredObject.from_dict(raw).capabilities


def test_dag_cycle_rejected_even_without_declared_capabilities():
    raw = example('cyclic_network')
    raw['topology'] = {'kind': 'dag'}
    with pytest.raises(StructuredContractError, match='cycles'):
        StructuredObject.from_dict(raw)


def test_ordering_is_not_invented():
    raw = example()
    raw['topology'].pop('capabilities')
    raw['edges'][0]['ordinal'] = None
    assert 'ordered_children' not in StructuredObject.from_dict(raw).capabilities
    raw = example()
    raw['topology'].pop('capabilities')
    raw['edges'][3]['ordinal'] = raw['edges'][2]['ordinal']
    assert 'ordered_children' not in StructuredObject.from_dict(raw).capabilities


@pytest.mark.parametrize('mutation,message', [
    (lambda d: d['observations'].append(deepcopy(d['observations'][0])), 'duplicate observation'),
    (lambda d: d['observations'][0]['values'].pop('value'), 'aligned variables'),
    (lambda d: d['observations'][0]['values']['name'].__setitem__('status', 'parse_error'), 'invalid cell state'),
    (lambda d: d['observations'][1]['values']['value'].__setitem__('value', 0), 'null state'),
    (lambda d: d['observations'][1]['values']['name'].__setitem__('value', None), 'empty state'),
    (lambda d: d['observations'][0]['values']['name'].__setitem__('value', None), 'present value'),
    (lambda d: d['observations'][0]['values']['name'].__setitem__('occurrences', []), 'multiple state'),
    (lambda d: d['observations'][0]['values']['name'].__setitem__('value', float('nan')), 'NaN'),
])
def test_observation_validation_fails_closed(mutation, message):
    raw = example()
    mutation(raw)
    with pytest.raises(StructuredContractError, match=message):
        StructuredObject.from_dict(raw)


def test_multiple_occurrences_preserved_and_checked():
    raw = example()
    raw['observations'][0]['values']['value'] = {
        'status': 'multiple', 'value': None,
        'occurrences': [
            {'status': 'present', 'value': 1},
            {'status': 'explicit_null', 'value': None, 'source_reason': 'explicit_nil'},
            {'status': 'present', 'value': 3},
        ],
    }
    obj = StructuredObject.from_dict(raw)
    assert [o['value'] for o in obj.to_dict()['observations'][0]['values']['value']['occurrences']] == [1, None, 3]
    assert StructuredObject.from_json(obj.to_json()).to_dict() == obj.to_dict()
    raw['observations'][0]['values']['value']['occurrences'][0]['status'] = 'multiple'
    with pytest.raises(StructuredContractError, match='nested multiple'):
        StructuredObject.from_dict(raw)


def test_require_aligned_rejects_missing_vars_and_empty_study():
    obj = StructuredObject.from_dict(example('xml_tree'))
    obj.require_aligned(['name', 'value'])
    with pytest.raises(StructuredContractError, match='aligned variables'):
        obj.require_aligned(['name', 'not_a_variable'])
    with pytest.raises(StructuredContractError, match='at least one'):
        StructuredObject.from_dict(example('cyclic_network')).require_aligned(['x'])


def test_strict_json_and_schema_validation():
    obj = StructuredObject.from_dict(example())
    with pytest.raises(StructuredContractError, match='duplicate JSON field'):
        StructuredObject.from_json('{"schema":"one","schema":"two"}')
    with pytest.raises(StructuredContractError, match='JSON constant'):
        StructuredObject.from_json('{"x":NaN}')
    with pytest.raises(StructuredContractError, match='structured JSON'):
        StructuredObject.from_json('{missing')
    raw = example()
    raw['fixture_id'] = 'do-not-leak-into-core-schema'
    with pytest.raises(StructuredContractError, match='unsupported fields'):
        StructuredObject.from_dict(raw)
    raw = example()
    raw['schema'] = 'ml-lab.structured-object@99'
    with pytest.raises(StructuredContractError, match='unsupported structured schema'):
        StructuredObject.from_dict(raw)
    assert obj.to_dict()['schema'] == CONTRACT_SCHEMA


def test_source_specific_names_only_reside_in_opaque_metadata():
    obj = StructuredObject.from_dict(example('xml_tree'))
    assert obj.observations[1]['values']['value']['source_reason'] == 'path_absent'
    assert obj.observations[2]['values']['value']['source_reason'] == 'explicit_xsi_nil'
    source = (Path(__file__).parents[1] / 'src/ml_lab/structured/object.py').read_text()
    for phrase in ('from ml_lab.data', 'from ml_lab.ui', 'source_type ==', 'xsi:nil', 'CodeAnalyzer', 'Product Factor'):
        assert phrase not in source
