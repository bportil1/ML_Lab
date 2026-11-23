"""ML-UW-27 pre-implementation contract fixtures and failure gates.

This is a fixture-level audit, not a StructuredObject runtime implementation.
ML-UW-28 will implement object validation and serialization using these cases.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml_lab.structured import (
    CONTRACT_SCHEMA, CAPABILITIES, CELL_STATES, NULL_REASONS, TOPOLOGIES,
    StructuredContractError, require_capabilities, require_observation_alignment,
)

FIXTURES = Path(__file__).parent / 'fixtures' / 'structured_uncertainty'
CASES = ['xml_tree', 'ast_tree', 'shared_child_dag', 'cyclic_network']


def load_case(name: str) -> dict:
    return json.loads((FIXTURES / f'{name}.json').read_text())


def _cycle_exists(nodes: list[str], edges: list[dict]) -> bool:
    remaining = {node: 0 for node in nodes}
    children = {node: [] for node in nodes}
    for edge in edges:
        remaining[edge['target_id']] += 1
        children[edge['source_id']].append(edge['target_id'])
    ready = [node for node, indegree in remaining.items() if indegree == 0]
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for child in children[node]:
            remaining[child] -= 1
            if remaining[child] == 0:
                ready.append(child)
    return visited != len(nodes)


@pytest.mark.parametrize('case', CASES)
def test_fixture_ids_and_topology_contract(case):
    fixture = load_case(case)
    assert fixture['schema'] == CONTRACT_SCHEMA
    assert fixture['topology']['kind'] in TOPOLOGIES
    declared = set(fixture['topology']['capabilities'])
    assert declared <= CAPABILITIES
    ids = [node['id'] for node in fixture['nodes']]
    assert len(ids) == len(set(ids)) and all(ids)
    edge_ids = [edge['id'] for edge in fixture['edges']]
    assert len(edge_ids) == len(set(edge_ids))
    assert all(edge['source_id'] in ids and edge['target_id'] in ids
               for edge in fixture['edges'])
    parents = {node: 0 for node in ids}
    for edge in fixture['edges']:
        parents[edge['target_id']] += 1
    cycle = _cycle_exists(ids, fixture['edges'])
    if 'acyclic' in declared:
        assert not cycle
    if 'unique_parent' in declared:
        assert max(parents.values()) <= 1
    if 'rooted_hierarchy' in declared:
        roots = sum(count == 0 for count in parents.values())
        assert roots == (1 if fixture['topology']['kind'] == 'tree' else roots)
        assert roots >= 1
    if fixture['topology']['kind'] == 'dag':
        assert not cycle
    if fixture['topology']['kind'] == 'tree':
        assert len(fixture['edges']) == len(ids) - 1


def test_fixture_encodes_multi_parent_without_tree_coercion():
    graph = load_case('shared_child_dag')
    assert len([e for e in graph['edges'] if e['target_id'] == 'shared']) == 2
    with pytest.raises(StructuredContractError, match='rooted_hierarchy'):
        require_capabilities(graph['topology']['capabilities'], 'depth_reduction')


def test_cycle_is_rejected_by_probability_dag_gate():
    graph = load_case('cyclic_network')
    assert _cycle_exists([n['id'] for n in graph['nodes']], graph['edges'])
    require_capabilities(graph['topology']['capabilities'], 'graph_summary')
    with pytest.raises(StructuredContractError, match='acyclic'):
        require_capabilities({*graph['topology']['capabilities'],
                              'aligned_observations', 'explicit_probability_model'},
                             'hierarchical_probability_inference')


def test_source_structure_is_not_a_probability_model():
    tree = load_case('xml_tree')
    require_capabilities(tree['topology']['capabilities'], 'hierarchy_summary')
    with pytest.raises(StructuredContractError, match='explicit_probability_model'):
        require_capabilities(tree['topology']['capabilities'], 'hierarchical_probability_inference')


@pytest.mark.parametrize('case,variables', [
    ('xml_tree', ['name', 'value']),
    ('ast_tree', ['expression', 'identifier']),
    ('shared_child_dag', ['upstream_a', 'shared']),
])
def test_explicit_observation_alignment(case, variables):
    fixture = load_case(case)
    require_observation_alignment(fixture['observation_unit'], fixture['observations'], variables)
    for row in fixture['observations']:
        for value in row['values'].values():
            assert value['status'] in CELL_STATES
            assert (value['value'] is None) == (value['status'] in NULL_REASONS)


def test_unpaired_or_duplicated_observations_are_rejected():
    case = load_case('xml_tree')
    rows = case['observations']
    with pytest.raises(StructuredContractError, match='aligned'):
        require_observation_alignment('record', [{**rows[0], 'values': {'name': rows[0]['values']['name']}}],
                                      ['name', 'value'])
    with pytest.raises(StructuredContractError, match='unique'):
        require_observation_alignment('record', [rows[0], rows[0]], ['name'])
    with pytest.raises(StructuredContractError, match='observation unit'):
        require_observation_alignment('unknown', rows, ['name'])
    with pytest.raises(StructuredContractError, match='at least one'):
        require_observation_alignment('record', [], ['name'])


def test_null_is_not_empty_or_parse_failure():
    case = load_case('xml_tree')
    rows = case['observations']
    assert rows[1]['values']['name'] == {'status': 'empty', 'value': ''}
    assert rows[1]['values']['value']['source_reason'] == 'path_absent'
    assert rows[2]['values']['value']['source_reason'] == 'explicit_xsi_nil'
    with pytest.raises(StructuredContractError, match='null reason'):
        require_observation_alignment('record', [
            {'observation_id': 'bad', 'values': {'x': {'status': 'missing', 'value': 0}}}], ['x'])
    with pytest.raises(StructuredContractError, match='invalid state'):
        require_observation_alignment('record', [
            {'observation_id': 'bad', 'values': {'x': {'status': 'parse_error', 'value': None}}}], ['x'])


def test_unknown_operation_and_capability_fail_closed():
    with pytest.raises(StructuredContractError, match='unknown operation'):
        require_capabilities(['graph_edges'], 'magical_operation')
    with pytest.raises(StructuredContractError, match='unsupported capabilities'):
        require_capabilities(['graph_edges', 'xml_element'], 'graph_summary')
    with pytest.raises(StructuredContractError, match='collection'):
        require_capabilities('graph_edges', 'graph_summary')


def test_core_contracts_do_not_import_source_adapters():
    code = (Path(__file__).parents[1] / 'src/ml_lab/structured/contracts.py').read_text()
    assert 'from ml_lab.data' not in code
    assert 'from ml_lab.ui' not in code
    assert 'source_type ==' not in code
