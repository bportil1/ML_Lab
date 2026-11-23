"""ML-UW-30: domain-neutral full/partial/flat projections with explicit loss."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
from copy import deepcopy

import pytest

from ml_lab.structured import (
    StructuredContractError, StructuredObject, TransformationPlan, transform,
)
from ml_lab.structured.adapters import JSONAdapter, GraphAdapter

FIXTURES = Path(__file__).parent / 'fixtures' / 'structured_uncertainty'


def fixture(name='xml_tree'):
    return StructuredObject.from_dict(json.loads((FIXTURES / f'{name}.json').read_text()))


def test_full_view_is_exact_identity_and_round_trips():
    original = fixture()
    result = transform(original, TransformationPlan('full'))
    assert result.object is original
    assert result.object.to_json() == original.to_json()
    assert result.lossless and result.reversible
    assert result.audit['losses'] == []
    assert result.audit['source_object_id'] == original.object_id
    assert result.audit['input_observation_ids'] == result.audit['output_observation_ids']
    assert TransformationPlan.from_dict(result.plan.to_dict()) == result.plan
    assert TransformationPlan.from_dict(json.loads(result.plan.to_json())) == result.plan
    with pytest.raises((FrozenInstanceError, AttributeError)):
        result.plan.mode = 'flat'


def test_retain_subtree_maintains_observations_and_unique_parent():
    original = fixture()
    result = transform(original, TransformationPlan('subtree', roots=('item',)))
    assert {node['id'] for node in result.object.nodes} == {'item', 'name', 'value'}
    assert {edge['id'] for edge in result.object.edges} == {'e3', 'e4'}
    assert result.object.topology['kind'] == 'tree'
    result.object.require('hierarchy_summary')
    assert result.audit['removed_node_ids'] == ['root', 'section']
    assert result.audit['removed_edge_ids'] == ['e1', 'e2']
    assert result.audit['node_mapping']['root'] is None
    assert result.audit['node_mapping']['item'] == 'item'
    assert result.audit['observation_unit'] == 'record'
    assert result.object.observations == original.observations
    assert not result.lossless and not result.reversible
    assert 'edge_relationships' in result.audit['losses']
    # Caller data and previous object were never mutated.
    assert len(original.nodes) == 5 and len(original.edges) == 4


def test_subtree_selection_with_overlapping_roots_deduplicates():
    original = fixture()
    result = transform(original, TransformationPlan('subtree', roots=('section', 'item')))
    assert {n['id'] for n in result.object.nodes} == {'section', 'item', 'name', 'value'}
    assert result.object.topology['kind'] == 'tree'
    assert len(result.object.edges) == 3


def test_multiple_selected_roots_create_forest_not_false_single_tree():
    original = fixture()
    result = transform(original, TransformationPlan('subtree', roots=('name', 'value')))
    assert result.object.topology['kind'] == 'forest'
    assert not result.object.edges
    assert {n['id'] for n in result.object.nodes} == {'name', 'value'}
    assert result.audit['retained_node_ids'] == ['name', 'value']


def test_depth_reduction_preserves_boundary_and_ids():
    original = fixture()
    result = transform(original, TransformationPlan('depth', max_depth=2))
    assert {n['id'] for n in result.object.nodes} == {'root', 'section', 'item'}
    assert {e['id'] for e in result.object.edges} == {'e1', 'e2'}
    assert {n['id'] for n in original.nodes} == {'root', 'section', 'item', 'name', 'value'}
    assert result.audit['removed_node_ids'] == ['name', 'value']
    assert result.object.observations == original.observations


def test_aggregation_counts_only_removed_structure_and_no_values():
    original = fixture()
    plan = TransformationPlan('aggregate', max_depth=1)
    result = transform(original, plan)
    assert {n['id'] for n in result.object.nodes} == {'root', 'section'}
    boundary = next(n for n in result.object.nodes if n['id'] == 'section')
    summary = boundary['metadata']['ml_lab_structural_aggregate_v1']
    assert summary['removed_nodes'] == 3
    assert summary['removed_edges'] == 3
    assert summary['node_kinds'] == {'uninterpreted': 3}
    assert summary['value_aggregation'] == 'none'
    assert result.audit['collapsed_into'] == {
        'item': 'section', 'name': 'section', 'value': 'section',
    }
    assert not result.lossless and not result.reversible
    assert result.object.observations == original.observations
    assert not any('ml_lab_structural_aggregate_v1' in n.get('metadata', {}) for n in original.nodes)
    assert transform(original, plan).object.to_json() == result.object.to_json()
    assert transform(original, plan).audit == result.audit


def test_aggregation_collisions_rejected_not_overwritten():
    raw = fixture().to_dict()
    raw['nodes'][1]['metadata'] = {'ml_lab_structural_aggregate_v1': {'owner': 'custom'}}
    source = StructuredObject.from_dict(raw)
    with pytest.raises(StructuredContractError, match='collision'):
        transform(source, TransformationPlan('aggregate', max_depth=1))


def test_flat_view_returns_two_independent_tables_no_joins():
    original = fixture()
    result = transform(original, TransformationPlan('flat'))
    assert result.object.topology['kind'] == 'directed_graph'
    assert 'rooted_hierarchy' not in result.object.capabilities
    with pytest.raises(StructuredContractError, match='rooted_hierarchy'):
        result.object.require('depth_reduction')
    assert len(result.object.nodes) == len(original.nodes)
    assert not result.object.edges
    tables = result.flat_tables()
    assert len(tables['nodes']) == 5
    assert len(tables['observations']) == 3  # not 5 * 3
    assert tables['observations'][1]['values']['value']['source_reason'] == 'path_absent'
    assert tables['observations'][2]['values']['value']['status'] == 'explicit_null'
    assert result.audit['observation_join_performed'] is False
    assert result.audit['removed_node_ids'] == []
    assert len(result.audit['removed_edge_ids']) == 4
    assert not result.lossless and not result.reversible
    with pytest.raises(StructuredContractError, match='flat structural views'):
        transform(original, TransformationPlan('full')).flat_tables()


@pytest.mark.parametrize('name', ['shared_child_dag', 'cyclic_network'])
def test_non_tree_graphs_may_be_flattened_but_not_hierarchically_reduced(name):
    source = fixture(name)
    flat = transform(source, TransformationPlan('flat'))
    assert flat.object.topology['kind'] == 'directed_graph'
    assert len(flat.object.nodes) == len(source.nodes)
    assert flat.audit['removed_node_ids'] == []
    for plan in (TransformationPlan('depth', max_depth=1),
                 TransformationPlan('aggregate', max_depth=0),
                 TransformationPlan('subtree', roots=(source.nodes[0]['id'],))):
        with pytest.raises(StructuredContractError, match='rooted_hierarchy'):
            transform(source, plan)
    # Original multi-parent edges/cycles are not edited by a failed operation.
    assert source == fixture(name)


def test_generic_graph_adapter_works_without_source_semantics():
    result = GraphAdapter().adapt({
        'schema': 'ml-lab.node-edge@1', 'topology': 'forest',
        'nodes': [
            {'id': 'a', 'kind': 'x', 'attributes': {}},
            {'id': 'b', 'kind': 'y', 'attributes': {}},
        ], 'edges': [],
    })
    assert transform(result.object, TransformationPlan('depth', max_depth=0)).lossless


def test_json_adapter_depth_and_subtree_are_domain_neutral():
    obj = JSONAdapter().adapt({'group': [{'value': 1}, {'value': None}]}).object
    assert obj.topology['kind'] == 'tree'
    result = transform(obj, TransformationPlan('subtree', roots=('#/group/0',)))
    assert {n['id'] for n in result.object.nodes} == {'#/group/0', '#/group/0/value'}
    assert len(result.object.edges) == 1
    assert len(result.object.observations) == 0  # no fabricated record samples
    assert transform(obj, TransformationPlan('depth', max_depth=1)).object.topology['kind'] == 'tree'


def test_explicit_observation_projection_never_joins_or_reindexes():
    original = fixture()
    result = transform(original, TransformationPlan('full', variables=('value',)))
    assert result.object is not original
    assert result.audit['removed_variables'] == ['name']
    assert result.audit['retained_variables'] == ['value']
    assert [o['observation_id'] for o in result.object.observations] == ['r1', 'r2', 'r3']
    for row in result.object.observations:
        assert list(row['values']) == ['value']
    assert result.object.observations[1]['values']['value']['source_reason'] == 'path_absent'
    assert not result.lossless and not result.reversible
    result.object.require_aligned(['value'])
    with pytest.raises(StructuredContractError, match='aligned variables'):
        result.object.require_aligned(['name'])


def test_complete_variable_projection_never_pretends_to_be_identity():
    original = fixture()
    result = transform(original, TransformationPlan('full', variables=('value', 'name')))
    assert result.object is not original
    assert list(result.object.observations[0]['values']) == ['value', 'name']
    assert result.audit['removed_variables'] == []
    assert not result.reversible
    assert result.audit['output_object_id'] == result.object.object_id


def test_no_projection_of_unaligned_or_missing_variables():
    for source in (fixture('cyclic_network'), JSONAdapter().adapt([1, 2]).object):
        with pytest.raises(StructuredContractError, match='aligned observations'):
            transform(source, TransformationPlan('full', variables=('x',)))
    with pytest.raises(StructuredContractError, match='unknown aligned'):
        transform(fixture(), TransformationPlan('full', variables=('not-in-rows',)))


@pytest.mark.parametrize('kwargs,fragment', [
    ({'mode': 'invalid'}, 'unsupported'),
    ({'mode': 'full', 'roots': ('root',)}, 'only supported'),
    ({'mode': 'subtree'}, 'requires'),
    ({'mode': 'subtree', 'roots': ('root', 'root')}, 'unique'),
    ({'mode': 'flat', 'max_depth': 2}, 'only supported'),
    ({'mode': 'depth', 'max_depth': -1}, 'nonnegative'),
    ({'mode': 'depth', 'max_depth': True}, 'nonnegative'),
    ({'mode': 'depth'}, 'nonnegative'),
    ({'mode': 'full', 'variables': ()}, 'nonempty'),
    ({'mode': 'full', 'variables': ('v', 'v')}, 'unique'),
])
def test_invalid_plan_rejected(kwargs, fragment):
    with pytest.raises(StructuredContractError, match=fragment):
        TransformationPlan(**kwargs)


def test_plan_unknown_fields_and_unknown_nodes_fail_closed():
    plan = TransformationPlan('subtree', roots=('item',))
    raw = plan.to_dict()
    raw['unsupported'] = True
    with pytest.raises(StructuredContractError, match='unsupported'):
        TransformationPlan.from_dict(raw)
    with pytest.raises(StructuredContractError, match='schema'):
        TransformationPlan.from_dict({'mode': 'full'})
    with pytest.raises(StructuredContractError, match='unknown subtree'):
        transform(fixture(), TransformationPlan('subtree', roots=('bad',)))
    with pytest.raises(StructuredContractError, match='validated'):
        transform(fixture().to_dict(), TransformationPlan('full'))


def test_report_detached_and_replay_deterministic():
    obj = fixture()
    plan = TransformationPlan('flat')
    a, b = transform(obj, plan), transform(obj, TransformationPlan.from_dict(plan.to_dict()))
    assert a.audit == b.audit and a.object.to_json() == b.object.to_json()
    report = a.audit
    report['removed_edge_ids'].clear()
    assert len(a.audit['removed_edge_ids']) == 4
    data = a.flat_tables()
    data['nodes'][0]['attributes']['new'] = 'local edit'
    assert 'new' not in a.object.nodes[0]['attributes']
    assert a.audit['source_data_verified'] is False


def test_reduced_object_may_be_reduced_again_with_lineage_chain():
    obj = fixture()
    partial = transform(obj, TransformationPlan('depth', max_depth=2)).object
    followup = transform(partial, TransformationPlan('flat'))
    history = followup.object.provenance['structural_transformations']
    assert len(history) == 2
    assert history[0]['source_object_id'] == obj.object_id
    assert history[1]['source_object_id'] == partial.object_id
    assert followup.audit['source_object_id'] == partial.object_id


def test_multiple_occurrences_and_null_reasons_survive_all_structural_modes():
    raw = fixture().to_dict()
    raw['observations'][0]['values']['value'] = {
        'status': 'multiple', 'value': None,
        'occurrences': [
            {'status': 'present', 'value': 17},
            {'status': 'explicit_null', 'value': None,
             'source_reason': 'provided_null'},
        ],
    }
    original = StructuredObject.from_dict(raw)
    for plan in (TransformationPlan('full'), TransformationPlan('flat'),
                 TransformationPlan('subtree', roots=('value',)),
                 TransformationPlan('depth', max_depth=1),
                 TransformationPlan('aggregate', max_depth=1)):
        result = transform(original, plan)
        cell = result.object.observations[0]['values']['value']
        assert cell['status'] == 'multiple'
        assert len(cell['occurrences']) == 2
        assert cell['occurrences'][1]['source_reason'] == 'provided_null'
        assert result.object.observation_unit == 'record'
        assert len(result.object.observations) == 3
