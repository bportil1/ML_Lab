"""SU-05 scientific contracts: explicit source bindings, never inferred joins."""
from __future__ import annotations

import json

import pytest

from ml_lab.structured.bindings import (
    ObservationBindingPlan, VariableBinding, bind_observations, source_sha256,
)
from ml_lab.structured.contracts import StructuredContractError
from ml_lab.structured.object import StructuredObject


def source_fixture(name='xml_tree') -> StructuredObject:
    from pathlib import Path
    path = Path(__file__).parent / 'fixtures' / 'structured_uncertainty' / f'{name}.json'
    return StructuredObject.from_json(path.read_text())


def plan_for(obj, *entries, groups=(), weight=None):
    return ObservationBindingPlan(source_sha256(obj), obj.observation_unit,
                                  tuple(VariableBinding(*entry) for entry in entries),
                                  group_variables=groups, weight_variable=weight)


def test_tree_branch_membership_exact_ids_and_population_size():
    obj = source_fixture()
    plan = plan_for(obj, ('name', ('name',)), ('value', ('value',)))
    result = bind_observations(obj, plan)
    assert result.report['support']['aligned_rows'] == 3
    assert result.report['support']['independence_verified'] is False
    by_id = {row['node_id']: row for row in result.report['branches']}
    assert by_id['root']['subtree_variable_ids'] == ['name', 'value']
    assert by_id['item']['subtree_variable_ids'] == ['name', 'value']
    assert by_id['name']['direct_variable_ids'] == ['name']
    assert by_id['value']['direct_variable_ids'] == ['value']
    assert result.rows[1]['values']['value']['status'] == 'missing'
    assert result.rows[2]['values']['value']['source_reason'] == 'explicit_xsi_nil'
    assert obj.to_json() == source_fixture().to_json()
    json.dumps(result.to_dict(), allow_nan=False)


def test_plan_versioned_serialization_and_stale_source_rejection():
    obj = source_fixture()
    plan = plan_for(obj, ('name', ('name',)))
    assert ObservationBindingPlan.from_json(plan.to_json()) == plan
    assert bind_observations(obj, plan).report['plan'] == plan.to_dict()
    changed = obj.to_dict()
    changed.pop('object_id')
    changed['nodes'][0]['attributes']['revised'] = True
    with pytest.raises(StructuredContractError, match='fingerprint'):
        bind_observations(StructuredObject.from_dict(changed), plan)
    with pytest.raises(StructuredContractError, match='duplicate'):
        ObservationBindingPlan.from_json(plan.to_json().replace('"schema":', '"schema":"dupe","schema":', 1))


def test_unknown_nodes_and_observation_variables_fail_closed():
    obj = source_fixture()
    with pytest.raises(StructuredContractError, match='unknown source'):
        bind_observations(obj, plan_for(obj, ('name', ('not_here',))))
    with pytest.raises(StructuredContractError, match='unknown aligned'):
        bind_observations(obj, plan_for(obj, ('ghost', ('name',))))
    with pytest.raises(StructuredContractError, match='unit mismatch'):
        bind_observations(obj, ObservationBindingPlan(source_sha256(obj), 'node',
                                                     (VariableBinding('name', ('name',)),)))
    with pytest.raises(StructuredContractError, match='merged_sources'):
        VariableBinding('name', ('name', 'value'))


def test_explicit_many_to_one_mapping_does_not_allocate_occurrences():
    obj = source_fixture()
    plan = ObservationBindingPlan(source_sha256(obj), 'record',
            (VariableBinding('name', ('name', 'value'), merged_sources=True),))
    result = bind_observations(obj, plan)
    assert result.report['variables'][0]['merged_node_identity_ambiguous'] is True
    assert 'merged_source_nodes_have_no_occurrence_level_allocation' in result.report['limitations']
    assert result.report['support']['aligned_rows'] == 3


def repeated_source():
    obj = source_fixture()
    doc = obj.to_dict()
    doc.pop('object_id')
    doc['observations'][0]['values']['value'] = {
        'status': 'multiple', 'value': None,
        'occurrences': [
            {'status': 'present', 'value': 5, 'lineage': {'ordinal': 0}},
            {'status': 'present', 'value': 7, 'lineage': {'ordinal': 1}},
        ],
        'metadata': {'recorded_order': True},
    }
    return StructuredObject.from_dict(doc)


def test_repetition_preserved_as_one_record_not_two_samples():
    obj = repeated_source()
    result = bind_observations(obj, plan_for(obj, ('value', ('value',))))
    assert result.report['support']['aligned_rows'] == 3
    assert result.report['variables'][0]['repeated_rows'] == 1
    assert result.report['variables'][0]['recorded_occurrences_in_repeated_rows'] == 2
    assert result.rows[0]['values']['value']['status'] == 'multiple'
    assert [part['value'] for part in result.rows[0]['values']['value']['occurrences']] == [5, 7]
    assert 'lineage' in result.rows[0]['values']['value']['occurrences'][0]
    with pytest.raises(StructuredContractError, match='scalar_only'):
        bind_observations(obj, plan_for(obj, ('value', ('value',), 'scalar_only')))


def test_numeric_mean_is_explicit_and_keeps_source_intact():
    obj = repeated_source()
    plan = plan_for(obj, ('value', ('value',), 'mean_numeric'))
    result = bind_observations(obj, plan)
    assert result.rows[0]['values']['value']['value'] == 6
    assert result.rows[0]['values']['value']['lineage']['original_multiple_cell']['occurrences'][0]['lineage']['ordinal'] == 0
    assert result.rows[1]['values']['value']['status'] == 'missing'
    assert result.report['variables'][0]['repeated_rows'] == 1
    assert obj.observations[0]['values']['value']['status'] == 'multiple'
    assert result.to_json() == bind_observations(obj, plan).to_json()
    bad = obj.to_dict()
    bad.pop('object_id')
    bad['observations'][0]['values']['value']['occurrences'][1] = {
        'status': 'explicit_null', 'value': None, 'source_reason': 'upstream_nil'
    }
    bad = StructuredObject.from_dict(bad)
    with pytest.raises(StructuredContractError, match='all occurrences numeric'):
        bind_observations(bad, plan_for(bad, ('value', ('value',), 'mean_numeric')))


def test_grouping_and_weights_do_not_change_sample_counts():
    obj = source_fixture()
    doc = obj.to_dict()
    doc.pop('object_id')
    for row, group, weight in zip(doc['observations'], ['a', 'a', 'b'], [1, 2, 3]):
        row['values']['group'] = {'status': 'present', 'value': group}
        row['values']['weight'] = {'status': 'present', 'value': weight}
    weighted = StructuredObject.from_dict(doc)
    plan = plan_for(weighted, ('name', ('name',)), groups=('group',), weight='weight')
    result = bind_observations(weighted, plan)
    assert result.report['support']['group_count'] == 2
    assert sorted(group['row_count'] for group in result.report['support']['group_support']) == [1, 2]
    assert result.report['support']['sample_weight_total'] == 6
    assert result.report['support']['aligned_rows'] == 3
    assert result.rows[0]['group_key'][0]['value'] == 'a'
    doc = weighted.to_dict()
    doc.pop('object_id')
    doc['observations'][0]['values']['weight']['value'] = -1
    bad = StructuredObject.from_dict(doc)
    with pytest.raises(StructuredContractError, match='weight'):
        bind_observations(bad, plan_for(bad, ('name', ('name',)), weight='weight'))


@pytest.mark.parametrize('case', ['shared_child_dag', 'cyclic_network'])
def test_no_implicit_tree_on_nonhierarchical_graphs(case):
    obj = source_fixture(case)
    var = next(iter(obj.observations[0]['values'])) if obj.observations else 'x'
    node = obj.nodes[0]['id']
    result = bind_observations(obj, plan_for(obj, (var, (node,))))
    assert result.report['branches'] is None
    assert 'branch_descendants_not_defined_for_nonhierarchy' in result.report['limitations']
    assert result.report['support']['aligned_rows'] == len(obj.observations)
    if not obj.observations:
        assert 'no_empirical_observations_supplied' in result.report['limitations']


def test_no_observations_are_not_inferred_from_graph():
    obj = source_fixture('cyclic_network')
    result = bind_observations(obj, plan_for(obj, ('synthetic', (obj.nodes[0]['id'],))))
    assert result.report['support']['aligned_rows'] == 0
    assert result.rows == []


def test_immutable_detached_results():
    obj = source_fixture()
    result = bind_observations(obj, plan_for(obj, ('name', ('name',))))
    mutable = result.to_dict()
    mutable['rows'][0]['values']['name']['value'] = 'different'
    assert result.rows[0]['values']['name']['value'] == 'Alpha'


def test_bad_plan_types_fall_back_to_contract_error():
    obj = source_fixture()
    with pytest.raises(StructuredContractError, match='group variable'):
        ObservationBindingPlan(source_sha256(obj), 'record',
                               (VariableBinding('name', ('name',)),), group_variables=([],))
    with pytest.raises(StructuredContractError, match='occurrence'):
        VariableBinding('name', ('name',), occurrence_policy=[])
    with pytest.raises(StructuredContractError, match='node_ids'):
        VariableBinding('name', ('name', 'name'))
    with pytest.raises(StructuredContractError, match='variable bindings'):
        ObservationBindingPlan(source_sha256(obj), 'record',
                               (VariableBinding('name', ('name',)), VariableBinding('name', ('value',))))


def test_unavailable_group_retained_with_independent_null_exclusion_diagnostic():
    obj = source_fixture()
    raw = obj.to_dict()
    raw.pop('object_id')
    for idx, row in enumerate(raw['observations']):
        row['values']['group'] = ({'status': 'present', 'value': 'A'} if idx < 2 else
                                  {'status': 'missing', 'value': None, 'source_reason': 'path_absent'})
    obj = StructuredObject.from_dict(raw)
    report = bind_observations(obj, plan_for(obj, ('name', ('name',)), groups=('group',)))
    assert report.report['support']['aligned_rows'] == 3
    assert report.report['support']['group_count'] == 1
    assert report.report['support']['unavailable_group_rows'] == 1
    assert report.rows[2]['group_key'] is None


def test_numeric_aggregation_does_not_silently_coerce_categorical_values():
    obj = source_fixture()
    with pytest.raises(StructuredContractError, match='numeric'):
        bind_observations(obj, plan_for(obj, ('name', ('name',), 'mean_numeric')))
    raw = repeated_source().to_dict()
    raw.pop('object_id')
    raw['observations'][0]['values']['value']['occurrences'][0]['value'] = 10**1000
    huge = StructuredObject.from_dict(raw)
    with pytest.raises(StructuredContractError, match='overflow'):
        bind_observations(huge, plan_for(huge, ('value', ('value',), 'mean_numeric')))


def test_branch_aggregation_deep_chain_and_empty_observation_support():
    nodes = [{'id': f'n{i}', 'kind': 'generic', 'attributes': {}} for i in range(450)]
    edges = [{'id': f'e{i}', 'kind': 'child', 'source_id': f'n{i}', 'target_id': f'n{i+1}'}
             for i in range(449)]
    obj = StructuredObject.from_dict({
        'schema': 'ml-lab.structured-object@1', 'topology': {'kind': 'tree'},
        'nodes': nodes, 'edges': edges, 'observation_unit': 'source',
        'observations': [], 'provenance': {'fixture': 'deep'}})
    result = bind_observations(obj, plan_for(obj, ('observation', ('n449',))))
    assert result.report['support']['aligned_rows'] == 0
    assert result.report['branches'][0]['subtree_variable_ids'] == ['observation']


def test_nonhierarchical_node_graph_does_not_generate_descendant_scopes():
    obj = source_fixture('shared_child_dag')
    if obj.observations:
        variable = next(iter(obj.observations[0]['values']))
    else:
        variable = 'unobserved'
    result = bind_observations(obj, plan_for(obj, (variable, (obj.nodes[0]['id'],))))
    assert result.report['branches'] is None
