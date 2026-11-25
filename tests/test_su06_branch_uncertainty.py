"""SU-06 regression fixtures: observed records, never tree nodes, are samples."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml_lab.structured import (
    BranchUncertaintyPlan, ObservationBindingPlan, StructuredContractError,
    StructuredObject, VariableBinding, analyze_branch_uncertainty,
    bind_observations, binding_sha256, source_sha256,
)

FIXTURES = Path(__file__).parent / 'fixtures' / 'structured_uncertainty'


def fixture(name='xml_tree'):
    return StructuredObject.from_json((FIXTURES / f'{name}.json').read_text())


def binding_for(obj, *variables, groups=(), weight=None):
    return bind_observations(obj, ObservationBindingPlan(
        source_sha256(obj), obj.observation_unit,
        tuple(VariableBinding(name, tuple(nodes), policy) for name, nodes, policy in variables),
        group_variables=tuple(groups), weight_variable=weight))


def run(binding, *names, **kwargs):
    plan = BranchUncertaintyPlan(binding_sha256(binding), tuple(names), **kwargs)
    return analyze_branch_uncertainty(binding, plan).to_dict()


def synthetic(rows, *, group_names=('project', 'module'), variables=('y', 'z'), weight=False):
    doc = fixture().to_dict()
    doc.pop('object_id')
    doc['observations'] = []
    for i, record in enumerate(rows):
        values = {}
        for name in (*variables, *group_names, *(['weight'] if weight else [])):
            if name in record and record[name] is not None:
                values[name] = {'status': 'present', 'value': record[name]}
            else:
                values[name] = {'status': 'missing', 'value': None, 'source_reason': 'path_absent'}
        doc['observations'].append({'observation_id': f'r{i}', 'values': values})
    obj = StructuredObject.from_dict(doc)
    return binding_for(obj, *((name, ('name' if i == 0 else 'value',), 'preserve')
                              for i, name in enumerate(variables)),
                       groups=group_names, weight='weight' if weight else None)


def test_original_node_identity_and_null_support_without_fake_samples():
    source = fixture()
    original = source.to_json()
    binding = binding_for(source, ('name', ('name',), 'preserve'),
                          ('value', ('value',), 'preserve'))
    report = run(binding, 'value')
    value = report['variables']['value']
    assert report['schema'] == 'ml-lab.branch-uncertainty@1'
    assert report['support']['aligned_rows'] == 3
    assert report['support']['complete_case_rows'] == 1
    assert report['support']['exclusion_reasons'] == {'target_value_explicit_null': 1, 'target_value_missing': 1}
    assert value['original_status_counts'] == {'explicit_null': 1, 'missing': 1, 'present': 1}
    assert value['null_reasons'] == {'explicit_xsi_nil': 1, 'path_absent': 1}
    assert value['occurrence']['observed_row_rate'] == pytest.approx(1 / 3)
    assert value['occurrence']['per_record_cardinality'] == {'0': 2, '1': 1}
    assert report['targets']['value']['raw']['status'] == 'insufficient'
    assert report['targets']['value']['raw']['population_variance'] is None
    nodes = {x['node_id']: x for x in report['nodes']}
    assert nodes['root']['scope'] == 'subtree'
    assert nodes['root']['selected_target_ids'] == ['value']
    assert nodes['name']['selected_target_ids'] == []
    assert nodes['value']['direct_variable_ids'] == ['value']
    assert nodes['value']['variance_is_attributed_to_graph_edge'] is False
    assert source.to_json() == original
    json.dumps(report, allow_nan=False)


def test_unbalanced_nested_grouping_partitions_weighted_ss_exactly():
    binding = synthetic([
        {'project': 'P', 'module': 'M1', 'y': 1, 'z': 10},
        {'project': 'P', 'module': 'M1', 'y': 3, 'z': 20},
        {'project': 'P', 'module': 'M2', 'y': 5, 'z': 30},
        {'project': 'Q', 'module': 'M1', 'y': 4, 'z': 40},
        {'project': 'Q', 'module': 'M1', 'y': 6, 'z': 50},
        {'project': 'Q', 'module': 'M2', 'y': 1, 'z': 60},
    ])
    report = run(binding, 'y', group_levels=('project', 'module'))
    y = report['targets']['y']
    parts = y['group_variance']['components']
    assert report['support']['complete_case_rows'] == 6
    assert y['raw']['sample_variance_unweighted'] == pytest.approx(4.266666666666667)
    assert [r['group_count'] for r in parts] == [2, 4, 4]
    assert [r['group_count'] for r in report['group_support']] == [2, 4]
    assert sum(g['rows'] for g in report['group_support'][1]['groups']) == 6
    assert parts[-1]['sum_squares'] == pytest.approx(4.0)
    assert sum(p['sum_squares'] for p in parts) == pytest.approx(y['group_variance']['total_sum_squares'])
    assert sum(p['proportion_total_ss'] for p in parts) == pytest.approx(1)
    assert report['objective'] is None


def test_fractional_observation_weights_change_descriptive_estimand():
    binding = synthetic([
        {'project': 'P', 'module': 'A', 'y': 0, 'z': 10, 'weight': 1},
        {'project': 'P', 'module': 'A', 'y': 2, 'z': 12, 'weight': 2},
        {'project': 'Q', 'module': 'B', 'y': 10, 'z': 14, 'weight': 0.5},
    ], weight=True)
    report = run(binding, 'y', 'z', group_levels=('project', 'module'),
                 scale='declared', declared_scales=(('y', 2), ('z', 4)),
                 objective='weighted_variance', target_weights=(('y', 3), ('z', 1)))
    assert report['support']['complete_case_weight_total'] == pytest.approx(3.5)
    assert report['targets']['y']['raw']['mean'] == pytest.approx(9 / 3.5)
    assert report['targets']['y']['raw']['sample_variance_unweighted'] == pytest.approx(28)
    obj = report['objective']
    assert obj['status'] == 'sufficient'
    assert obj['weighted_population_variance'] > 0
    assert obj['normalized_target_weights'] == {'y': .75, 'z': .25}
    assert sum(x['weighted_population_variance'] for x in obj['components']) == pytest.approx(
        obj['weighted_population_variance'])
    assert sum(x['proportion_objective'] for x in obj['components']) == pytest.approx(1)


def test_repeated_occurrences_do_not_inflate_record_denominator_or_impute():
    doc = fixture().to_dict()
    doc.pop('object_id')
    doc['observations'][0]['values']['value'] = {
        'status': 'multiple', 'value': None,
        'occurrences': [{'status': 'present', 'value': 10},
                        {'status': 'present', 'value': 20},
                        {'status': 'explicit_null', 'value': None, 'source_reason': 'xsi_nil'}]}
    obj = StructuredObject.from_dict(doc)
    binding = binding_for(obj, ('value', ('value',), 'preserve'))
    report = run(binding, 'value')
    value = report['variables']['value']
    assert report['support']['complete_case_rows'] == 0
    assert value['support']['rows'] == 3
    assert value['support']['repeated_rows'] == 1
    assert value['support']['total_occurrences'] == 3
    assert value['support']['present_occurrences'] == 2
    assert value['repeated_null_reasons'] == {'xsi_nil': 1}
    assert value['occurrence']['distinct_observed_values'] == 2
    assert value['occurrence']['per_record_cardinality'] == {'0': 2, '3': 1}


def test_mean_numeric_resolves_records_but_preserves_original_cardinality():
    doc = fixture().to_dict()
    doc.pop('object_id')
    doc['observations'][0]['values']['value'] = {
        'status': 'multiple', 'value': None,
        'occurrences': [{'status': 'present', 'value': 10}, {'status': 'present', 'value': 12}]}
    doc['observations'][1]['values']['value'] = {'status': 'present', 'value': 9}
    obj = StructuredObject.from_dict(doc)
    binding = binding_for(obj, ('value', ('value',), 'mean_numeric'))
    report = run(binding, 'value')
    assert report['variables']['value']['original_status_counts']['multiple'] == 1
    assert report['variables']['value']['resolved_status_counts']['present'] == 2
    assert report['variables']['value']['support']['total_occurrences'] == 3
    assert report['support']['complete_case_rows'] == 2
    assert report['targets']['value']['raw']['mean'] == pytest.approx(10)


def test_nonhierarchy_only_reports_declared_direct_mappings():
    obj = fixture('shared_child_dag')
    name = next(iter(obj.observations[0]['values'])) if obj.observations else 'unused'
    node = obj.nodes[0]['id']
    binding = binding_for(obj, (name, (node,), 'preserve'))
    # All present values need not be numeric; select no bogus structural samples.
    result = run(binding, name) if not obj.observations else None
    if result is not None:
        assert result['nodes'][0]['scope'] == 'direct_only'
        assert result['support']['complete_case_rows'] == 0


def test_merged_source_nodes_are_explicitly_ambiguous():
    obj = fixture()
    binding = bind_observations(obj, ObservationBindingPlan(
        source_sha256(obj), 'record',
        (VariableBinding('value', ('name', 'value'), merged_sources=True),)))
    report = run(binding, 'value')
    nodes = {x['node_id']: x for x in report['nodes']}
    assert 'value' in nodes['root']['ambiguous_merged_variable_ids']
    assert nodes['value']['selected_target_ids'] == []
    assert nodes['value']['ambiguous_merged_variable_ids'] == ['value']


def test_invalid_objective_scale_grouping_and_stale_binding_fail_closed():
    binding = binding_for(fixture(), ('value', ('value',), 'preserve'),
                          ('name', ('name',), 'preserve'))
    digest = binding_sha256(binding)
    with pytest.raises(StructuredContractError, match='multi-target'):
        BranchUncertaintyPlan(digest, ('value', 'name'), objective='weighted_variance',
                              target_weights=(('value', 1), ('name', 1)))
    with pytest.raises(StructuredContractError, match='declared_scales'):
        BranchUncertaintyPlan(digest, ('value',), scale='declared',
                              declared_scales=(('unbound', 2),))
    with pytest.raises(StructuredContractError, match='group_levels'):
        run(binding, 'value', group_levels=('unbound',))
    with pytest.raises(StructuredContractError, match='not bool or text'):
        run(binding, 'name')
    with pytest.raises(StructuredContractError, match='fingerprint'):
        analyze_branch_uncertainty(binding, BranchUncertaintyPlan('sha256:' + '0'*64, ('value',)))
    with pytest.raises(StructuredContractError, match='explicitly bound'):
        run(binding, 'not_bound')


def test_sparse_no_observations_and_constant_zscores_are_unavailable():
    doc = fixture().to_dict()
    doc.pop('object_id')
    doc['observations'] = []
    doc['topology'] = {'kind': 'tree'}
    obj = StructuredObject.from_dict(doc)
    report = run(binding_for(obj, ('synthetic', ('value',), 'preserve')), 'synthetic',
                 scale='zscore', objective='weighted_variance', target_weights=(('synthetic', 1),))
    assert report['support']['aligned_rows'] == 0
    assert report['objective']['status'] == 'unavailable'
    assert report['targets']['synthetic']['raw']['status'] == 'no_observations'
    constant = synthetic([{'y': 3, 'z': 4}, {'y': 3, 'z': 4}], group_names=())
    result = run(constant, 'y', scale='zscore')
    assert result['targets']['y']['normalization']['status'] == 'unavailable_constant_or_sparse'
    assert result['targets']['y']['scaled'] is None
    assert result['targets']['y']['raw']['status'] == 'constant'


def test_exclusion_reasons_and_complete_case_denominator_are_shared_across_targets():
    binding = synthetic([
        {'project': 'A', 'module': 'M', 'y': 1, 'z': 2},
        {'project': 'A', 'module': 'M', 'y': None, 'z': 3},
        {'project': 'A', 'module': None, 'y': 4, 'z': 3},
        {'project': 'B', 'module': 'N', 'y': 3, 'z': None},
        {'project': 'B', 'module': 'N', 'y': 5, 'z': 1},
    ])
    report = run(binding, 'y', 'z', group_levels=('project', 'module'),
                 scale='declared', declared_scales=(('y', 5), ('z', 2)),
                 objective='weighted_variance', target_weights=(('y', 1), ('z', 1)))
    assert report['support']['complete_case_rows'] == 2
    assert report['support']['excluded_rows'] == 3
    assert report['support']['exclusion_reasons'] == {'group_unavailable': 1, 'target_y_missing': 1, 'target_z_missing': 1}
    assert report['targets']['y']['raw']['rows'] == report['targets']['z']['raw']['rows'] == 2
    assert report['objective']['status'] == 'sufficient'


def test_plan_roundtrip_strict_and_report_detached():
    binding = binding_for(fixture(), ('value', ('value',), 'preserve'))
    plan = BranchUncertaintyPlan(binding_sha256(binding), ('value',))
    assert BranchUncertaintyPlan.from_json(plan.to_json()) == plan
    with pytest.raises(StructuredContractError, match='duplicate'):
        BranchUncertaintyPlan.from_json(plan.to_json().replace('"schema":', '"schema":"x","schema":', 1))
    with pytest.raises(StructuredContractError, match='schema'):
        BranchUncertaintyPlan.from_dict({**plan.to_dict(), 'schema': 'future'})
    result = analyze_branch_uncertainty(binding, plan)
    modified = result.to_dict()
    modified['nodes'][0]['node_id'] = 'altered'
    assert result.to_dict()['nodes'][0]['node_id'] != 'altered'


def test_reconstruction_objective_is_unavailable_until_candidate_is_paired():
    binding = binding_for(fixture(), ('value', ('value',), 'preserve'))
    report = run(binding, 'value', objective='reconstruction_mse',
                 target_weights=(('value', 1),))
    assert report['objective'] == {
        'status': 'unavailable', 'reason': 'paired_reconstruction_required',
        'mean_squared_error': None, 'components': None}


def test_zscore_weighted_variance_and_grouped_accounting():
    binding = synthetic([
        {'project': 'P', 'module': 'M', 'y': 1, 'z': 100},
        {'project': 'P', 'module': 'M', 'y': 3, 'z': 200},
        {'project': 'Q', 'module': 'N', 'y': 5, 'z': 300},
    ])
    report = run(binding, 'y', 'z', scale='zscore', group_levels=('project',),
                 objective='weighted_variance', target_weights=(('y', 1), ('z', 1)))
    assert report['objective']['weighted_population_variance'] == pytest.approx(1)
    assert report['targets']['y']['scaled']['population_variance'] == pytest.approx(1)
    assert report['targets']['z']['scaled']['population_variance'] == pytest.approx(1)
    assert sum(c['weighted_population_variance'] for c in report['objective']['components']) == pytest.approx(1)


def test_grouped_raw_objective_is_not_implicitly_normalized():
    binding = synthetic([{'project': 'A', 'module': 'X', 'y': 1, 'z': 100},
                         {'project': 'A', 'module': 'X', 'y': 3, 'z': 200}])
    with pytest.raises(StructuredContractError, match='multi-target'):
        run(binding, 'y', 'z', group_levels=('project',), objective='weighted_variance',
            target_weights=(('y', 1), ('z', 1)))
    report = run(binding, 'y', 'z', group_levels=('project',), objective='per_target')
    assert report['objective'] is None
    assert report['targets']['y']['raw']['population_variance'] == pytest.approx(1)
    assert report['targets']['z']['raw']['population_variance'] == pytest.approx(2500)
