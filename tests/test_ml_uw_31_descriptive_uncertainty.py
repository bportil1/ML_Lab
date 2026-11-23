"""ML-UW-31: domain-neutral descriptive uncertainty acceptance and safeguards."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from ml_lab.structured import (
    StructuredObject, StructuredContractError, TransformationPlan, transform,
    describe_structure, describe_collection, describe_observations,
    describe_nested_variance,
)

FIXTURES = Path(__file__).parent / 'fixtures' / 'structured_uncertainty'


def fixture(name='xml_tree'):
    return StructuredObject.from_json((FIXTURES / f'{name}.json').read_text())


def from_rows(rows, *, topology='tree'):
    """Create paired records, not one record per node or edge."""
    all_vars = ('y', 'project', 'module')
    payload = fixture().to_dict()
    payload.pop('object_id')
    payload['topology'] = {'kind': topology}
    payload['observations'] = [
        {'observation_id': str(index), 'values': {
            name: ({'status': 'missing', 'value': None, 'source_reason': 'not_available'}
                   if data.get(name) is None
                   else {'status': 'present', 'value': data[name]})
            for name in all_vars}}
        for index, data in enumerate(rows)
    ]
    return StructuredObject.from_dict(payload)


def test_single_tree_structure_is_not_source_occurrence_rate():
    obj = fixture()
    report = describe_structure(obj)
    assert report['schema'].endswith('@1')
    assert report['support']['nodes'] == 5
    assert report['support']['edges'] == 4
    assert report['support']['aligned_observation_rows'] == 3
    assert report['hierarchy']['root_ids'] == ['root']
    assert report['hierarchy']['maximum_depth'] == 3
    assert report['hierarchy']['depth_counts'] == {0: 1, 1: 1, 2: 1, 3: 2}
    assert report['hierarchy']['child_count_distribution'] == {0: 2, 1: 2, 2: 1}
    assert 'occurrence' not in report
    assert 'no node/path occurrence probability' in report['limitations'][0]
    json.dumps(report, allow_nan=False)


def test_forest_can_profile_roots_and_child_counts_without_observations():
    raw = fixture().to_dict()
    raw.pop('object_id')
    raw['topology'] = {'kind': 'forest'}
    raw['edges'] = raw['edges'][1:]
    raw['observations'] = []
    obj = StructuredObject.from_dict(raw)
    report = describe_structure(obj)
    assert report['hierarchy']['root_count'] == 2
    assert report['support']['aligned_observation_rows'] == 0
    assert report['hierarchy']['child_cardinality']['count'] == 5


@pytest.mark.parametrize('case', ('shared_child_dag', 'cyclic_network'))
def test_non_tree_graph_has_graph_summary_but_no_hierarchy(case):
    report = describe_structure(fixture(case))
    assert report['degree']
    assert report['hierarchy'] is None
    assert report['support']['nodes'] > 0


def test_collection_occurrence_support_and_absence_are_explicit():
    one = fixture()
    raw = one.to_dict()
    raw.pop('object_id')
    raw['nodes'] = raw['nodes'][:-1]
    raw['edges'] = raw['edges'][:-1]
    raw['topology'] = {'kind': 'tree'}
    two = StructuredObject.from_dict(raw)
    report = describe_collection([one, two], node_identity='id', edge_identity='id')
    rates = {x['identity']: x for x in report['node_occurrence']}
    assert rates['root']['empirical_presence_rate'] == 1
    assert rates['value']['empirical_presence_rate'] == 0.5
    assert rates['value']['absent_objects'] == 1
    erates = {x['identity']: x for x in report['edge_occurrence']}
    assert erates['e4']['empirical_presence_rate'] == 0.5
    assert report['support']['objects'] == 2


def test_collection_requires_explicit_alignment_and_never_dedupes_identical_objects():
    obj = fixture()
    with pytest.raises(TypeError):
        describe_collection([obj, obj])
    report = describe_collection([obj, obj], node_identity='id', edge_identity='id')
    assert report['support']['objects'] == 2
    assert all(r['present_objects'] == 2 for r in report['node_occurrence'])
    with pytest.raises(StructuredContractError, match='identity'):
        describe_collection([obj], node_identity='attributes:canonical', edge_identity='id')


def test_collection_same_semantic_path_can_have_multiple_instances():
    raw = fixture().to_dict()
    raw.pop('object_id')
    for node in raw['nodes']:
        node['attributes']['logical_path'] = '/same' if node['id'] in ('name', 'value') else node['id']
    for edge in raw['edges']:
        edge['attributes'] = {'relation': 'member'}
    obj = StructuredObject.from_dict(raw)
    report = describe_collection([obj], node_identity='attributes:logical_path', edge_identity='id')
    same = next(r for r in report['node_occurrence'] if r['identity'] == '/same')
    assert same['present_objects'] == 1
    assert same['total_occurrences'] == 2


def test_observation_statuses_denominator_and_null_reasons():
    report = describe_observations(fixture(), variables=('name', 'value'), numeric_variables=('value',))
    assert report['support']['aligned_rows'] == 3
    value = report['variables']['value']
    assert value['status_counts'] == {'explicit_null': 1, 'missing': 1, 'present': 1}
    assert value['status_rates']['missing'] == pytest.approx(1/3)
    assert value['null_source_reasons'] == {'explicit_xsi_nil': 1, 'path_absent': 1}
    assert value['numeric']['count'] == 1
    assert value['numeric']['sample_variance'] is None
    assert value['numeric']['variance_support'] == 'insufficient'
    assert report['variables']['name']['support']['empty_rows'] == 1
    assert 'node_count_not_used_as_sample_size' in report['support']
    json.dumps(report, allow_nan=False)


def test_repeated_observations_do_not_inflate_sample_sizes():
    raw = fixture().to_dict()
    raw.pop('object_id')
    raw['observations'][0]['values']['value'] = {
        'status': 'multiple', 'value': None,
        'occurrences': [
            {'status': 'present', 'value': 10},
            {'status': 'present', 'value': 12},
            {'status': 'explicit_null', 'value': None},
        ],
    }
    report = describe_observations(StructuredObject.from_dict(raw),
                                   variables=('value',), numeric_variables=('value',))
    value = report['variables']['value']
    assert value['support']['rows'] == 3
    assert value['status_counts']['multiple'] == 1
    assert value['repeated_occurrence_status_counts'] == {'explicit_null': 1, 'present': 2}
    assert value['numeric']['count'] == 0
    assert value['numeric']['sample_variance'] is None


def test_numeric_rejects_text_booleans_nonfinite_and_missing_alignment():
    with pytest.raises(StructuredContractError, match='numeric'):
        describe_observations(fixture(), variables=('name',), numeric_variables=('name',))
    with pytest.raises(StructuredContractError, match='aligned'):
        describe_observations(fixture(), variables=('missing_var',))
    with pytest.raises(StructuredContractError, match='validated'):
        describe_structure(fixture().to_dict())
    raw = fixture().to_dict()
    raw.pop('object_id')
    raw['observations'][0]['values']['value']['value'] = True
    with pytest.raises(StructuredContractError, match='numeric'):
        describe_observations(StructuredObject.from_dict(raw), variables=('value',), numeric_variables=('value',))
    with pytest.raises(StructuredContractError, match='NaN'):
        raw['observations'][0]['values']['value']['value'] = float('nan')
        StructuredObject.from_dict(raw)


def test_nested_variance_decomposes_unbalanced_groups_and_reports_support():
    # Y grand mean 3.33333; nested groups need not be balanced.
    obj = from_rows([
        {'project': 'P', 'module': 'M1', 'y': 1},
        {'project': 'P', 'module': 'M1', 'y': 3},
        {'project': 'P', 'module': 'M2', 'y': 5},
        {'project': 'Q', 'module': 'M1', 'y': 4},
        {'project': 'Q', 'module': 'M1', 'y': 6},
        {'project': 'Q', 'module': 'M2', 'y': 1},
    ])
    report = describe_nested_variance(obj, value_variable='y', group_levels=('project', 'module'))
    assert report['support']['complete_case_rows'] == 6
    assert report['support']['excluded_rows'] == 0
    assert report['outcome']['sample_variance'] == pytest.approx(4.266666666666667)
    assert [r['group_count'] for r in report['components']] == [2, 4, 4]
    assert sum(r['sum_squares'] for r in report['components']) == pytest.approx(report['total_sum_squares'])
    assert report['components'][-1]['sum_squares'] == pytest.approx(4)
    assert sum(r['proportion_total_ss'] for r in report['components']) == pytest.approx(1)
    json.dumps(report, allow_nan=False)


def test_nested_variance_constant_outcome_no_fake_information_fraction():
    obj = from_rows([{'project': 'P', 'module': 'M', 'y': 3},
                     {'project': 'Q', 'module': 'M', 'y': 3}])
    report = describe_nested_variance(obj, value_variable='y', group_levels=('project', 'module'))
    assert report['status'] == 'constant'
    assert report['total_sum_squares'] == 0
    assert all(part['proportion_total_ss'] is None for part in report['components'])


def test_variance_missing_groups_and_outcomes_are_not_imputed():
    obj = from_rows([
        {'project': 'P', 'module': 'M', 'y': 1},
        {'project': 'P', 'module': None, 'y': 2},
        {'project': 'Q', 'module': 'M', 'y': None},
        {'project': 'Q', 'module': 'M', 'y': 3},
    ])
    report = describe_nested_variance(obj, value_variable='y', group_levels=('project', 'module'))
    assert report['support']['complete_case_rows'] == 2
    assert report['support']['numeric_outcome_rows'] == 3
    assert report['support']['excluded_rows'] == 2
    assert report['support']['exclusion_reasons'] == {'group_module_missing': 1, 'outcome_missing': 1}


def test_insufficient_variance_does_not_become_zero():
    obj = from_rows([{'project': 'P', 'module': 'M', 'y': 2}])
    report = describe_nested_variance(obj, value_variable='y', group_levels=('project',))
    assert report['components'] is None
    assert report['outcome']['sample_variance'] is None
    assert report['status'] == 'insufficient'


def test_invalid_grouping_and_numeric_type_fail_closed():
    obj = from_rows([{'project': 'P', 'module': 'M', 'y': 1},
                     {'project': 'P', 'module': 'M', 'y': 3}])
    with pytest.raises(StructuredContractError, match='different'):
        describe_nested_variance(obj, value_variable='project', group_levels=('project',))
    with pytest.raises(StructuredContractError, match='unique'):
        describe_nested_variance(obj, value_variable='y', group_levels=('project', 'project'))
    with pytest.raises(StructuredContractError, match='aligned'):
        describe_nested_variance(obj, value_variable='y', group_levels=('ghost',))


def test_transformed_tree_report_keeps_observation_unit():
    obj = fixture()
    truncated = transform(obj, TransformationPlan('depth', max_depth=1)).object
    assert describe_structure(truncated)['support']['aligned_observation_rows'] == 3
    assert describe_observations(truncated, variables=('value',), numeric_variables=('value',))['support']['aligned_rows'] == 3
    flat = transform(obj, TransformationPlan('flat')).object
    assert describe_structure(flat)['hierarchy'] is None
    assert describe_observations(flat, variables=('value',))['support']['aligned_rows'] == 3


def test_core_reports_do_not_mutate_original_object():
    obj = fixture()
    before = obj.to_json()
    report = describe_observations(obj, variables=('name', 'value'), numeric_variables=('value',))
    report['variables']['value']['status_counts']['missing'] = 999
    assert obj.to_json() == before


def test_repeated_null_provenance_is_reported_without_treating_children_as_rows():
    raw = fixture().to_dict()
    raw.pop('object_id')
    raw['observations'][0]['values']['value'] = {
        'status': 'multiple', 'value': None,
        'occurrences': [
            {'status': 'missing', 'value': None, 'source_reason': 'my_adapter_missing'},
            {'status': 'explicit_null', 'value': None, 'source_reason': 'my_adapter_nil'},
            {'status': 'present', 'value': 3},
        ],
    }
    report = describe_observations(StructuredObject.from_dict(raw), variables=('value',))
    summary = report['variables']['value']
    assert summary['support']['rows'] == 3
    assert summary['repeated_occurrence_null_reasons'] == {
        'my_adapter_missing': 1, 'my_adapter_nil': 1,
    }


def test_all_missing_outcome_within_groups_reports_insufficient_not_zero():
    obj = from_rows([
        {'project': 'P', 'module': 'M', 'y': None},
        {'project': 'P', 'module': 'M', 'y': None},
    ])
    report = describe_nested_variance(obj, value_variable='y', group_levels=('project',))
    assert report['support']['complete_case_rows'] == 0
    assert report['components'] is None
    assert report['outcome']['sample_variance'] is None
    assert report['status'] == 'insufficient'


def test_numeric_overflow_fails_closed_and_cannot_emit_nan():
    obj = from_rows([
        {'project': 'P', 'module': 'M', 'y': 1e200},
        {'project': 'Q', 'module': 'M', 'y': -1e200},
    ])
    with pytest.raises(StructuredContractError, match='numerical range'):
        describe_observations(obj, variables=('y',), numeric_variables=('y',))
    with pytest.raises(StructuredContractError, match='numerical range'):
        describe_nested_variance(obj, value_variable='y', group_levels=('project',))


def test_reduced_graph_does_not_imply_probability_from_degree():
    graph = fixture('cyclic_network')
    report = describe_structure(graph)
    assert report['hierarchy'] is None
    assert 'probability' not in report['degree']
