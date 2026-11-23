"""ML-UW-32: scientific acceptance tests for aligned discrete information."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml_lab.structured import (
    StructuredObject, StructuredContractError, TransformationPlan, transform,
    estimate_information, estimate_entropy, estimate_conditional_entropy,
    estimate_mutual_information, estimate_conditional_mutual_information,
)

FIXTURE = Path(__file__).parent / 'fixtures' / 'structured_uncertainty' / 'xml_tree.json'


def build(rows, *, topology=None):
    """Supply explicitly aligned statistical rows independent of graph nodes."""
    payload = json.loads(FIXTURE.read_text())
    payload.pop('object_id', None)
    if topology:
        payload['topology'] = {'kind': topology}
        if topology == 'dag':
            payload['edges'].append({
                'id': 'extra-parent', 'kind': 'test', 'source_id': 'root', 'target_id': 'name'})
        if topology == 'directed_graph':
            payload['edges'].append({
                'id': 'cycle', 'kind': 'test', 'source_id': 'value', 'target_id': 'root'})
    keys = tuple(rows[0]) if rows else ('x', 'y', 'z')
    payload['observations'] = [
        {'observation_id': f'row-{i}', 'values': {
            name: cell(value) for name, value in record.items()
        }}
        for i, record in enumerate(rows)
    ]
    assert all(set(record) == set(keys) for record in rows)
    return StructuredObject.from_dict(payload)


def cell(value):
    if isinstance(value, tuple) and value[0] == 'status':
        state, reason = value[1], value[2] if len(value) > 2 else 'test_source'
        return {'status': state, 'value': None if state != 'empty' else '',
                'source_reason': reason}
    if value is None:
        return {'status': 'missing', 'value': None, 'source_reason': 'not_observed'}
    if value == '':
        return {'status': 'empty', 'value': ''}
    return {'status': 'present', 'value': value}


def xor_rows(repeats=3):
    return [
        {'x': x, 'y': y, 'z': x ^ y}
        for _ in range(repeats) for x in (0, 1) for y in (0, 1)
    ]


def assert_json(report):
    assert report['schema'] == 'ml-lab.structured-information@1'
    json.dumps(report, allow_nan=False)


def test_exact_binary_entropy_and_bit_units():
    obj = build(xor_rows())
    report = estimate_entropy(obj, 'x')
    assert report['value'] == pytest.approx(1)
    assert report['log_base'] == 2
    assert report['support']['included_rows'] == 12
    assert report['support']['distinct_states']['x'] == 2
    assert_json(report)


def test_entropy_constant_variable_is_zero_with_adequate_support():
    obj = build([{'x': 'one'} for _ in range(4)])
    r = estimate_entropy(obj, 'x')
    assert r['status'] == 'ok' and r['value'] == 0


def test_entropy_uses_explicit_log_base():
    obj = build(xor_rows())
    r = estimate_entropy(obj, 'x', log_base=4)
    assert r['value'] == pytest.approx(.5)


def test_mutual_information_independent_vs_perfect_association():
    obj = build(xor_rows())
    assert estimate_mutual_information(obj, 'x', 'y')['value'] == pytest.approx(0, abs=1e-12)
    assert estimate_mutual_information(obj, 'x', 'z')['value'] == pytest.approx(0, abs=1e-12)
    perfect = build([{'x': x, 'y': x} for _ in range(3) for x in (0, 1)])
    assert estimate_mutual_information(perfect, 'x', 'y')['value'] == pytest.approx(1)
    assert estimate_mutual_information(perfect, 'y', 'x')['value'] == pytest.approx(1)


def test_conditional_entropy_and_mi_xor_identity():
    obj = build(xor_rows())
    assert estimate_conditional_entropy(obj, 'x', ('z',))['value'] == pytest.approx(1)
    assert estimate_conditional_mutual_information(obj, 'x', 'y', ('z',))['value'] == pytest.approx(1)
    assert estimate_conditional_mutual_information(obj, 'y', 'x', ('z',))['value'] == pytest.approx(1)
    assert estimate_conditional_entropy(obj, 'x', ('y', 'z'))['value'] == pytest.approx(0)


def test_conditional_entropy_zero_when_target_determined_by_context():
    obj = build([{'x': x, 'group': x} for _ in range(3) for x in (0, 1)])
    assert estimate_conditional_entropy(obj, 'x', ('group',))['value'] == pytest.approx(0)


def test_two_conditioning_variables_use_complete_joint_context():
    rows = [{'x': x, 'y': y, 'c1': z, 'c2': w}
            for _ in range(3) for x in (0, 1) for y in (0, 1)
            for z in (0, 1) for w in (0, 1)]
    obj = build(rows)
    report = estimate_conditional_mutual_information(obj, 'x', 'y', ('c1', 'c2'))
    assert report['value'] == pytest.approx(0, abs=1e-12)
    assert report['support']['conditioning_strata'] == 4


def test_sources_do_not_change_graph_independent_statistics():
    obj = build(xor_rows(), topology='dag')
    report = estimate_conditional_mutual_information(obj, 'x', 'y', ('z',))
    assert 'rooted_hierarchy' not in obj.capabilities
    assert report['value'] == pytest.approx(1)
    cyclic = build(xor_rows(), topology='directed_graph')
    assert 'acyclic' not in cyclic.capabilities
    assert estimate_mutual_information(cyclic, 'x', 'y')['value'] == pytest.approx(0, abs=1e-12)


def test_structural_transformations_do_not_duplicate_information_samples():
    obj = build(xor_rows())
    original = estimate_mutual_information(obj, 'x', 'y')
    flattened = estimate_mutual_information(transform(obj, TransformationPlan('flat')).object,
                                           'x', 'y')
    assert flattened['support']['included_rows'] == original['support']['included_rows']
    assert flattened['value'] == pytest.approx(original['value'])


def test_complete_case_missingness_exclusion_and_reasons():
    obj = build([
        {'x': 0, 'y': 0}, {'x': 1, 'y': 1},
        {'x': ('status', 'missing', 'path_absent'), 'y': 0},
        {'x': 1, 'y': ('status', 'explicit_null', 'nil')},
        {'x': 1, 'y': 0}, {'x': 0, 'y': 1},
    ])
    r = estimate_mutual_information(obj, 'x', 'y')
    assert r['support']['aligned_rows'] == 6
    assert r['support']['included_rows'] == 4
    assert r['support']['excluded_rows'] == 2
    assert r['support']['exclusion_reasons'] == {'incomplete_row': 2}
    assert r['support']['cell_status_counts']['x']['missing'] == 1
    assert r['support']['null_source_reasons']['x'] == {'path_absent': 1}
    assert r['support']['null_source_reasons']['y'] == {'nil': 1}
    assert r['status'] == 'ok'
    assert_json(r)


def test_as_category_retains_distinct_generic_null_states():
    rows = [
        {'x': ('status', 'missing', 'path_absent'), 'y': 0},
        {'x': ('status', 'missing', 'missing_attribute'), 'y': 0},
        {'x': ('status', 'explicit_null', 'explicit_xsi_nil'), 'y': 1},
        {'x': ('status', 'unknown', 'unresolved'), 'y': 1},
        {'x': '', 'y': 0}, {'x': 'real', 'y': 1},
    ]
    r = estimate_mutual_information(build(rows), 'x', 'y', missing_policy='as_category')
    assert r['support']['included_rows'] == 6
    assert r['support']['distinct_states']['x'] == 5  # same generic missing despite 2 reasons
    assert r['support']['null_source_reasons']['x']['path_absent'] == 1
    assert r['value'] is not None


def test_empty_distinct_from_missing_and_regular_string():
    rows = [{'x': ''}, {'x': ''}, {'x': None}, {'x': None}]
    r = estimate_entropy(build(rows), 'x', missing_policy='as_category')
    assert r['value'] == pytest.approx(1)
    assert r['support']['distinct_states']['x'] == 2


def test_missing_policy_error_fails_closed():
    obj = build([{'x': 1}, {'x': None}])
    with pytest.raises(StructuredContractError, match='missing value'):
        estimate_entropy(obj, 'x', missing_policy='error')


def test_insufficient_complete_case_does_not_fabricate_entropy():
    obj = build([{'x': 1}, {'x': None}, {'x': None}])
    r = estimate_entropy(obj, 'x')
    assert r['status'] == 'insufficient_samples'
    assert r['value'] is None
    assert r['support']['included_rows'] == 1


def test_under_supported_conditioning_strata_report_unavailable():
    obj = build([{'x': 0, 'y': 0}, {'x': 1, 'y': 0}, {'x': 0, 'y': 1}])
    r = estimate_conditional_entropy(obj, 'x', ('y',))
    assert r['value'] is None
    assert r['status'] == 'insufficient_conditioning_support'
    assert r['support']['under_supported_strata'] == 1
    allowed = estimate_conditional_entropy(obj, 'x', ('y',), min_stratum_samples=1)
    assert allowed['status'] == 'ok'


def test_binning_quantile_repeated_ties_not_split():
    obj = build([{'x': x} for x in [0., 0., 1., 1., 2., 2., 3., 3.]])
    r = estimate_entropy(obj, 'x', bins={'x': {'mode': 'quantile', 'bins': 2}})
    assert r['discretization']['x']['actual_bins'] == 2
    assert r['value'] == pytest.approx(1)


def test_equal_width_binning_is_deterministic_and_typed():
    obj = build([{'x': x} for x in [0., 0., 0.25, 0.75, 1., 1.]])
    opt = {'bins': {'x': {'mode': 'equal_width', 'bins': 2}}}
    first = estimate_entropy(obj, 'x', **opt)
    second = estimate_entropy(obj, 'x', **opt)
    assert first == second
    assert first['value'] == pytest.approx(1)
    assert first['discretization']['x']['bounds'] == [0., 1.]
    assert first['discretization']['x']['actual_bins'] == 2


def test_explicit_numeric_binning_categorical_mixed_variables():
    obj = build([{'x': x, 'y': y} for x, y in [(1., 'a'), (2., 'b'),
                                               (1., 'a'), (2., 'b')]])
    report = estimate_mutual_information(obj, 'x', 'y',
                                         bins={'x': {'mode': 'quantile', 'bins': 2}})
    assert report['value'] == pytest.approx(1)


def test_discrete_types_do_not_collide_as_categories():
    obj = build([{'x': x} for x in [1, True, '1', 1, True, '1']])
    assert estimate_entropy(obj, 'x')['support']['distinct_states']['x'] == 3


def test_constant_numeric_bin_has_one_actual_category():
    obj = build([{'x': 2.5} for _ in range(5)])
    for mode in ('quantile', 'equal_width'):
        r = estimate_entropy(obj, 'x', bins={'x': {'mode': mode, 'bins': 5}})
        assert r['discretization']['x']['actual_bins'] == 1
        assert r['value'] == 0


def test_floating_point_without_discretization_is_rejected():
    with pytest.raises(StructuredContractError, match='discretization'):
        estimate_entropy(build([{'x': 0.5}, {'x': 1.5}]), 'x')


def test_multiple_valued_cell_requires_explicit_aggregation():
    raw = build([{'x': 0}, {'x': 1}]).to_dict()
    raw.pop('object_id', None)
    raw['observations'][0]['values']['x'] = {
        'status': 'multiple', 'value': None,
        'occurrences': [cell(1), cell(2)]}
    obj = StructuredObject.from_dict(raw)
    with pytest.raises(StructuredContractError, match='repeated observations'):
        estimate_entropy(obj, 'x', missing_policy='as_category')


@pytest.mark.parametrize('measure,x,y,given', [
    ('not_a_measure', 'x', None, ()),
    ('entropy', 'x', 'y', ()),
    ('entropy', 'x', None, ('y',)),
    ('mutual_information', 'x', None, ()),
    ('mutual_information', 'x', 'x', ()),
    ('conditional_mutual_information', 'x', 'y', ()),
    ('conditional_mutual_information', 'x', 'y', ('y',)),
    ('conditional_entropy', 'x', None, ()),
    ('conditional_entropy', 'x', None, ('x',)),
])
def test_invalid_measure_or_variable_contracts_fail_closed(measure, x, y, given):
    with pytest.raises(StructuredContractError):
        estimate_information(build(xor_rows()), measure=measure, x=x, y=y, given=given)


@pytest.mark.parametrize('options', [
    {'min_samples': 1}, {'min_samples': True}, {'min_stratum_samples': 0},
    {'log_base': 1}, {'log_base': -2}, {'log_base': float('nan')},
    {'missing_policy': 'ignore_all'}, {'estimator': 'knn'},
    {'bins': {'ghost': {'mode': 'quantile', 'bins': 2}}},
    {'bins': {'x': {'mode': 'bogus', 'bins': 2}}},
    {'bins': {'x': {'mode': 'quantile', 'bins': 1}}},
    {'bins': {'x': {'mode': 'quantile', 'bins': 2, 'other': True}}},
])
def test_invalid_policy_and_estimator_options_rejected(options):
    with pytest.raises(StructuredContractError):
        estimate_entropy(build(xor_rows()), 'x', **options)


def test_rejects_unaligned_variable_and_unvalidated_inputs():
    obj = build(xor_rows())
    with pytest.raises(StructuredContractError, match='aligned'):
        estimate_entropy(obj, 'not_a_variable')
    with pytest.raises(StructuredContractError, match='validated'):
        estimate_entropy(obj.to_dict(), 'x')


def test_returns_detached_report_and_does_not_mutate_input():
    obj = build(xor_rows())
    before = obj.to_json()
    report = estimate_mutual_information(obj, 'x', 'y')
    report['support']['cell_status_counts']['x']['present'] = 999
    assert obj.to_json() == before


def test_information_provenance_ignores_xml_terms_and_uses_generic_states():
    obj = build([{'x': ('status', 'explicit_null', 'explicit_xsi_nil')},
                 {'x': ('status', 'missing', 'path_absent')},
                 {'x': ('status', 'missing', 'other')}, {'x': 'present'}])
    report = estimate_entropy(obj, 'x', missing_policy='as_category')
    assert report['support']['distinct_states']['x'] == 3
    assert report['support']['null_source_reasons']['x'] == {
        'explicit_xsi_nil': 1, 'other': 1, 'path_absent': 1}
    assert report['value'] > 0


def test_equal_width_handles_subnormal_spans_and_opposite_large_bounds():
    for low, high in [(5e-324, 1e-323), (-1e308, 1e308)]:
        obj = build([{'x': low}, {'x': low}, {'x': high}, {'x': high}])
        report = estimate_entropy(obj, 'x',
                                  bins={'x': {'mode': 'equal_width', 'bins': 2}})
        assert report['value'] == pytest.approx(1)
