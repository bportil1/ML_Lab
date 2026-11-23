"""ML-UW-35: matched, domain-neutral structural preservation studies."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest

from ml_lab.structured import (
    ComparisonCandidate, ComparisonStudy, StructuredContractError,
    StructuredObject, TransformationPlan, compare_representations,
)

FIXTURE = (Path(__file__).parent / 'fixtures' / 'structured_uncertainty'
           / 'xml_tree.json')


def structured(rows=None):
    raw = json.loads(FIXTURE.read_text())
    raw.pop('object_id', None)
    raw['observations'] = []
    if rows is None:
        # Repeated siblings share a source group and are held out together.
        rows = []
        for group in range(40):
            for sibling in range(2):
                x = (group // 2) % 2
                z = (group // 4) % 2
                rows.append((str(group), str(x), str(z), str(x ^ z)))
    for i, (group, x, z, target) in enumerate(rows):
        def cell(value):
            if value is None:
                return {'status': 'missing', 'value': None, 'source_reason': 'fixture_missing'}
            if value == '':
                return {'status': 'empty', 'value': ''}
            return {'status': 'present', 'value': value}
        raw['observations'].append({
            'observation_id': f'row-{i}',
            'values': {'group': cell(group), 'x': cell(x), 'z': cell(z),
                       'target': cell(target)},
        })
    return StructuredObject.from_dict(raw)


def study(*, group_variable='group', independent=False, include_reduced=True):
    candidates = [ComparisonCandidate('full', TransformationPlan('full'), ('x', 'z'))]
    if include_reduced:
        candidates.append(ComparisonCandidate(
            'partial', TransformationPlan('depth', max_depth=1,
                                          variables=('target', 'x')), ('x',)))
    candidates.append(ComparisonCandidate('flat', TransformationPlan('flat'), ('x', 'z')))
    return ComparisonStudy('target', tuple(candidates), group_variable=group_variable,
                           acknowledge_independent_rows=independent,
                           seed=14, holdout_fraction=0.25)


def by_label(result):
    return {candidate['label']: candidate for candidate in result['candidates']}


def test_matched_study_reports_structural_loss_separately_from_feature_loss():
    source = structured()
    original = source.to_json()
    result = compare_representations(source, study())
    items = by_label(result)
    assert result['schema'] == 'ml-lab.representation-comparison@1'
    assert result['source']['object_id'] == source.object_id
    assert result['support']['common_complete_rows'] == 80
    assert result['split']['disjoint_groups'] is True
    assert result['split']['train_rows'] + result['split']['test_rows'] == 80
    assert items['full']['structural_audit']['lossless'] is True
    assert items['flat']['retention']['retained_node_count'] == 5
    assert items['flat']['retention']['retained_edge_count'] == 0
    assert items['partial']['retention']['retained_node_count'] == 2
    assert items['partial']['retention']['omitted_baseline_predictors'] == ['z']
    assert items['partial']['structural_audit']['removed_variables'] == ['group', 'z']
    assert items['full']['predictive_holdout'] == items['flat']['predictive_holdout']
    assert items['flat']['delta_holdout_log_loss_vs_full_nats'] == 0
    assert items['partial']['predictive_holdout']['holdout_log_loss_nats'] > items['full']['predictive_holdout']['holdout_log_loss_nats']
    assert items['full']['information_train_only']['target_predictor_mi_bits'] == pytest.approx(1, abs=.05)
    assert items['partial']['information_train_only']['omitted_predictor_cmi_given_retained']['bits'] is not None
    assert source.to_json() == original
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_split_is_group_disjoint_and_reproducible_across_candidates():
    source = structured()
    plan = study()
    a = compare_representations(source, plan)
    b = compare_representations(source, plan)
    assert a == b
    assert a['study_id'] == b['study_id']
    assert ComparisonStudy.from_dict(json.loads(plan.to_json())) == plan
    train_groups = {int(item.split('-')[1]) // 2 for item in a['split']['train_observation_ids']}
    test_groups = {int(item.split('-')[1]) // 2 for item in a['split']['test_observation_ids']}
    assert train_groups.isdisjoint(test_groups)
    for item in a['candidates']:
        assert item['information_train_only']['support_rows'] == a['split']['train_rows']
        assert item['structural_audit']['input_observation_ids'] == item['structural_audit']['output_observation_ids']
        assert item['predictive_holdout']['status'] == 'ok'


def test_comparison_report_does_not_imply_graph_causes_predictions():
    source = structured()
    result = compare_representations(source, study(include_reduced=False))
    items = by_label(result)
    assert items['full']['information_train_only'] == items['flat']['information_train_only']
    assert items['full']['predictive_holdout'] == items['flat']['predictive_holdout']
    assert result['advisory']['structural_loss_without_feature_change_is_not_predictive_evidence']
    assert result['advisory']['probabilistic_graph_structure_not_inferred']
    assert result['advisory']['uncertainty_intervals_or_significance_tests'] == 'not_available'
    assert 'independent final test' in result['selection_warning']


def test_common_case_exclusion_avoids_unequal_support_and_keeps_null_semantics():
    rows = [(str(i), str(i % 2), str(i % 2), str(i % 2)) for i in range(18)]
    rows[0] = ('0', '0', None, '0')
    rows[1] = ('1', '1', '1', None)
    rows[2] = (None, '0', '0', '0')
    obj = structured(rows)
    report = compare_representations(obj, study())
    assert report['support']['source_rows'] == 18
    assert report['support']['excluded_rows'] == 3
    assert report['support']['common_complete_rows'] == 15
    assert report['split']['train_rows'] + report['split']['test_rows'] == 15
    assert all(item['information_train_only']['support_rows'] == report['split']['train_rows']
               for item in report['candidates'])


def test_missing_groups_or_explicit_independence_approval_required():
    with pytest.raises(StructuredContractError, match='group_variable'):
        study(group_variable=None)
    independent = study(group_variable=None, independent=True)
    result = compare_representations(structured(), independent)
    assert result['split']['disjoint_groups'] is None
    assert result['split']['independence_assumption'] == 'unverified'
    with pytest.raises(StructuredContractError, match='conflicts'):
        study(group_variable='group', independent=True)


def test_rejects_target_leakage_and_unpreserved_predictors():
    with pytest.raises(StructuredContractError, match='target must not'):
        ComparisonStudy('target', (
            ComparisonCandidate('full', TransformationPlan('full'), ('target',)),
            ComparisonCandidate('flat', TransformationPlan('flat'), ()),
        ), group_variable='group')
    bad = ComparisonStudy('target', (
        ComparisonCandidate('full', TransformationPlan('full'), ('x', 'z')),
        ComparisonCandidate('partial', TransformationPlan('depth', max_depth=1,
                            variables=('target',)), ('x',)),
    ), group_variable='group')
    with pytest.raises(StructuredContractError, match='projection removes'):
        compare_representations(structured(), bad)
    with pytest.raises(StructuredContractError, match='subset'):
        ComparisonStudy('target', (
            ComparisonCandidate('full', TransformationPlan('full'), ('x',)),
            ComparisonCandidate('flat', TransformationPlan('flat'), ('z',)),
        ), group_variable='group')


@pytest.mark.parametrize('invalid', [
    {'seed': -1}, {'seed': True}, {'holdout_fraction': 0},
    {'holdout_fraction': 1}, {'holdout_fraction': float('nan')},
    {'pseudocount': 0}, {'pseudocount': True}, {'max_joint_states': 0},
    {'min_train_rows': 0}, {'min_test_rows': 0},
])
def test_invalid_study_configuration(invalid):
    base = study()
    kwargs = base.to_dict()
    kwargs.update(invalid)
    with pytest.raises(StructuredContractError):
        ComparisonStudy.from_dict(kwargs)


def test_bad_roundtrip_schema_rejected():
    raw = study().to_dict()
    raw['schema'] = 'wrong'
    with pytest.raises(StructuredContractError, match='schema'):
        ComparisonStudy.from_dict(raw)
    raw = study().to_dict()
    raw['extra'] = 'oops'
    with pytest.raises(StructuredContractError, match='unknown'):
        ComparisonStudy.from_dict(raw)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        study().candidates[0].label = 'changed'


def test_floating_features_require_explicit_preprocessing():
    obj = structured([(str(i), float(i), str(i%2), str(i%2)) for i in range(20)])
    with pytest.raises(StructuredContractError, match='discretize'):
        compare_representations(obj, study())


def test_unscorable_holdout_target_states_not_dropped_silently():
    obj = structured([(str(i), str(i%2), str(i%3), str(i)) for i in range(18)])
    result = compare_representations(obj, study())
    for item in result['candidates']:
        assert item['predictive_holdout']['status'] == 'unseen_holdout_target_state'
        assert item['predictive_holdout']['holdout_log_loss_nats'] is None
        assert item['predictive_holdout']['holdout_accuracy'] is None
    assert result['advisory']['lowest_observed_holdout_log_loss'] is None


def test_pure_api_does_not_write_runtime_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    compare_representations(structured(), study())
    assert not list(tmp_path.iterdir())


def test_internal_names_cannot_collide_with_real_predictors():
    baseline = structured()
    raw = baseline.to_dict()
    raw.pop('object_id', None)
    for row in raw['observations']:
        row['values']['__target__'] = row['values'].pop('x')
    colliding = StructuredObject.from_dict(raw)
    candidates = (
        ComparisonCandidate('full', TransformationPlan('full'), ('__target__', 'z')),
        ComparisonCandidate('flat', TransformationPlan('flat'), ('__target__', 'z')),
    )
    renamed = compare_representations(colliding, ComparisonStudy(
        'target', candidates, group_variable='group', seed=14))
    original = compare_representations(baseline, study(include_reduced=False))
    assert by_label(renamed)['full']['predictive_holdout'] == by_label(original)['full']['predictive_holdout']
    assert by_label(renamed)['full']['information_train_only']['target_predictor_mi_bits'] == pytest.approx(
        by_label(original)['full']['information_train_only']['target_predictor_mi_bits'])


def test_sparse_conditioning_returns_unavailable_cmi_not_zero():
    rows = [(str(i), f'unique-{i}', str(i % 2), str(i % 2)) for i in range(30)]
    report = compare_representations(structured(rows), study())
    partial = by_label(report)['partial']
    cmi = partial['information_train_only']['omitted_predictor_cmi_given_retained']
    assert cmi['status'] == 'insufficient_conditioning_support'
    assert cmi['bits'] is None
    assert cmi['under_supported_strata'] > 0
    assert partial['predictive_holdout']['status'] == 'ok'


def test_categorical_joint_state_budget_is_enforced():
    rows = [(str(i), str(i), str(i % 2), str(i % 2)) for i in range(36)]
    original = study()
    configured = original.to_dict()
    configured['max_joint_states'] = 3
    with pytest.raises(StructuredContractError, match='max_joint_states'):
        compare_representations(structured(rows), ComparisonStudy.from_dict(configured))
