"""SU-09: source-pinned paired retention and leakage-safe scoring."""
from dataclasses import replace
import json

import pytest

from ml_lab.structured import (
    ObservationBindingPlan, StructuralCandidatePlan, StructuralRetentionStudy,
    StructuredContractError, StructuredObject, VariableBinding,
    bind_observations, binding_sha256, compare_structural_retention, source_sha256,
)


def dataset(*, kind='categorical', count=48, constant=False):
    doc = {
        'schema': 'ml-lab.structured-object@1',
        'topology': {'kind': 'tree', 'capabilities': ['graph_edges', 'directed_edges',
            'acyclic', 'rooted_hierarchy', 'unique_parent', 'aligned_observations']},
        'nodes': [{'id': name, 'kind': 'uninterpreted', 'attributes': {}}
                  for name in ('root', 'a', 'b', 't')],
        'edges': [{'id': f'e{i}', 'kind': 'structural', 'source_id': 'root',
                   'target_id': name} for i, name in enumerate(('a', 'b', 't'))],
        'observation_unit': 'record', 'provenance': {'fixture': 'su09'},
        'observations': [],
    }
    for i in range(count):
        a = 0 if constant else (i * 13) % 19
        b = 'A' if constant else f'cat-{(i * 7) % 5}'
        y = ('same' if constant else ('high' if a >= 9 else 'low')) if kind == 'categorical' else float(a * 2 + (i % 3))
        doc['observations'].append({'observation_id': f'r{i:03}', 'values': {
            'a': {'status': 'present', 'value': a},
            'b': {'status': 'present', 'value': b},
            'target': {'status': 'present', 'value': y},
            'group': {'status': 'present', 'value': f'g{i//3:02}'},
        }})
    source = StructuredObject.from_dict(doc)
    binding = ObservationBindingPlan(source_sha256(source), 'record', (
        VariableBinding('a', ('a',)), VariableBinding('b', ('b',)),
        VariableBinding('target', ('t',))), group_variables=('group',))
    return source, binding


def study_for(source, binding, *, kind='categorical_target', target='target',
              grouped=True, **overrides):
    digest = binding_sha256(bind_observations(source, binding))
    base = {'source_digest': source_sha256(source), 'binding_digest': digest,
            'candidates': (
                StructuralCandidatePlan(source_sha256(source), 'preserve', binding_digest=digest),
                StructuralCandidatePlan(source_sha256(source), 'flatten', binding_digest=digest),
                StructuralCandidatePlan(source_sha256(source), 'prune', ('b',),
                    observation_policy='drop_exclusive', binding_digest=digest),
                StructuralCandidatePlan(source_sha256(source), 'prune', ('a',),
                    observation_policy='drop_exclusive', binding_digest=digest)),
            'predictors': ('a', 'b'), 'kind': kind, 'target': target,
            'group_variable': 'group' if grouped else None,
            'acknowledge_independent_rows': not grouped,
            'seed': 17, 'min_train_rows': 5, 'min_test_rows': 3}
    base.update(overrides)
    return StructuralRetentionStudy(**base)


def entries(report):
    rows = report.to_dict()['candidates']
    return rows[0], rows[1], rows[2], rows[3]


def test_categorical_predictive_scores_identical_when_only_graph_changes():
    source, binding = dataset()
    original = source.to_json()
    plan = study_for(source, binding, kind='categorical_target')
    result = compare_structural_retention(source, binding, plan)
    preserve, flat, drop_b, drop_a = entries(result)
    assert result.to_dict()['schema'] == 'ml-lab.structural-retention-comparison@1'
    assert preserve['output_digest'] == source_sha256(source)
    assert flat['complexity']['retained_edges'] == 0
    assert preserve['metrics'] == flat['metrics']
    assert preserve['support']['train_rows'] == flat['support']['train_rows']
    assert drop_a['feature_impact']['dropped_variable_ids'] == ['a']
    assert drop_b['evaluated_predictors'] == ['a']
    assert drop_a['evaluated_predictors'] == ['b']
    assert drop_b['metrics']['holdout_accuracy']['status'] == 'ok'
    assert preserve['metrics']['retained_entropy_fraction']['value'] == pytest.approx(1)
    assert result.to_dict()['split']['group_leakage_prevented'] is True
    assert set(result.to_dict()['split']['train_observation_ids']).isdisjoint(
        result.to_dict()['split']['test_observation_ids'])
    assert 'train_only_bin_cutpoints' in result.to_dict()['estimation']
    assert source.to_json() == original
    assert compare_structural_retention(source, binding, plan).to_json() == result.to_json()
    json.dumps(result.to_dict(), allow_nan=False)


def test_numeric_predictive_mse_r2_and_topology_invariance():
    source, binding = dataset(kind='numeric')
    result = compare_structural_retention(source, binding,
        study_for(source, binding, kind='numeric_target'))
    full, flat, _, removed_a = entries(result)
    assert full['metrics'] == flat['metrics']
    assert full['metrics']['holdout_mse']['status'] == 'ok'
    assert full['metrics']['holdout_r2']['status'] == 'ok'
    assert full['metrics']['heldout_plugin_mi_bits']['status'] == 'numeric_target_information_not_defined'
    assert removed_a['metrics']['holdout_r2']['status'] == 'ok'
    assert result.to_dict()['pareto']['metric'] == 'holdout_r2'


def test_target_free_retained_variance_entropy_and_numeric_reconstruction():
    source, binding = dataset()
    plan = study_for(source, binding, kind='target_free', target=None,
                     variance_variables=('a',), reconstruction_variables=('a',),
                     threshold_metric='retained_variance_fraction', minimum_threshold=.95)
    report = compare_structural_retention(source, binding, plan)
    full, flat, drop_b, drop_a = entries(report)
    assert full['metrics'] == flat['metrics']
    assert full['metrics']['retained_variance_fraction']['value'] == pytest.approx(1)
    assert drop_b['metrics']['retained_variance_fraction']['value'] == pytest.approx(1)
    assert drop_a['metrics']['retained_variance_fraction']['value'] == pytest.approx(0)
    assert full['metrics']['reconstruction_mse']['value'] == 0
    assert drop_a['metrics']['reconstruction_mse']['value'] > 0
    assert drop_a['metrics']['reconstruction_r2']['value'] < 1
    assert full['meets_declared_threshold'] is True
    assert drop_b['meets_declared_threshold'] is True
    assert drop_a['meets_declared_threshold'] is False
    assert full['metrics']['retained_entropy_fraction']['value'] == pytest.approx(1)
    assert report.to_dict()['pareto']['threshold_metric'] == 'retained_variance_fraction'


def test_preserve_only_observation_policy_never_claims_empirical_loss():
    source, binding = dataset()
    plan = study_for(source, binding, kind='target_free', target=None)
    plans = list(plan.candidates)
    plans[2] = replace(plans[2], observation_policy='preserve')
    plans[3] = replace(plans[3], observation_policy='preserve')
    result = compare_structural_retention(source, binding, replace(plan, candidates=tuple(plans)))
    reference = result.to_dict()['candidates'][0]['metrics']
    assert all(item['metrics'] == reference for item in result.to_dict()['candidates'])
    assert result.to_dict()['candidates'][3]['feature_impact']['dropped_variable_ids'] == []
    assert result.to_dict()['candidates'][3]['structural_loss']['removed_node_count'] == 1


def test_constant_reference_entropy_and_train_scale_return_unavailable_not_nan():
    source, binding = dataset(constant=True)
    plan = study_for(source, binding, kind='target_free', target=None,
                     variance_variables=('a',), reconstruction_variables=('a',))
    result = compare_structural_retention(source, binding, plan)
    full = result.to_dict()['candidates'][0]['metrics']
    assert full['retained_entropy_fraction']['status'] == 'zero_reference_denominator'
    assert full['retained_variance_fraction']['status'] == 'zero_reference_denominator'
    assert full['reconstruction_r2']['status'] == 'zero_reference_denominator'


def test_missingness_excluded_once_across_all_candidates():
    source, _ = dataset()
    doc = source.to_dict(); doc.pop('object_id')
    doc['observations'][2]['values']['b'] = {'status': 'missing', 'value': None,
                                              'source_reason': 'path_absent'}
    changed = StructuredObject.from_dict(doc)
    new_binding = ObservationBindingPlan(source_sha256(changed), 'record', (
        VariableBinding('a', ('a',)), VariableBinding('b', ('b',)),
        VariableBinding('target', ('t',))), group_variables=('group',))
    result = compare_structural_retention(changed, new_binding,
        study_for(changed, new_binding, kind='categorical_target'))
    support = result.to_dict()['complete_case_support']
    assert support['source_rows'] == 48
    assert support['included_rows'] == 47
    assert support['excluded_statuses'] == {'b:missing': 1}
    assert len({row['support']['test_rows'] for row in result.to_dict()['candidates']}) == 1


def test_stale_source_binding_and_group_requirements_fail_closed():
    source, binding = dataset()
    plan = study_for(source, binding)
    with pytest.raises(StructuredContractError, match='source fingerprint'):
        compare_structural_retention(dataset(count=51)[0], binding, plan)
    other = ObservationBindingPlan(source_sha256(source), 'record', (
        VariableBinding('a', ('a',)), VariableBinding('b', ('b',)),
        VariableBinding('target', ('t',))))
    with pytest.raises(StructuredContractError, match='binding fingerprint'):
        compare_structural_retention(source, other, plan)
    with pytest.raises(StructuredContractError, match='group_variable must be declared'):
        other_digest = binding_sha256(bind_observations(source, other))
        compare_structural_retention(source, other, replace(plan, binding_digest=other_digest,
                candidates=tuple(replace(p, binding_digest=other_digest) for p in plan.candidates)))
    with pytest.raises(StructuredContractError, match='independent-row acknowledgment'):
        replace(plan, group_variable=None)


def test_plan_roundtrip_strict_validation_and_duplicate_prevention():
    source, binding = dataset()
    plan = study_for(source, binding)
    assert StructuralRetentionStudy.from_json(plan.to_json()) == plan
    with pytest.raises(StructuredContractError, match='duplicate'):
        StructuralRetentionStudy.from_json(plan.to_json().replace('"seed":17', '"seed":17,"seed":1'))
    with pytest.raises(StructuredContractError, match='duplicate candidate'):
        replace(plan, candidates=(plan.candidates[0], plan.candidates[0]))
    with pytest.raises(StructuredContractError, match='source/binding digest'):
        replace(plan, candidates=(replace(plan.candidates[0],
                                          binding_digest='sha256:'+'0'*64), *plan.candidates[1:]))
    with pytest.raises(StructuredContractError, match='target/group variable'):
        replace(plan, predictors=('target',))
    with pytest.raises(StructuredContractError, match='fraction/accuracy threshold'):
        replace(plan, threshold_metric='holdout_accuracy', minimum_threshold=1.1)


def test_unacknowledged_dependency_or_insufficient_partition_rejected():
    source, binding = dataset(count=9)
    plan = study_for(source, binding, min_train_rows=8)
    with pytest.raises(StructuredContractError, match='support thresholds'):
        compare_structural_retention(source, binding, plan)
    with pytest.raises(StructuredContractError, match='independent-row acknowledgment'):
        replace(plan, group_variable=None, acknowledge_independent_rows=False)


def test_no_unbounded_joint_state_or_misleading_statistical_sample_claim():
    source, binding = dataset()
    plan = study_for(source, binding, kind='categorical_target', max_joint_states=1)
    report = compare_structural_retention(source, binding, plan).to_dict()
    assert report['sampling_independence_verified'] is False
    assert report['candidates'][0]['metrics']['holdout_accuracy']['status'] == 'joint_state_limit'
    assert report['candidates'][0]['metrics']['retained_entropy_fraction']['status'] == 'joint_state_limit'
    assert report['candidates'][0]['pareto_eligible'] is False


def test_missing_group_key_prevents_group_split():
    source, _ = dataset()
    doc = source.to_dict(); doc.pop('object_id')
    doc['observations'][0]['values']['group'] = {'status': 'missing', 'value': None}
    changed = StructuredObject.from_dict(doc)
    binding = ObservationBindingPlan(source_sha256(changed), 'record', (
        VariableBinding('a', ('a',)), VariableBinding('b', ('b',)),
        VariableBinding('target', ('t',))), group_variables=('group',))
    with pytest.raises(StructuredContractError, match='missing grouping key'):
        compare_structural_retention(changed, binding, study_for(changed, binding))


def test_paired_predictive_deltas_reference_exactly_zero():
    source, binding = dataset()
    rows = compare_structural_retention(source, binding, study_for(source, binding)).to_dict()['candidates']
    for metric in ('holdout_accuracy', 'holdout_log_loss_nats'):
        delta = metric + '_delta_vs_preserve'
        assert rows[0]['metrics'][delta]['value'] == 0
        assert rows[1]['metrics'][delta]['value'] == 0
        assert rows[0]['metrics'][metric] == rows[1]['metrics'][metric]


def test_invalid_unrequested_thresholds_rejected():
    source, binding = dataset()
    study = study_for(source, binding, kind='target_free', target=None)
    with pytest.raises(StructuredContractError, match='requires variance_variables'):
        replace(study, threshold_metric='retained_variance_fraction', minimum_threshold=.9)
    with pytest.raises(StructuredContractError, match='requires reconstruction_variables'):
        replace(study, threshold_metric='reconstruction_r2', minimum_threshold=.9)


def test_unresolved_repeated_cell_excluded_and_not_extra_sample():
    source, _ = dataset()
    doc = source.to_dict(); doc.pop('object_id')
    doc['observations'][0]['values']['a'] = {'status': 'multiple', 'value': None, 'occurrences': [
        {'status': 'present', 'value': 1}, {'status': 'present', 'value': 3}]}
    source = StructuredObject.from_dict(doc)
    binding = ObservationBindingPlan(source_sha256(source), 'record', (
        VariableBinding('a', ('a',)), VariableBinding('b', ('b',)),
        VariableBinding('target', ('t',))), group_variables=('group',))
    result = compare_structural_retention(source, binding, study_for(source, binding))
    assert result.to_dict()['complete_case_support']['excluded_statuses']['a:multiple'] == 1
    assert result.to_dict()['complete_case_support']['included_rows'] == 47
    resolved = ObservationBindingPlan(source_sha256(source), 'record', (
        VariableBinding('a', ('a',), 'mean_numeric'), VariableBinding('b', ('b',)),
        VariableBinding('target', ('t',))), group_variables=('group',))
    result = compare_structural_retention(source, resolved, study_for(source, resolved))
    assert result.to_dict()['complete_case_support']['included_rows'] == 48


def test_weighted_binding_is_explicitly_unsupported_not_ignored():
    source, _ = dataset()
    doc = source.to_dict(); doc.pop('object_id')
    for obs in doc['observations']:
        obs['values']['sample_weight'] = {'status': 'present', 'value': 1.0}
    source = StructuredObject.from_dict(doc)
    binding = ObservationBindingPlan(source_sha256(source), 'record', (
        VariableBinding('a', ('a',)), VariableBinding('b', ('b',)),
        VariableBinding('target', ('t',))), group_variables=('group',),
        weight_variable='sample_weight')
    with pytest.raises(StructuredContractError, match='weighted binding unsupported'):
        compare_structural_retention(source, binding, study_for(source, binding))
