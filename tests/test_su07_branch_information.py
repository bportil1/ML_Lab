"""SU-07: empirical associations use aligned rows, not source edges or nodes."""
from __future__ import annotations

from itertools import product
import json
import math
from pathlib import Path

import pytest

from ml_lab.structured import (
    BranchInformationPlan, ObservationBindingPlan, StructuredContractError,
    StructuredObject, VariableBinding, analyze_branch_information,
    bind_observations, binding_sha256, source_sha256,
)

FIXTURES = Path(__file__).parent / 'fixtures' / 'structured_uncertainty'


def _source(rows, names=('a', 'b', 'y'), topology='xml_tree'):
    original = StructuredObject.from_json((FIXTURES / f'{topology}.json').read_text())
    doc = original.to_dict()
    doc.pop('object_id')
    if topology == 'xml_tree':
        for name in names:
            doc['nodes'].append({'id': name, 'kind': 'uninterpreted', 'attributes': {}})
            doc['edges'].append({'id': f'feature_{name}', 'source_id': 'item',
                                 'target_id': name, 'kind': 'structural', 'ordinal': 100 + len(doc['edges'])})
    else:
        names = tuple(node['id'] for node in doc['nodes'][:len(names)])
    if topology != 'xml_tree':
        doc['topology']['capabilities'] = [c for c in doc['topology']['capabilities']
                                           if c != 'aligned_observations']
    doc['observations'] = []
    for i, record in enumerate(rows):
        values = {}
        for name in names:
            val = record.get(name)
            if isinstance(val, dict):
                values[name] = val
            elif val is None:
                values[name] = {'status': 'missing', 'value': None, 'source_reason': 'path_absent'}
            else:
                values[name] = {'status': 'present', 'value': val}
        if 'weight' in record:
            values['weight'] = {'status': 'present', 'value': record['weight']}
        doc['observations'].append({'observation_id': f'row-{i:04d}', 'values': values})
    return StructuredObject.from_dict(doc), names


def _binding(rows, names=('a', 'b', 'y'), *, policies=None, weight=None, topology='xml_tree', merged=None):
    obj, effective = _source(rows, names, topology=topology)
    policies = policies or {}
    rules = []
    for name in effective:
        node_ids = (name,) if merged != name else (name, 'item')
        rules.append(VariableBinding(name, node_ids, policies.get(name, 'preserve'),
                                     merged_sources=(merged == name)))
    bound = bind_observations(obj, ObservationBindingPlan(
        source_sha256(obj), obj.observation_unit, tuple(rules), weight_variable=weight))
    return obj, bound


def _run(binding, *, nodes=('b',), target='y', retained=('a',), **kwargs):
    plan = BranchInformationPlan(binding_sha256(binding), nodes=nodes, target=target,
                                 retained_variables=retained, **kwargs)
    result = analyze_branch_information(binding, plan)
    return result.to_dict()


def _metric(report, key):
    return report['nodes'][0]['information']['relevance'][key]


def test_xor_conditional_information_vs_zero_marginal_mi():
    records = [{'a': a, 'b': b, 'y': a ^ b}
               for a, b in list(product((0, 1), repeat=2)) * 20]
    obj, binding = _binding(records)
    before = obj.to_json()
    report = _run(binding)
    assert report['schema'] == 'ml-lab.branch-information@1'
    assert report['support']['aligned_rows'] == 80
    node = report['nodes'][0]
    assert node['node_id'] == 'b' and node['scope'] == 'subtree'
    assert node['candidate_variable_ids'] == ['b']
    assert _metric(report, 'target_candidate_mi')['value'] == pytest.approx(0, abs=1e-12)
    assert _metric(report, 'target_candidate_cmi_given_retained')['value'] == pytest.approx(1)
    assert node['information']['redundancy_candidate_retained_mi']['value'] == pytest.approx(0)
    assert node['information']['entropy']['candidate_joint']['value'] == pytest.approx(1)
    assert node['information']['support']['conditioning_strata'] == 2
    assert report['association_layer']['source_edges'] == []
    assert report['association_layer']['probabilistic_edges'] == []
    assert report['association_layer']['causal_interpretation'] is False
    assert report['association_layer']['node_overlays'][0]['node_id'] == 'b'
    assert obj.to_json() == before
    assert json.dumps(report, allow_nan=False)
    assert report == _run(binding)


def test_redundant_clone_marginally_relevant_but_no_incremental_information():
    rows = [{'a': bit, 'clone': bit, 'y': bit} for bit in (0, 1) * 30]
    _, binding = _binding(rows, ('a', 'clone', 'y'))
    report = _run(binding, nodes=('clone',))
    assert _metric(report, 'target_candidate_mi')['value'] == pytest.approx(1)
    assert _metric(report, 'target_candidate_cmi_given_retained')['value'] == pytest.approx(0)
    assert report['nodes'][0]['information']['redundancy_candidate_retained_mi']['value'] == pytest.approx(1)
    assert report['nodes'][0]['information']['entropy']['candidate_given_retained']['value'] == pytest.approx(0)


def test_rare_informative_feature_is_not_discarded_for_low_variance_or_entropy():
    records = [{'a': i % 2, 'b': int(i == 99), 'y': int(i == 99)} for i in range(100)]
    _, binding = _binding(records)
    report = _run(binding, retained=())
    mi = _metric(report, 'target_candidate_mi')['value']
    assert mi == pytest.approx(-(0.99 * math.log2(.99) + .01 * math.log2(.01)))
    assert _metric(report, 'target_candidate_cmi_given_retained')['value'] == pytest.approx(mi)
    assert report['nodes'][0]['information']['redundancy_candidate_retained_mi']['status'] == 'no_retained_variables'
    assert 'redundancy_and_low_marginal_mi_do_not_authorize_pruning' in report['limitations']


def test_root_scope_excludes_target_and_retained_from_candidate_and_is_not_edge_attribution():
    records = [{'a': a, 'b': b, 'y': b} for a, b in product((0, 1), repeat=2) for _ in range(3)]
    _, binding = _binding(records)
    report = _run(binding, nodes=('root',))
    node = report['nodes'][0]
    assert node['candidate_variable_ids'] == ['b']
    assert node['retained_overlap_variable_ids'] == ['a']
    assert node['scope_variable_ids'] == ['a', 'b']
    assert node['information']['relevance']['target_scope_mi']['value'] == pytest.approx(1)
    assert node['information']['relevance']['target_candidate_cmi_given_retained']['value'] == pytest.approx(1)
    assert node['source_graph_edges_are_not_statistical_associations'] is True
    assert _run(binding, nodes=('a',))['nodes'][0]['information']['status'] == 'no_new_observed_variables'


def test_missingness_policies_have_explicit_record_support_and_conditioning_limits():
    records = [
        {'a': 0, 'b': 0, 'y': 0}, {'a': 0, 'b': 1, 'y': 1},
        {'a': 0, 'b': None, 'y': 1}, {'a': 1, 'b': 1, 'y': 0},
        {'a': 1, 'b': 0, 'y': None},
    ]
    _, binding = _binding(records)
    report = _run(binding, min_samples=2, min_stratum_samples=2)
    info = report['nodes'][0]['information']
    assert info['support']['included_rows'] == 3
    assert info['support']['excluded_rows'] == 2
    assert info['support']['exclusion_reasons'] == {'b:missing': 1, 'y:missing': 1}
    assert info['support']['under_supported_strata'] == 1
    assert info['relevance']['target_candidate_mi']['status'] == 'ok'
    assert info['relevance']['target_candidate_cmi_given_retained']['status'] == 'insufficient_conditioning_support'
    assert info['entropy']['candidate_given_retained']['status'] == 'insufficient_conditioning_support'
    expanded = _run(binding, missing_policy='as_category')
    assert expanded['nodes'][0]['information']['support']['included_rows'] == 5
    with pytest.raises(StructuredContractError, match='missing observation'):
        _run(binding, missing_policy='error')


def test_float_discretization_is_explicit_frozen_and_reported():
    rows = [{'a': k % 2, 'b': float(k), 'y': k % 2} for k in range(20)]
    _, binding = _binding(rows)
    with pytest.raises(StructuredContractError, match='floating-point'):
        _run(binding)
    result = _run(binding, bins=(('b', 'equal_width', 4),))
    spec = result['estimator']['discretization']['b']
    assert spec['mode'] == 'equal_width' and spec['actual_bins'] == 4
    assert spec['bounds'] == [0.0, 19.0]
    assert result['estimator']['bin_fit_scope'] == 'all_available_aligned_rows_not_fit_per_branch'
    assert result['nodes'][0]['information']['status'] == 'ok'
    quantile = _run(binding, bins=(('b', 'quantile', 3),))
    assert quantile['estimator']['discretization']['b']['mode'] == 'quantile'


def test_repeated_observations_require_explicit_su05_resolution():
    cells = [{'a': i % 2, 'b': i % 2, 'y': i % 2} for i in range(6)]
    cells[0]['b'] = {'status': 'multiple', 'value': None, 'occurrences': [
        {'status': 'present', 'value': 0}, {'status': 'present', 'value': 1}]}
    _, binding = _binding(cells)
    with pytest.raises(StructuredContractError, match='repeated observations'):
        _run(binding)
    _, mean_bound = _binding(cells, policies={'b': 'mean_numeric'})
    # The per-record mean becomes a float, so the information estimator still
    # requires explicit discretization rather than inventing a category.
    with pytest.raises(StructuredContractError, match='floating-point'):
        _run(mean_bound)
    result = _run(mean_bound, bins=(('b', 'equal_width', 2),))
    assert result['nodes'][0]['information']['support']['included_rows'] == 6


def test_merged_identity_excluded_without_fake_source_allocation():
    records = [{'a': v, 'b': v, 'y': v} for v in (0, 1) * 4]
    _, binding = _binding(records, merged='b')
    report = _run(binding)
    node = report['nodes'][0]
    assert node['candidate_variable_ids'] == []
    assert node['ambiguous_merged_variable_ids'] == ['b']
    assert node['information']['status'] == 'no_new_observed_variables'
    assert _run(binding, nodes=('root',))['nodes'][0]['ambiguous_merged_variable_ids'] == ['b']


def test_nonhierarchical_analysis_reports_direct_only_not_subtree():
    obj, names = _source([], topology='shared_child_dag')
    binding = bind_observations(obj, ObservationBindingPlan(
        source_sha256(obj), obj.observation_unit,
        (VariableBinding('unobserved', (names[0],)),)))
    report = _run(binding, nodes=(names[0],), target=None, retained=())
    assert report['nodes'][0]['scope'] == 'direct_only'
    assert report['nodes'][0]['information']['status'] == 'insufficient_samples'
    assert report['support']['aligned_rows'] == 0


def test_nonunit_weights_rejected_not_silently_ignored():
    records = [{'a': i % 2, 'b': i % 2, 'y': i % 2, 'weight': 2}
               for i in range(8)]
    _, bound = _binding(records, weight='weight')
    with pytest.raises(StructuredContractError, match='unit record weights'):
        _run(bound)


def test_plan_schema_validation_digest_and_detached_reports():
    _, binding = _binding([{'a': 1, 'b': 1, 'y': 1}, {'a': 0, 'b': 0, 'y': 0}])
    plan = BranchInformationPlan(binding_sha256(binding), nodes=('b',), target='y',
                                 retained_variables=('a',))
    assert BranchInformationPlan.from_json(plan.to_json()) == plan
    with pytest.raises(StructuredContractError, match='duplicate'):
        BranchInformationPlan.from_json(plan.to_json().replace('"schema":', '"schema":"x","schema":', 1))
    with pytest.raises(StructuredContractError, match='target cannot'):
        BranchInformationPlan(plan.binding_digest, target='a', retained_variables=('a',))
    with pytest.raises(StructuredContractError, match='bins'):
        BranchInformationPlan.from_dict({**plan.to_dict(), 'bins': ['bad']})
    with pytest.raises(StructuredContractError, match='fingerprint'):
        analyze_branch_information(binding, BranchInformationPlan('sha256:' + '0' * 64))
    with pytest.raises(StructuredContractError, match='exact source node'):
        _run(binding, nodes=('absent',))
    with pytest.raises(StructuredContractError, match='explicitly bound'):
        _run(binding, target='other')
    with pytest.raises(StructuredContractError, match='explicitly bound'):
        _run(binding, retained=('other',))
    result = analyze_branch_information(binding, plan)
    mutable = result.to_dict()
    mutable['nodes'][0]['node_id'] = 'changed'
    assert result.to_dict()['nodes'][0]['node_id'] == 'b'


def test_insufficient_support_and_no_target_do_not_manufacture_estimates():
    _, binding = _binding([{'a': 1, 'b': 1, 'y': 1}])
    report = _run(binding)
    assert report['nodes'][0]['information']['status'] == 'insufficient_samples'
    assert _metric(report, 'target_candidate_cmi_given_retained')['value'] is None
    no_target = _run(binding, target=None, retained=())
    assert no_target['nodes'][0]['information']['relevance']['target_candidate_mi']['status'] == 'no_target_declared'


def test_joint_feature_information_is_joint_not_sum_of_marginal_mi():
    records = [{'a': a, 'b': b, 'y': a ^ b} for a, b in product((0, 1), repeat=2) for _ in range(8)]
    _, binding = _binding(records)
    report = _run(binding, nodes=('item',), retained=())
    assert report['nodes'][0]['candidate_variable_ids'] == ['a', 'b']
    assert report['nodes'][0]['information']['entropy']['candidate_joint']['value'] == pytest.approx(2)
    assert report['nodes'][0]['information']['relevance']['target_candidate_mi']['value'] == pytest.approx(1)
