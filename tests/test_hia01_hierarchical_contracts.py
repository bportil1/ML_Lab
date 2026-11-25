"""HIA-01 gate contracts: no estimator/graph/UI execution is performed."""
from pathlib import Path
import copy
import json

import pytest

from ml_lab.structured import (
    HierarchicalScope, HierarchicalAnalysisRequest,
    HIA_REQUEST_SCHEMA, HIA_ELIGIBILITY_SCHEMA,
    ObservationBindingPlan, VariableBinding, bind_observations,
    binding_sha256, source_sha256, inspect_hierarchical_eligibility,
    StructuredContractError, StructuredObject,
)

FIXTURES = Path(__file__).parent / 'fixtures' / 'structured_uncertainty'


def src(name='xml_tree'):
    return StructuredObject.from_json((FIXTURES / f'{name}.json').read_text())


def bound(source, *, ambiguous=False):
    vars_ = (VariableBinding('name', ('name', 'value'), merged_sources=True),) if ambiguous else (
        VariableBinding('name', ('name',)), VariableBinding('value', ('value',)))
    return bind_observations(source, ObservationBindingPlan(
        source_sha256(source), source.observation_unit, vars_))


def request(source, *, kind='node', node='name', level=None, metrics=('node_structure',),
            binding=None, **kwargs):
    if kind in {'node', 'branch'}:
        scope = HierarchicalScope(kind, (node,))
    elif kind == 'level':
        scope = HierarchicalScope(kind, level=level)
    else:
        scope = HierarchicalScope(kind)
    return HierarchicalAnalysisRequest(
        source_sha256(source), scope, tuple(metrics),
        binding_digest=binding_sha256(binding) if binding else None, **kwargs)


def test_source_only_node_scope_is_valid_and_no_information_is_fabricated():
    source = src()
    req = request(source, metrics=('node_structure', 'node_descriptive'))
    result = inspect_hierarchical_eligibility(source, req)
    assert result['schema'] == HIA_ELIGIBILITY_SCHEMA
    assert result['metrics'][0]['status'] == 'deferred'
    assert result['metrics'][1]['status'] == 'unavailable'
    assert result['metrics'][1]['reason'] == 'verified_observation_binding_required'
    assert all(item['estimate'] is None and item['estimator_support'] is None
               for item in result['metrics'])
    assert result['source_graph_is_statistical_graph'] is False
    assert result['statistical_estimates_computed'] is False
    assert source.to_json() == src().to_json()


def test_node_and_branch_use_exact_su05_mapping_without_additional_samples():
    source = src()
    binding = bound(source)
    node = request(source, metrics=('node_descriptive', 'node_neighbor_associations'),
                   binding=binding)
    report = inspect_hierarchical_eligibility(source, node, binding)
    assert report['observed_rows_available'] == 3
    assert report['mapped_variable_ids'] == ['name']
    assert all(entry['status'] == 'deferred' for entry in report['metrics'])
    assert report['assumption']['independence_established_by_topology'] is False
    branch = request(source, kind='branch', node='root', metrics=('branch_joint_entropy',),
                     binding=binding)
    assert inspect_hierarchical_eligibility(source, branch, binding)['mapped_variable_ids'] == ['name', 'value']


def test_stale_source_and_binding_fingerprints_fail_closed():
    source = src()
    binding = bound(source)
    req = request(source, binding=binding, metrics=('node_descriptive',))
    other = source.to_dict(); other.pop('object_id'); other['nodes'][0]['attributes']['changed'] = True
    with pytest.raises(StructuredContractError, match='source fingerprint mismatch'):
        inspect_hierarchical_eligibility(StructuredObject.from_dict(other), req, binding)
    with pytest.raises(StructuredContractError, match='binding fingerprint mismatch'):
        inspect_hierarchical_eligibility(source, request(source, metrics=('node_descriptive',),
            binding=binding_sha256(binding) if False else None,
            ), binding)
    other_req = HierarchicalAnalysisRequest(source_sha256(source), HierarchicalScope('node', ('name',)),
                                            ('node_descriptive',), binding_digest='sha256:'+'f'*64)
    with pytest.raises(StructuredContractError, match='binding fingerprint mismatch'):
        inspect_hierarchical_eligibility(source, other_req, binding)


def test_merged_source_attribution_is_reported_unavailable_not_split():
    source = src()
    binding = bound(source, ambiguous=True)
    req = request(source, binding=binding, metrics=('node_descriptive',))
    result = inspect_hierarchical_eligibility(source, req, binding)
    assert result['ambiguous_variable_ids'] == ['name']
    assert result['metrics'][0]['reason'] == 'merged_source_node_attribution_ambiguous'


def test_level_scope_requires_reproducible_policy_and_defers_membership():
    source = src()
    binding = bound(source)
    req = request(source, kind='level', level=2, metrics=('level_joint_entropy',),
                  level_policy='rooted_depth', binding=binding)
    result = inspect_hierarchical_eligibility(source, req, binding)
    assert result['metrics'][0]['status'] == 'deferred'
    assert result['metrics'][0]['reason'] == 'HIA-02_level_assignment_required'
    assert result['source_topology'] == 'tree'
    with pytest.raises(StructuredContractError, match='requires rooted_depth'):
        inspect_hierarchical_eligibility(source, request(source, kind='level', level=2,
            metrics=('level_joint_entropy',), level_policy='dag_longest_path', binding=binding), binding)


def test_missing_or_unmappable_node_is_not_silently_matched_to_label():
    source = src()
    req = request(source, node='Name', metrics=('node_structure',))
    with pytest.raises(StructuredContractError, match='unknown original'):
        inspect_hierarchical_eligibility(source, req)


def test_conditional_entropy_and_target_cmi_require_declared_scope_and_target():
    source = src()
    binding = bound(source)
    with pytest.raises(StructuredContractError, match='conditional metrics'):
        request(source, kind='level', level=1, metrics=('level_conditional_entropy',),
                level_policy='rooted_depth', binding=binding)
    with pytest.raises(StructuredContractError, match='explicit target'):
        request(source, kind='level', level=1, metrics=('level_target_increment',),
                level_policy='rooted_depth', conditioning_scope='prior_levels', binding=binding)
    req = request(source, kind='level', level=1, metrics=('level_target_increment',),
                  level_policy='rooted_depth', conditioning_scope='prior_levels',
                  target_variable='name', binding=binding)
    assert inspect_hierarchical_eligibility(source, req, binding)['metrics'][0]['status'] == 'deferred'


def test_conditioned_factorization_requires_explicit_consent_and_prior_levels():
    source = src()
    base = dict(kind='level', level=1,
                metrics=('level_factorized_conditional_entropy',),
                level_policy='rooted_depth', conditioning_scope='prior_levels')
    with pytest.raises(StructuredContractError, match='factorized'):
        request(source, **base)
    with pytest.raises(StructuredContractError, match='explicit consent'):
        request(source, **base, independence_model='conditioned_factorization')
    req = request(source, **base, independence_model='conditioned_factorization',
                  acknowledge_assumption=True)
    assert req.independence_model == 'conditioned_factorization'
    report = inspect_hierarchical_eligibility(source, req)
    assert report['assumption']['acknowledged'] is True
    assert report['assumption']['independence_established_by_topology'] is False
    with pytest.raises(StructuredContractError, match='previous levels'):
        inspect_hierarchical_eligibility(source, request(source, kind='level', level=0,
            metrics=('level_conditional_entropy',), level_policy='rooted_depth',
            conditioning_scope='prior_levels'))


def test_scope_shape_and_metric_compatibility_fail_closed():
    source = src()
    for invalid in [
        lambda: HierarchicalScope('node'),
        lambda: HierarchicalScope('node', ('name', 'value')),
        lambda: HierarchicalScope('network', ('name',)),
        lambda: HierarchicalScope('level', level=True),
        lambda: HierarchicalScope('level', level=-1),
    ]:
        with pytest.raises(StructuredContractError):
            invalid()
    with pytest.raises(StructuredContractError, match='incompatible'):
        request(source, metrics=('level_joint_entropy',))
    with pytest.raises(StructuredContractError, match='only supported'):
        request(source, metrics=('unknown_analysis',))


def test_strict_versioned_roundtrip_rejects_duplicate_fields_and_unknown_keys():
    source = src()
    original = request(source, metrics=('node_structure', 'node_descriptive'))
    clone = HierarchicalAnalysisRequest.from_json(original.to_json())
    assert clone == original
    assert clone.to_dict()['schema'] == HIA_REQUEST_SCHEMA
    injected = original.to_dict(); injected['surprise'] = 'untrusted'
    with pytest.raises(StructuredContractError, match='schema or fields'):
        HierarchicalAnalysisRequest.from_dict(injected)
    with pytest.raises(StructuredContractError, match='duplicate'):
        HierarchicalAnalysisRequest.from_json(original.to_json().replace('"schema":', '"schema":false,"schema":', 1))
    with pytest.raises(StructuredContractError):
        HierarchicalAnalysisRequest.from_json(original.to_json().replace('"source_digest":', '"source_digest":NaN,"source_digest":', 1))


def test_level_scope_for_general_graphs_rejected_without_hierarchy_claim():
    source = src()
    payload = source.to_dict(); payload.pop('object_id')
    payload['topology'] = {'kind': 'directed_graph'}
    # A valid directed graph can happen to be acyclic; it does not assert rooted levels.
    graph = StructuredObject.from_dict(payload)
    req = request(graph, kind='level', level=1, metrics=('level_joint_entropy',),
                  level_policy='rooted_depth')
    with pytest.raises(StructuredContractError, match='unavailable'):
        inspect_hierarchical_eligibility(graph, req)
    with pytest.raises(StructuredContractError, match='rooted tree'):
        inspect_hierarchical_eligibility(graph, request(graph, kind='branch', node='root',
            metrics=('branch_structure',)))


def test_dag_level_policy_no_parent_independence_claim():
    source = src()
    payload = source.to_dict(); payload.pop('object_id')
    payload['topology'] = {'kind': 'dag'}
    # The original tree edges are a valid DAG, but depth semantics are separately requested.
    dag = StructuredObject.from_dict(payload)
    req = request(dag, kind='level', level=1, metrics=('level_joint_entropy',),
                  level_policy='dag_longest_path')
    report = inspect_hierarchical_eligibility(dag, req)
    assert report['level_policy'] == 'dag_longest_path'
    assert report['source_graph_is_statistical_graph'] is False


def test_binding_declared_variable_ids_cannot_be_guessed_from_topology():
    source = src(); binding = bound(source)
    req = request(source, kind='level', level=1,
                  metrics=('level_conditional_entropy',), level_policy='rooted_depth',
                  conditioning_scope='explicit_variables', conditioning_variables=('ghost',),
                  binding=binding)
    with pytest.raises(StructuredContractError, match='unknown SU-05'):
        inspect_hierarchical_eligibility(source, req, binding)


def test_no_mutation_and_json_report_contract():
    source = src(); binding = bound(source)
    before, bound_before = source.to_json(), binding.to_json()
    req = request(source, kind='network', metrics=('network_structure', 'network_information'),
                  binding=binding)
    first = inspect_hierarchical_eligibility(source, req, binding)
    assert first == inspect_hierarchical_eligibility(source, req, binding)
    assert json.loads(json.dumps(first, allow_nan=False)) == first
    assert source.to_json() == before and binding.to_json() == bound_before
