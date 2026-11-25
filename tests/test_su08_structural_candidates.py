"""SU-08: deterministic, non-destructive structural transformation candidates."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import json

import pytest

from ml_lab.structured import (
    ObservationBindingPlan, StructuredContractError, StructuredObject,
    StructuralCandidatePlan, VariableBinding, apply_structural_candidate,
    enumerate_structural_candidates, replay_structural_candidate, source_sha256,
)
from ml_lab.structured.branch_uncertainty import binding_sha256
from ml_lab.structured.bindings import bind_observations

FIXTURES = Path(__file__).parent / 'fixtures' / 'structured_uncertainty'


def fixture(name='xml_tree'):
    return StructuredObject.from_json((FIXTURES / f'{name}.json').read_text())


def binding_for(source, *rules, group_vars=(), weight=None):
    return ObservationBindingPlan(source_sha256(source), source.observation_unit,
        tuple(VariableBinding(*rule) for rule in rules),
        group_variables=group_vars, weight_variable=weight)


def plan_for(source, operation, *, binding=None, **kwargs):
    digest = binding_sha256(bind_observations(source, binding)) if binding else None
    return StructuralCandidatePlan(source_sha256(source), operation,
                                   binding_digest=digest, **kwargs)


def apply(source, op, *, binding=None, **kwargs):
    return apply_structural_candidate(source, plan_for(source, op, binding=binding, **kwargs),
                                      binding=binding)


def test_preserve_is_source_identity_and_replay_is_deterministic():
    obj = fixture()
    before = obj.to_json()
    result = apply(obj, 'preserve')
    assert result.object is obj
    assert result.manifest['source_digest'] == source_sha256(obj)
    assert result.manifest['output_digest'] == source_sha256(obj)
    assert result.manifest['lossless_representation']
    assert result.manifest['output_alone_can_reconstruct_source']
    assert result.manifest['feature_impact']['observed_cell_values_changed'] is False
    assert result.manifest['feature_impact']['observation_count'] == 3
    assert replay_structural_candidate(obj, result.manifest).to_json() == result.to_json()
    assert replay_structural_candidate(obj, json.dumps(result.manifest)).to_json() == result.to_json()
    assert obj.to_json() == before
    assert json.dumps(result.to_dict(), allow_nan=False)


def test_prune_removes_entire_explicit_subtree_without_removing_observations():
    obj = fixture()
    result = apply(obj, 'prune', node_ids=('item',))
    assert {node['id'] for node in result.object.nodes} == {'root', 'section'}
    assert {edge['id'] for edge in result.object.edges} == {'e1'}
    assert result.manifest['structural_loss']['removed_node_ids'] == ['item', 'name', 'value']
    assert result.manifest['structural_loss']['removed_edge_count'] == 3
    assert result.manifest['feature_impact']['dropped_variable_ids'] == []
    assert result.object.observations == obj.observations
    assert result.object.topology['kind'] == 'tree'
    assert not result.manifest['lossless_representation']
    assert not result.manifest['output_alone_can_reconstruct_source']
    assert result.manifest['feature_impact']['empirical_information_loss_estimated'] is False
    assert replay_structural_candidate(obj, result.manifest).to_json() == result.to_json()
    assert 'structural_candidates' not in obj.provenance


def test_summarize_keeps_selected_boundary_but_only_counts_structure():
    obj = fixture()
    result = apply(obj, 'summarize', node_ids=('item',))
    assert {n['id'] for n in result.object.nodes} == {'root', 'section', 'item'}
    item = next(n for n in result.object.nodes if n['id'] == 'item')
    summary = item['metadata']['ml_lab_su08_structural_summary_v1']
    assert summary['removed_nodes'] == 2
    assert summary['removed_edges'] == 2
    assert summary['value_aggregation'] == 'none'
    assert result.manifest['structural_loss']['collapsed_into'] == {'name': 'item', 'value': 'item'}
    assert result.manifest['structural_summary_only'] is True
    assert result.object.observations == obj.observations
    assert apply(obj, 'summarize', node_ids=('item',)).to_json() == result.to_json()
    assert replay_structural_candidate(obj, result.manifest).to_json() == result.to_json()


def test_summarize_boundary_collision_refuses_overwrite():
    raw = fixture().to_dict()
    raw.pop('object_id')
    raw['nodes'][2]['metadata'] = {'ml_lab_su08_structural_summary_v1': {'custom': 1}}
    obj = StructuredObject.from_dict(raw)
    with pytest.raises(StructuredContractError, match='collision'):
        apply(obj, 'summarize', node_ids=('item',))


def test_aggregate_existing_legacy_behavior_and_explicit_counts():
    obj = fixture()
    result = apply(obj, 'aggregate', max_depth=1)
    assert {n['id'] for n in result.object.nodes} == {'root', 'section'}
    section = next(n for n in result.object.nodes if n['id'] == 'section')
    assert section['metadata']['ml_lab_structural_aggregate_v1']['removed_nodes'] == 3
    assert result.manifest['structural_loss']['removed_node_count'] == 3
    assert result.manifest['structural_loss']['collapsed_into'] == {
        'item': 'section', 'name': 'section', 'value': 'section'}
    assert result.manifest['structural_summary_only']
    assert result.object.observations == obj.observations


def test_depth_reduce_retains_boundary_and_keeps_original_rows():
    obj = fixture()
    result = apply(obj, 'depth_reduce', max_depth=2)
    assert result.manifest['structural_loss']['removed_node_ids'] == ['name', 'value']
    assert result.manifest['feature_impact']['output_variable_ids'] == ['name', 'value']
    assert result.object.observations == obj.observations
    assert not result.manifest['structural_summary_only']


@pytest.mark.parametrize('name', ['xml_tree', 'shared_child_dag', 'cyclic_network'])
def test_flatten_only_removes_edges_never_joins_observations(name):
    obj = fixture(name)
    result = apply(obj, 'flatten')
    assert len(result.object.nodes) == len(obj.nodes)
    assert len(result.object.edges) == 0
    assert len(result.object.observations) == len(obj.observations)
    assert result.manifest['structural_loss']['removed_node_count'] == 0
    assert result.manifest['structural_loss']['removed_edge_count'] == len(obj.edges)
    assert result.manifest['feature_impact']['dropped_variable_count'] == 0
    assert result.manifest['feature_impact']['observation_join_performed'] is False
    assert replay_structural_candidate(obj, result.manifest).to_json() == result.to_json()


@pytest.mark.parametrize('name', ['shared_child_dag', 'cyclic_network'])
@pytest.mark.parametrize('operation,params', [
    ('prune', {'node_ids': ('a',)}),
    ('summarize', {'node_ids': ('a',)}),
    ('aggregate', {'max_depth': 1}),
    ('depth_reduce', {'max_depth': 1}),
])
def test_nonhierarchy_reduction_rejects_instead_of_inventing_tree(name, operation, params):
    obj = fixture(name)
    existing = obj.to_json()
    if operation in ('prune', 'summarize'):
        params = {'node_ids': (obj.nodes[0]['id'],)}
    with pytest.raises(StructuredContractError, match='rooted_hierarchy'):
        apply(obj, operation, **params)
    assert obj.to_json() == existing


def test_disjoint_prune_can_produce_valid_forest():
    doc = fixture().to_dict()
    doc.pop('object_id')
    doc['nodes'].append({'id': 'other_root', 'kind': 'uninterpreted', 'attributes': {}})
    doc['topology'] = {'kind': 'forest'}
    obj = StructuredObject.from_dict(doc)
    result = apply(obj, 'prune', node_ids=('root',))
    assert [n['id'] for n in result.object.nodes] == ['other_root']
    assert result.object.topology['kind'] == 'tree'


def test_prune_entire_graph_and_overlapping_selection_fail_closed():
    obj = fixture()
    with pytest.raises(StructuredContractError, match='remove every'):
        apply(obj, 'prune', node_ids=('root',))
    with pytest.raises(StructuredContractError, match='overlap'):
        apply(obj, 'prune', node_ids=('section', 'value'))
    with pytest.raises(StructuredContractError, match='unknown'):
        apply(obj, 'summarize', node_ids=('typo',))


def test_bound_variable_risk_report_without_auto_observation_loss():
    obj = fixture()
    bound = binding_for(obj, ('name', ('name',)), ('value', ('value',)))
    result = apply(obj, 'prune', binding=bound, node_ids=('name',))
    assert result.manifest['feature_impact']['fully_removed_source_mapping_variables'] == ['name']
    assert result.manifest['feature_impact']['source_nodes_removed_for_bound_variables'] == ['name']
    assert result.manifest['feature_impact']['dropped_variable_ids'] == []
    assert result.manifest['binding_digest'] == binding_sha256(bind_observations(obj, bound))
    assert result.object.observations == obj.observations
    assert replay_structural_candidate(obj, result.manifest, binding=bound).to_json() == result.to_json()
    with pytest.raises(StructuredContractError, match='binding fingerprint'):
        replay_structural_candidate(obj, result.manifest)


def test_explicit_drop_exclusive_only_drops_fully_removed_bound_columns():
    obj = fixture()
    bound = binding_for(obj, ('name', ('name',)), ('value', ('value',)))
    result = apply(obj, 'prune', binding=bound, node_ids=('name',),
                   observation_policy='drop_exclusive')
    assert result.manifest['feature_impact']['dropped_variable_ids'] == ['name']
    assert result.manifest['feature_impact']['output_variable_ids'] == ['value']
    assert [r['observation_id'] for r in result.object.observations] == ['r1', 'r2', 'r3']
    assert result.object.observations[1]['values']['value']['source_reason'] == 'path_absent'
    assert replay_structural_candidate(obj, result.manifest, binding=bound).to_json() == result.to_json()
    assert obj.observations[0]['values']['name']['value'] == 'Alpha'


def test_multisource_binding_never_partially_assigns_variable_to_pruned_node():
    obj = fixture()
    bound = binding_for(obj, ('name', ('name', 'value'), 'preserve', True),
                        ('value', ('value',)))
    result = apply(obj, 'prune', binding=bound, node_ids=('name',),
                   observation_policy='drop_exclusive')
    impact = result.manifest['feature_impact']
    assert impact['mixed_retained_removed_source_mapping_variables'] == ['name']
    assert impact['fully_removed_source_mapping_variables'] == []
    assert impact['dropped_variable_ids'] == []
    assert result.object.observations == obj.observations


def test_unbound_column_is_never_silently_attributed_or_deleted():
    obj = fixture()
    bound = binding_for(obj, ('name', ('name',)))
    result = apply(obj, 'prune', binding=bound, node_ids=('name',),
                   observation_policy='drop_exclusive')
    assert result.manifest['feature_impact']['unbound_variable_ids'] == ['value']
    assert result.manifest['feature_impact']['output_variable_ids'] == ['value']


def test_candidate_can_not_drop_all_observed_variables():
    obj = fixture()
    bound = binding_for(obj, ('name', ('name',)), ('value', ('value',)))
    with pytest.raises(StructuredContractError, match='all aligned variables'):
        apply(obj, 'prune', binding=bound, node_ids=('item',), observation_policy='drop_exclusive')


def test_selected_group_variable_is_protected_against_drop():
    obj = fixture()
    bound = binding_for(obj, ('name', ('name',)), group_vars=('name',))
    with pytest.raises(StructuredContractError, match='group or weight'):
        apply(obj, 'prune', binding=bound, node_ids=('name',), observation_policy='drop_exclusive')


def test_unobserved_graph_is_not_treated_as_implicit_statistical_samples():
    obj = fixture('cyclic_network')
    result = apply(obj, 'preserve')
    assert result.manifest['feature_impact']['observation_count'] == 0
    assert result.manifest['feature_impact']['input_variable_ids'] == []
    bound = binding_for(obj, ('synthetic', (obj.nodes[0]['id'],)))
    with pytest.raises(StructuredContractError, match='aligned observations'):
        apply(obj, 'flatten', binding=bound, observation_policy='drop_exclusive')


def test_candidate_schema_rejects_stale_sources_invalid_types_and_unsupported_fields():
    obj = fixture()
    plan = plan_for(obj, 'prune', node_ids=('name',))
    assert StructuralCandidatePlan.from_json(plan.to_json()) == plan
    with pytest.raises(StructuredContractError, match='duplicate'):
        StructuralCandidatePlan.from_json(plan.to_json().replace('"schema":', '"schema":"duplicate","schema":', 1))
    with pytest.raises(StructuredContractError, match='schema or fields'):
        StructuralCandidatePlan.from_dict({**plan.to_dict(), 'unexpected': True})
    changed = obj.to_dict()
    changed.pop('object_id')
    changed['nodes'][0]['attributes']['changed'] = True
    with pytest.raises(StructuredContractError, match='source fingerprint mismatch'):
        apply_structural_candidate(StructuredObject.from_dict(changed), plan)
    with pytest.raises(StructuredContractError, match='nonnegative'):
        StructuralCandidatePlan(source_sha256(obj), 'aggregate', max_depth=True)
    with pytest.raises(StructuredContractError, match='binding_digest'):
        StructuralCandidatePlan(source_sha256(obj), 'prune', ('name',), observation_policy='drop_exclusive')
    with pytest.raises(StructuredContractError, match='node_ids'):
        StructuralCandidatePlan(source_sha256(obj), 'flatten', node_ids=('name',))
    with pytest.raises(StructuredContractError, match='unique'):
        StructuralCandidatePlan(source_sha256(obj), 'prune', node_ids=('name', 'name'))
    with pytest.raises(FrozenInstanceError):
        plan.operation = 'flatten'


def test_replay_detects_tampering_to_manifest_and_output_digest():
    obj = fixture()
    result = apply(obj, 'summarize', node_ids=('item',))
    tampered = result.manifest
    tampered['output_digest'] = source_sha256(obj)
    with pytest.raises(StructuredContractError, match='manifest mismatch'):
        replay_structural_candidate(obj, tampered)
    with pytest.raises(StructuredContractError, match='duplicate'):
        replay_structural_candidate(obj, '{"schema":1,"schema":2}')
    assert result.manifest['output_digest'] != source_sha256(obj)


def test_deterministic_explicit_bounded_candidate_enumeration():
    obj = fixture()
    kw = dict(node_ids=('value', 'name'), depths=(2, 1, 2))
    plans = enumerate_structural_candidates(obj, **kw)
    assert [(p.operation, p.node_ids, p.max_depth) for p in plans] == [
        ('preserve', (), None), ('prune', ('name',), None), ('prune', ('value',), None),
        ('aggregate', (), 1), ('aggregate', (), 2),
        ('depth_reduce', (), 1), ('depth_reduce', (), 2),
        ('summarize', ('name',), None), ('summarize', ('value',), None),
        ('flatten', (), None),
    ]
    assert [p.to_json() for p in plans] == [p.to_json() for p in enumerate_structural_candidates(obj, **kw)]
    for p in plans:
        assert apply_structural_candidate(obj, p).manifest['candidate_id'].startswith('sha256:')
    with pytest.raises(StructuredContractError, match='maximum'):
        enumerate_structural_candidates(obj, **kw, max_candidates=2)
    with pytest.raises(StructuredContractError, match='selected node IDs'):
        enumerate_structural_candidates(obj, node_ids=('typo',))


def test_nonhierarchical_enumeration_offers_only_semantically_valid_modes():
    obj = fixture('shared_child_dag')
    modes = [p.operation for p in enumerate_structural_candidates(obj,
        node_ids=(obj.nodes[0]['id'],), depths=(1,))]
    assert modes == ['preserve', 'flatten']
    assert apply(obj, 'flatten').object.topology['kind'] == 'directed_graph'


def test_projected_variables_keep_repeated_cells_and_provenance_unchanged():
    obj = fixture()
    doc = obj.to_dict()
    doc.pop('object_id')
    doc['observations'][0]['values']['name'] = {
        'status': 'multiple', 'value': None,
        'occurrences': [{'status': 'present', 'value': 'A'},
                        {'status': 'explicit_null', 'value': None, 'source_reason': 'not_present'}]}
    obj = StructuredObject.from_dict(doc)
    bound = binding_for(obj, ('name', ('name',)), ('value', ('value',)))
    result = apply(obj, 'prune', binding=bound, node_ids=('value',),
                   observation_policy='drop_exclusive')
    assert result.object.observations[0]['values']['name'] == obj.observations[0]['values']['name']
    assert result.manifest['feature_impact']['observation_count'] == 3


def test_binding_fingerprint_refuses_different_plan_even_with_same_source():
    obj = fixture()
    b1 = binding_for(obj, ('name', ('name',)))
    b2 = binding_for(obj, ('name', ('item',)))
    plan = plan_for(obj, 'prune', node_ids=('name',), binding=b1)
    with pytest.raises(StructuredContractError, match='binding fingerprint mismatch'):
        apply_structural_candidate(obj, plan, binding=b2)


def test_tree_root_is_not_enumerated_for_impossible_pruning():
    obj = fixture()
    plans = enumerate_structural_candidates(obj, node_ids=('root',),
        operations=('prune', 'summarize'))
    assert [p.operation for p in plans] == ['summarize']
    assert apply_structural_candidate(obj, plans[0]).object.topology['kind'] == 'tree'


def test_bad_operation_types_fail_with_contract_error():
    obj = fixture()
    with pytest.raises(StructuredContractError, match='operation'):
        StructuralCandidatePlan(source_sha256(obj), [])
    with pytest.raises(StructuredContractError, match='observation_policy'):
        StructuralCandidatePlan(source_sha256(obj), 'preserve', observation_policy=[])
    with pytest.raises(StructuredContractError, match='operations'):
        enumerate_structural_candidates(obj, operations=([],))
