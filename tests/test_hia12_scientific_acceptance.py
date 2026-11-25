"""HIA-12: cross-stage acceptance using one fully local structured source.

No external network, source repository, PAH host or scientific database is
needed. Expected bit counts follow from an exactly balanced finite population.
"""
from __future__ import annotations

import json

import pytest

from ml_lab.structured import (
    StructuredObject, source_sha256, atomic_node_statistics,
    analyze_neighbor_relationships, analyze_hierarchical_branch,
    analyze_hierarchical_level, analyze_hierarchical_conditional_level,
    analyze_hierarchical_factorized_level, analyze_hierarchical_network_accounting,
)
from ml_lab.ui.structured_hia_view import hierarchy_overview, inspect_workspace_scope

REVISION = 'sha256:' + 'f' * 64


def mapping(node, column):
    return {'node_id': node, 'variable_id': column}


def source(*, dependent=False, missing=False):
    """Three structural levels; Z, X, W independent unless dependent=True.

    Y = X XOR W. A structural source edge connects a to leaf but X and Y
    need not have any measured statistical association.
    """
    records = []
    for repeat in range(12):
        for z in (0, 1):
            for x in (0, 1):
                for w in (0, 1):
                    records.append((z, x, x if dependent else w, x ^ (x if dependent else w)))
    if missing:
        records = [(z, x, w, None if index % 4 == 0 else y)
                   for index, (z, x, w, y) in enumerate(records)]
    observations = []
    for i, row in enumerate(records):
        values = {}
        for key, value in zip(('z', 'x', 'w', 'y'), row):
            values[key] = ({'status': 'present', 'value': value} if value is not None
                           else {'status': 'explicit_null', 'value': None,
                                 'source_reason': 'path_absent'})
        observations.append({'observation_id': f'obs-{i}', 'values': values})
    return StructuredObject.from_dict({
        'schema': 'ml-lab.structured-object@1',
        'topology': {'kind': 'tree'},
        'nodes': [{'id': name, 'kind': 'atom', 'attributes': {}}
                  for name in ('root', 'a', 'b', 'leaf', 'unmapped')],
        'edges': [{'id': name, 'kind': 'source_link', 'source_id': first,
                   'target_id': second} for name, first, second in (
                       ('e0', 'root', 'a'), ('e1', 'root', 'b'),
                       ('e2', 'a', 'leaf'), ('e3', 'b', 'unmapped'))],
        'observation_unit': 'record', 'observations': observations, 'provenance': {},
    })


def test_all_hia_scales_cohere_and_source_is_unchanged():
    obj = source()
    original = obj.to_json()
    digest = source_sha256(obj)
    root = mapping('root', 'z')
    a, b, leaf = mapping('a', 'x'), mapping('b', 'w'), mapping('leaf', 'y')

    overview = hierarchy_overview(obj, source_revision=REVISION)
    assert overview['available'] and [v['node_count'] for v in overview['levels']] == [1, 2, 2]
    selection = inspect_workspace_scope(obj, source_revision=REVISION,
        scope_input={'kind': 'level', 'node_ids': [], 'level': 1})
    assert selection['selection']['node_ids'] == ['a', 'b']
    assert selection['selection']['prior_level_node_ids'] == ['root']

    atom = atomic_node_statistics(obj, node_id='a', variable_id='x')
    assert atom['source_digest'] == digest
    assert atom['entropy']['status'] == 'ok'
    assert atom['entropy']['bits'] == pytest.approx(1)

    # Branch root 'a' contains 'a' and 'leaf'. The pair is independent here
    # even though the source topology contains a directed source edge.
    branch = analyze_hierarchical_branch(obj, branch_root_id='a',
                                          mappings=[a, leaf], context=root)
    assert branch['branch_node_ids'] == ['a', 'leaf']
    assert branch['branch_information']['joint_entropy_bits'] == pytest.approx(2)
    assert branch['branch_information']['total_correlation_bits'] == pytest.approx(0)
    assert branch['context_information']['branch_context_mutual_information_bits'] == pytest.approx(0)

    level = analyze_hierarchical_level(obj, level=1, mappings=[a, b])
    stats = level['level_information']
    assert level['level_node_ids'] == ['a', 'b']
    assert stats['joint_entropy_bits'] == pytest.approx(2)
    assert stats['total_correlation_bits'] == pytest.approx(0)

    conditioned = analyze_hierarchical_conditional_level(obj, level=1,
        mappings=[a, b], preceding_mappings=[root], target=leaf)
    joint = conditioned['level_conditioned_information']
    assert joint['level_conditional_entropy_bits'] == pytest.approx(2)
    assert joint['level_prior_mutual_information_bits'] == pytest.approx(0)
    assert conditioned['target_increment_information']['target_incremental_information_bits'] == pytest.approx(1)

    factorized = analyze_hierarchical_factorized_level(obj, level=1,
        mappings=[a, b], preceding_mappings=[root],
        assumption_confirmed=True, factorization_mode='shared_joint_prior', parent_sets={})
    assert factorized['model']['empirical_model_kl_divergence_bits'] == pytest.approx(0, abs=1e-10)

    whole = analyze_hierarchical_network_accounting(obj,
        mappings=[root, a, b], target=leaf)
    info = whole['network_information']
    assert info['status'] == 'ok' and info['support'] == 96
    assert info['joint_entropy_bits'] == pytest.approx(3)
    assert info['total_correlation_bits'] == pytest.approx(0)
    assert [v['incremental_joint_entropy_given_prior_bits']
            for v in info['levels']] == pytest.approx([1, 2])
    assert [v['target_incremental_information_bits']
            for v in info['levels']] == pytest.approx([0, 1])
    assert sum(v['incremental_joint_entropy_given_prior_bits']
               for v in info['levels']) == pytest.approx(info['joint_entropy_bits'])
    assert sum(v['target_incremental_information_bits']
               for v in info['levels']) == pytest.approx(info['target_mutual_information_bits'])
    assert whole['unmapped_node_count'] == 2
    for report in (atom, branch, level, conditioned, factorized, whole):
        assert report['source_digest'] == digest
        assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert obj.to_json() == original


def test_empirical_redundancy_agrees_with_factorization_divergence():
    obj = source(dependent=True)
    a, b, root = mapping('a', 'x'), mapping('b', 'w'), mapping('root', 'z')
    level = analyze_hierarchical_level(obj, level=1, mappings=[a, b])
    factor = analyze_hierarchical_factorized_level(obj, level=1,
        mappings=[a, b], preceding_mappings=[root], assumption_confirmed=True,
        factorization_mode='shared_joint_prior', parent_sets={})
    assert level['level_information']['joint_entropy_bits'] == pytest.approx(1)
    assert level['level_information']['total_correlation_bits'] == pytest.approx(1)
    model = factor['model']
    assert model['empirical_model_kl_divergence_bits'] == pytest.approx(1)
    assert model['empirical_conditional_total_correlation_bits'] == pytest.approx(1)
    assert model['omitted_prior_information_bits'] == pytest.approx(0)


def test_missing_target_uses_one_common_network_cohort_and_preserves_other_support():
    obj = source(missing=True)
    a, b, root, leaf = (mapping('a', 'x'), mapping('b', 'w'),
                        mapping('root', 'z'), mapping('leaf', 'y'))
    level = analyze_hierarchical_level(obj, level=1, mappings=[a, b])
    network = analyze_hierarchical_network_accounting(obj, mappings=[root, a, b], target=leaf)
    assert level['level_information']['support'] == 96
    assert network['network_information']['support'] == 72
    assert network['network_information']['excluded_rows'] == 24
    assert sum(x['incremental_joint_entropy_given_prior_bits']
               for x in network['network_information']['levels']) == pytest.approx(
                   network['network_information']['joint_entropy_bits'])
    assert network['network_information']['finite_sample_bias_warning'] is True


def test_large_hierarchy_keeps_exact_inventory_and_deferred_unmapped_statistics():
    n = 5103
    raw = source().to_dict()
    raw['nodes'] = [{'id': f'v{i}', 'kind': 'atom', 'attributes': {}} for i in range(n)]
    raw['edges'] = [{'id': f'e{i}', 'kind': 'source_link', 'source_id': 'v0',
                     'target_id': f'v{i}'} for i in range(1, n)]
    obj = StructuredObject.from_dict(raw)
    result = analyze_hierarchical_network_accounting(obj,
        mappings=[mapping('v0', 'z'), mapping('v5102', 'x')])
    assert result['structural_node_count'] == n
    assert result['unmapped_node_count'] == n - 2
    assert result['structural_level_inventory'][1]['node_count'] == n - 1
    assert result['network_information']['status'] == 'ok'
    assert result['statistical_scope'] == 'explicit_mapped_network_subset'


def test_dag_graph_over_old_5000_15000_limits_is_exhaustive():
    from ml_lab.ui.structured_graph_view import make_shared_graph_view

    n = 5101
    arcs = ([(0, i) for i in range(1, n)]
            + [(i, i+1) for i in range(1, n-1)]
            + [(i, i+2) for i in range(1, n-2)])
    raw = {
        'schema': 'ml-lab.structured-object@1', 'topology': {'kind': 'dag'},
        'nodes': [{'id': f'v{i}', 'kind': 'atom', 'attributes': {}} for i in range(n)],
        'edges': [{'id': f'e{k}', 'kind': 'source_link',
                   'source_id': f'v{a}', 'target_id': f'v{b}'}
                  for k,(a,b) in enumerate(arcs)],
        'observation_unit': 'source', 'observations': [], 'provenance': {},
    }
    obj = StructuredObject.from_dict(raw)
    graph = make_shared_graph_view(obj, fingerprint='f'*64)
    overview = hierarchy_overview(obj, source_revision=REVISION)
    assert len(graph['nodes']) == n > 5000
    assert len(graph['edges']) == len(arcs) > 15000
    assert len({x['id'] for x in graph['nodes']}) == n
    assert len({x['id'] for x in graph['edges']}) == len(arcs)
    assert graph['display']['coverage'] == {
        'scope': 'exhaustive', 'total_nodes': n, 'total_edges': len(arcs)}
    assert overview['available'] and overview['level_policy'] == 'dag_longest_path'
    assert overview['node_count'] == n and overview['edge_count'] == len(arcs)
    assert not overview['branch_selection_allowed']
    assert overview['levels'][-1]['depth'] == n-1
    # Observation-free structural sources cannot be presented as measured MI.
    scoped = inspect_workspace_scope(obj, source_revision=REVISION,
        scope_input={'kind': 'level', 'node_ids': [], 'level': n-1})
    assert scoped['selection']['node_ids'] == [f'v{n-1}']
    assert all(metric['estimate'] is None for metric in scoped['selection']['metrics'])
