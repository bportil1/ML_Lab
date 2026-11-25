"""HIA-02: exact source-level navigation, no inferred information estimates."""
import json
from pathlib import Path

import pytest

from ml_lab.structured import (
    StructuredObject, StructuredContractError, HierarchicalScope,
    HierarchicalAnalysisRequest, ObservationBindingPlan, VariableBinding,
    bind_observations, binding_sha256, source_sha256,
    HIA_HIERARCHY_SCHEMA, HIA_SELECTION_SCHEMA,
    build_hierarchy_index, inspect_hierarchical_selection,
)

FIXTURES = Path(__file__).parent / 'fixtures' / 'structured_uncertainty'


def source_fixture():
    return StructuredObject.from_json((FIXTURES / 'xml_tree.json').read_text())


def graph(kind, nodes, edges):
    """Build independently validated source-only topology (no observations)."""
    return StructuredObject.from_dict({
        'schema': 'ml-lab.structured-object@1',
        'topology': {'kind': kind},
        'nodes': [{'id': name, 'kind': 'element', 'attributes': {}} for name in nodes],
        'edges': [{'id': f'e{i}', 'kind': 'link', 'source_id': a, 'target_id': b}
                  for i, (a, b) in enumerate(edges)],
        'observation_unit': 'source', 'observations': [], 'provenance': {},
    })


def selection_request(source, scope, metrics, *, binding=None, **params):
    return HierarchicalAnalysisRequest(
        source_sha256(source), scope, tuple(metrics),
        binding_digest=binding_sha256(binding) if binding else None,
        level_policy='rooted_depth' if source.topology['kind'] in ('tree', 'forest') and scope.kind == 'level'
                    else ('dag_longest_path' if scope.kind == 'level' else 'none'),
        **params,
    )


def test_rooted_tree_depths_are_exact_and_original_ids_preserved():
    source = source_fixture()
    original = source.to_json()
    index = build_hierarchy_index(source)
    assert index.level_policy == 'rooted_depth'
    assert index.topology == 'tree'
    assert index.roots == ('root',)
    assert index.levels == (('root',), ('section',), ('item',), ('name', 'value'))
    assert index.level_of('name') == 3
    assert index.parents_of('name') == ('item',)
    assert index.children_of('item') == ('name', 'value')
    assert index.ancestors_of('name') == ('root', 'section', 'item')
    assert index.descendants_of('section') == ('item', 'name', 'value')
    assert index.roots_for('value') == ('root',)
    assert index.branch_ids('item') == ('item', 'name', 'value')
    assert index.node_context('item')['descendant_count'] == 2
    assert index.prior_level_ids(3) == ('root', 'section', 'item')
    assert index.scope_ids(HierarchicalScope('network')) == ('root', 'section', 'item', 'name', 'value')
    assert source.to_json() == original
    report = index.to_dict()
    assert report['schema'] == HIA_HIERARCHY_SCHEMA
    assert report['source_digest'] == source_sha256(source)
    assert report['node_count'] == 5 and report['edge_count'] == 4
    assert report['source_edge_ids'] == sorted(e['id'] for e in source.edges)
    assert report['source_graph_is_statistical_graph'] is False
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_forest_has_distinct_root_memberships_and_exact_branch_boundaries():
    source = graph('forest', ['z', 'g', 'f', 'q', 'r'], [('z', 'g'), ('g', 'f'), ('q', 'r')])
    index = build_hierarchy_index(source)
    assert index.roots == ('q', 'z')
    assert index.levels == (('q', 'z'), ('g', 'r'), ('f',))
    assert index.roots_for('f') == ('z',)
    assert index.roots_for('r') == ('q',)
    assert index.branch_ids('z') == ('z', 'g', 'f')
    assert index.branch_ids('q') == ('q', 'r')
    assert index.parents_of('q') == ()


def test_dag_longest_path_for_multi_parent_and_skipping_edges():
    # c has direct root parent depth 0 AND indirect path via a -> b.
    source = graph('dag', ['c', 'b', 'r', 'a', 'other'], [
        ('r', 'c'), ('r', 'a'), ('a', 'b'), ('b', 'c'), ('other', 'c')])
    index = build_hierarchy_index(source)
    assert index.roots == ('other', 'r')
    assert index.levels == (('other', 'r'), ('a',), ('b',), ('c',))
    assert index.level_of('c') == 3  # longest, not shortest path
    assert index.parents_of('c') == ('b', 'other', 'r')
    assert index.roots_for('c') == ('other', 'r')
    assert index.ancestors_of('c') == ('other', 'r', 'a', 'b')
    assert index.descendants_of('r') == ('a', 'b', 'c')
    assert index.node_context('c')['parent_ids'] == ['b', 'other', 'r']
    with pytest.raises(StructuredContractError, match='branch membership'):
        index.branch_ids('r')
    req = selection_request(source, HierarchicalScope('level', level=3), ('level_joint_entropy',))
    result = inspect_hierarchical_selection(source, req, index=index)
    assert result['node_ids'] == ['c']
    assert result['prior_level_node_ids'] == ['other', 'r', 'a', 'b']
    assert result['metrics'][0]['estimate'] is None
    assert result['parent_edge_is_statistical_dependency'] is False


def test_dag_disconnected_component_and_isolated_node_are_rooted_without_invention():
    index = build_hierarchy_index(graph('dag', ['u', 'w', 'v'], [('u', 'v')]))
    assert index.roots == ('u', 'w')
    assert index.level_ids(0) == ('u', 'w')
    assert index.level_ids(1) == ('v',)
    assert index.roots_for('w') == ('w',)


def test_reordered_nodes_and_edges_keep_same_membership_without_masquerading_as_same_source():
    first = graph('dag', ['r', 'b', 'a', 'c'], [('r', 'a'), ('a', 'c'), ('r', 'b')])
    second = graph('dag', ['c', 'a', 'b', 'r'], [('r', 'b'), ('a', 'c'), ('r', 'a')])
    idx1, idx2 = build_hierarchy_index(first), build_hierarchy_index(second)
    assert idx1.levels == idx2.levels
    assert idx1.roots == idx2.roots
    assert idx1.parents_of('c') == idx2.parents_of('c')
    assert idx1.source_digest != idx2.source_digest


def test_bad_policy_unrecognized_topology_and_missing_level_fail_closed():
    source = source_fixture()
    with pytest.raises(StructuredContractError, match='requires rooted_depth'):
        build_hierarchy_index(source, level_policy='dag_longest_path')
    for topology in ('directed_graph', 'undirected_graph'):
        payload = source.to_dict()
        payload.pop('object_id')
        payload['topology'] = {'kind': topology}
        modified = StructuredObject.from_dict(payload)
        with pytest.raises(StructuredContractError, match='unavailable'):
            build_hierarchy_index(modified)
    index = build_hierarchy_index(source)
    with pytest.raises(StructuredContractError, match='does not exist'):
        index.level_ids(99)
    with pytest.raises(StructuredContractError, match='does not exist'):
        index.level_ids(True)
    with pytest.raises(StructuredContractError, match='unknown original'):
        index.ancestors_of('Name')
    with pytest.raises(StructuredContractError, match='does not exist'):
        index.scope_ids(HierarchicalScope('level', level=99))


def test_cyclic_dag_and_tree_rejected_by_structured_validation_not_repaired():
    for topology in ('tree', 'dag'):
        with pytest.raises(StructuredContractError, match='acyclic|cycle|unique parents'):
            graph(topology, ['a', 'b'], [('a', 'b'), ('b', 'a')])


def test_deep_hierarchy_avoids_recursion_and_fixed_graph_display_caps():
    n = 5201
    source = graph('tree', [f'n{i}' for i in range(n)],
                   [(f'n{i}', f'n{i + 1}') for i in range(n - 1)])
    index = build_hierarchy_index(source)
    assert index.level_count == n
    assert index.level_of(f'n{n - 1}') == n - 1
    assert len(index.ancestors_of(f'n{n - 1}')) == n - 1
    assert len(index.branch_ids('n0')) == n
    assert index.node_count == n
    with pytest.raises(TypeError):
        index._depth['n0'] = 1


def test_wide_dense_dag_source_edges_not_renumbered_or_truncated():
    nodes = ['r'] + [f'n{i}' for i in range(310)]
    edges = [('r', name) for name in nodes[1:]]
    # Dense second layer, more than 15k source edges, but small source validator.
    edges += [(f'n{i}', f'n{j}') for i in range(155) for j in range(155, 310)]
    source = graph('dag', nodes, edges)
    index = build_hierarchy_index(source)
    assert index.edge_count == len(edges) > 15000
    assert index.level_count == 3
    assert index.level_ids(2) == tuple(sorted(f'n{i}' for i in range(155,310)))
    assert index.node_count == len(nodes)


def test_binding_scope_level_deduplicates_and_rejects_cross_level_attribution():
    source = source_fixture()
    plan = ObservationBindingPlan(
        source_sha256(source), source.observation_unit,
        (VariableBinding('name', ('name',)), VariableBinding('value', ('value',))),
    )
    binding = bind_observations(source, plan)
    req = selection_request(source, HierarchicalScope('level', level=3),
                            ('level_joint_entropy',), binding=binding)
    report = inspect_hierarchical_selection(source, req, binding)
    assert report['schema'] == HIA_SELECTION_SCHEMA
    assert report['node_ids'] == ['name', 'value']
    assert report['mapped_variable_ids'] == ['name', 'value']
    assert report['variable_source_node_ids'] == {'name': ['name'], 'value': ['value']}
    assert report['prior_level_node_ids'] == ['root', 'section', 'item']
    assert report['metrics'][0]['status'] == 'deferred'
    assert report['metrics'][0]['reason'] == 'HIA_scientific_estimator_not_implemented'
    assert report['statistical_estimates_computed'] is False
    assert json.loads(json.dumps(report)) == report
    assert report == inspect_hierarchical_selection(source, req, binding)

    # SU-05 explicitly declares a variable whose identity merges two source
    # nodes in different levels: do not allocate it to either as two samples.
    merged = bind_observations(source, ObservationBindingPlan(
        source_sha256(source), source.observation_unit,
        (VariableBinding('name', ('name', 'item'), merged_sources=True),),
    ))
    req_merged = selection_request(source, HierarchicalScope('level', level=3),
                                   ('level_joint_entropy',), binding=merged)
    report_merged = inspect_hierarchical_selection(source, req_merged, merged)
    assert report_merged['mapped_variable_ids'] == ['name']
    assert report_merged['ambiguous_variable_ids'] == ['name']
    assert report_merged['cross_level_binding_variable_ids'] == ['name']
    assert report_merged['metrics'][0]['status'] == 'unavailable'


def test_empty_binding_scope_marked_unavailable_and_no_assumed_observations():
    source = source_fixture()
    binding = bind_observations(source, ObservationBindingPlan(
        source_sha256(source), source.observation_unit,
        (VariableBinding('name', ('name',)),),
    ))
    req = selection_request(source, HierarchicalScope('level', level=0),
                            ('level_joint_entropy',), binding=binding)
    result = inspect_hierarchical_selection(source, req, binding)
    assert result['mapped_variable_ids'] == []
    assert result['metrics'][0]['status'] == 'unavailable'
    assert result['metrics'][0]['reason'] == 'no_variables_bound_to_selected_scope'


def test_source_only_selection_stays_structural_and_no_fabricated_estimates():
    source = source_fixture()
    req = selection_request(source, HierarchicalScope('branch', ('item',)), ('branch_structure',))
    result = inspect_hierarchical_selection(source, req)
    assert result['node_ids'] == ['item', 'name', 'value']
    assert result['mapped_variable_ids'] == []
    assert result['metrics'][0]['status'] == 'deferred'
    assert result['metrics'][0]['estimate'] is None


def test_stale_index_source_and_binding_rejected():
    source = source_fixture()
    binding = bind_observations(source, ObservationBindingPlan(
        source_sha256(source), source.observation_unit,
        (VariableBinding('name', ('name',)),),
    ))
    req = selection_request(source, HierarchicalScope('node', ('name',)),
                            ('node_descriptive',), binding=binding)
    stale_source = source.to_dict()
    stale_source.pop('object_id')
    stale_source['nodes'][0]['attributes'] = {'changed': 1}
    other = StructuredObject.from_dict(stale_source)
    with pytest.raises(StructuredContractError, match='source fingerprint mismatch'):
        inspect_hierarchical_selection(other, req, binding)
    with pytest.raises(StructuredContractError, match='fingerprint mismatch'):
        inspect_hierarchical_selection(source, req, binding, index=build_hierarchy_index(other))
    with pytest.raises(StructuredContractError, match='binding fingerprint mismatch'):
        inspect_hierarchical_selection(source, req, bind_observations(source, ObservationBindingPlan(
            source_sha256(source), source.observation_unit,
            (VariableBinding('value', ('value',)),))),
                                    index=build_hierarchy_index(source))
