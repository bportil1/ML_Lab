"""ML-UW-37 local cross-module *artifact* integration; no external services."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml_lab.structured import StructuredContractError, StructuredObject, describe_structure
from ml_lab.structured.adapters import (
    AST_HANDOFF_SCHEMA, PYPIQUE_SCHEMA, SHARED_GRAPH_SCHEMA,
    HierarchyHandoffAdapter, PyPIQUEConcludedAdapter, cross_module_adapters, default_adapters,
)
from ml_lab.ui.structured_workbench import KINDS, read_source, analyze


def hierarchy():
    return {
        'schema': AST_HANDOFF_SCHEMA,
        'source_id': 'code-analyzer-export:fixture-1',
        'provenance': {'language': 'java', 'parser': 'synthetic-test-only'},
        'roots': ['unit'],
        'nodes': [
            {'id': 'unit', 'kind': 'compilation_unit', 'attributes': {'language': 'java'},
             'source_ref': {'path': 'src/Test.java', 'line': 1, 'column': 1}},
            {'id': 'method', 'kind': 'method_decl', 'attributes': {'name': 'run'},
             'source_ref': {'path': 'src/Test.java', 'line': 2}},
            {'id': 'return', 'kind': 'return_stmt', 'attributes': {'text': 'return 1;'},
             'source_ref': {'path': 'src/Test.java', 'line': 3}},
        ],
        'edges': [
            {'id': 'e1', 'parent': 'unit', 'child': 'method', 'kind': 'syntax_child', 'ordinal': 0},
            {'id': 'e2', 'parent': 'method', 'child': 'return', 'kind': 'syntax_child', 'ordinal': 0},
        ],
    }


def py_projection(*, mixed=False, backing=False):
    raw_main = {
        'graph_role': 'recorded_signed_information_and_reporting',
        'title': 'Recorded information / reporting',
        'interpretation': 'Information associations are not causal or probabilistic arcs.',
        'visible_measure_count': 1,
        'nodes': [
            {'id': 'feature:89', 'kind': 'measure', 'label': 'CWE-89', 'layer': 1,
             'unit': 'recorded_visible_measure', 'value': None,
             'details': {'visible_measure_identity': 'CWE-89', 'acceptance': 0.4}},
            {'id': 'latent:1:0', 'kind': 'latent', 'label': 'H0', 'layer': 2,
             'unit': 'recorded_signed_information_structure', 'details': {}},
            {'id': 'pf:reliability', 'kind': 'product_factor', 'label': 'Reliability',
             'layer': 3, 'details': {'membership': ['CWE-89']}},
        ],
        'edges': [
            {'id': '0:feature:89:latent:1:0', 'source': 'feature:89', 'target': 'latent:1:0',
             'relationship': 'signed_information',
             'semantics': 'recorded_signed_information_association_not_causal',
             'directed': True, 'details': {'mutual_information': 0.18}},
        ],
    }
    if mixed:
        raw_main['edges'].append({
            'id': '1:latent:1:0:pf:reliability', 'source': 'latent:1:0',
            'target': 'pf:reliability', 'relationship': 'recorded_information_attribution_trace',
            'semantics': 'derived_display_path_not_learned_edge',
            'directed': False, 'details': {'trace_only': True},
        })
    raw_backing = {
        'graph_role': 'immutable_learned_probabilistic_backing',
        'title': 'Learned backing', 'architecture_audit': {'visible_count': 1},
        'nodes': [
            {'id': 'feature:89', 'kind': 'measure', 'label': 'CWE-89', 'layer': 0},
            {'id': 'latent:1:0', 'kind': 'latent', 'label': 'H0', 'layer': 1},
        ],
        'edges': [{
            'id': '0:feature:89:latent:1:0', 'source': 'feature:89', 'target': 'latent:1:0',
            'relationship': 'learned_rbm_connection', 'directed': False,
            'semantics': 'encoder_weight_not_mutual_information',
            'details': {'weight': -2.3},
        }],
    }
    return {
        'schema': PYPIQUE_SCHEMA, 'schema_version': 1,
        'family': 'mi_informed_pique', 'immutable': True,
        'version_id': 'version-1', 'archive_sha256': 'a' * 64,
        'manifest_sha256': 'b' * 64, 'choice': 'something',
        'provenance': {'source_type': 'local-test-export'},
        'graph': raw_main,
        'learned_probabilistic_backing': raw_backing if backing else None,
    }


def test_registry_is_opt_in_and_original_registry_unchanged():
    assert default_adapters().names == ('json', 'node_edge', 'xml_structure')
    first, second = cross_module_adapters(), cross_module_adapters()
    assert first is not second
    assert {'hierarchy_handoff', 'pypique_graph', 'pypique_backing',
            'pypique_graph_directed', 'pypique_graph_undirected',
            'shared_graph'} <= set(first.names)
    assert set(KINDS) == {'structured', *first.names}


def test_ast_handoff_tree_preserves_spans_ordinals_and_provenance():
    result = HierarchyHandoffAdapter().adapt(hierarchy())
    obj = result.require_lossless()
    assert result.warnings and obj.topology['kind'] == 'tree'
    assert 'rooted_hierarchy' in obj.capabilities
    assert 'aligned_observations' not in obj.capabilities
    assert obj.observation_unit == 'graph' and not obj.observations
    assert obj.nodes[0]['source_ref']['path'] == 'src/Test.java'
    assert obj.edges[1]['ordinal'] == 0
    assert obj.provenance['source_provenance']['parser'] == 'synthetic-test-only'
    assert StructuredObject.from_json(obj.to_json()).to_dict() == obj.to_dict()
    assert describe_structure(obj)['support']['nodes'] == 3


def test_ast_handoff_multiple_roots_and_claim_validation():
    doc = hierarchy()
    doc['roots'].append('other')
    doc['nodes'].append({'id': 'other', 'kind': 'unit', 'attributes': {}})
    assert HierarchyHandoffAdapter().adapt(doc).object.topology['kind'] == 'forest'
    doc['roots'] = ['unit']
    with pytest.raises(StructuredContractError, match='root'):
        HierarchyHandoffAdapter().adapt(doc)
    doc = hierarchy()
    doc['edges'].append({'id': 'cycle', 'parent': 'return', 'child': 'unit',
                         'kind': 'syntax_child'})
    with pytest.raises(StructuredContractError, match='acyclic|root|unique'):
        HierarchyHandoffAdapter().adapt(doc)
    doc = hierarchy()
    doc['edges'].append({'id': 'extra', 'parent': 'unit', 'child': 'return',
                         'kind': 'syntax_child'})
    with pytest.raises(StructuredContractError, match='unique parents|unique parent|acyclic'):
        HierarchyHandoffAdapter().adapt(doc)


@pytest.mark.parametrize('tamper', [
    lambda d: d.__setitem__('schema', 'arbitrary.ast'),
    lambda d: d.__setitem__('nodes', [{'id': 'broken'}]),
    lambda d: d['edges'][0].__setitem__('child', 'missing'),
    lambda d: d['nodes'][0].__setitem__('unknown_field', 'drop-me'),
    lambda d: d['edges'][0].__setitem__('unknown_field', 'drop-me'),
    lambda d: d.__setitem__('source_id', ''),
])
def test_ast_handoff_rejects_unsupported_or_malformed_artifact(tamper):
    data = hierarchy()
    tamper(data)
    with pytest.raises(StructuredContractError):
        HierarchyHandoffAdapter().adapt(data)


def test_pypique_immutable_concluded_graph_preserves_explicit_semantics():
    obj = cross_module_adapters().adapt('pypique_graph', py_projection()).require_lossless()
    assert obj.topology['kind'] == 'directed_graph'
    assert len(obj.nodes) == 3 and len(obj.edges) == 1
    assert obj.nodes[0]['attributes']['details']['acceptance'] == 0.4
    assert obj.edges[0]['attributes']['semantics'].endswith('not_causal')
    assert obj.edges[0]['attributes']['details']['mutual_information'] == .18
    assert not obj.observations
    assert obj.provenance['source_export_metadata']['selected_part'] == 'graph'
    assert obj.provenance['source_export_metadata']['source_export_header']['archive_sha256'] == 'a' * 64
    assert StructuredObject.from_json(obj.to_json()).to_json() == obj.to_json()


def test_pypique_mixed_directions_fail_closed_then_project_with_audit():
    doc = py_projection(mixed=True)
    with pytest.raises(StructuredContractError, match='mixed directed/undirected'):
        cross_module_adapters().adapt('pypique_graph', doc)
    directed = cross_module_adapters().adapt('pypique_graph_directed', doc)
    assert len(directed.object.nodes) == 3 and len(directed.object.edges) == 1
    assert directed.losses and 'excluded' in directed.losses[0]
    assert directed.object.topology['kind'] == 'directed_graph'
    assert directed.object.edges[0]['kind'] == 'signed_information'
    undirected = cross_module_adapters().adapt('pypique_graph_undirected', doc)
    assert undirected.losses and len(undirected.object.edges) == 1
    assert undirected.object.topology['kind'] == 'undirected_graph'
    assert undirected.object.edges[0]['attributes']['details']['trace_only'] is True
    with pytest.raises(StructuredContractError, match='lossy'):
        directed.require_lossless()


def test_pypique_backing_remains_separate_and_nonprobabilistic():
    data = py_projection(backing=True)
    result = cross_module_adapters().adapt('pypique_backing', data)
    obj = result.require_lossless()
    assert obj.topology['kind'] == 'undirected_graph'
    assert obj.edges[0]['attributes']['details']['weight'] == -2.3
    assert obj.edges[0]['kind'] == 'learned_rbm_connection'
    assert obj.provenance['source_export_metadata']['selected_part'] == 'learned_probabilistic_backing'
    assert 'explicit_probability_model' not in obj.capabilities
    assert len(obj.observations) == 0
    with pytest.raises(StructuredContractError, match='absent'):
        cross_module_adapters().adapt('pypique_backing', py_projection(backing=False))


def test_pypique_rejects_wrong_contract_role_and_forged_header():
    doc = py_projection()
    doc['graph']['graph_role'] = 'immutable_learned_probabilistic_backing'
    with pytest.raises(StructuredContractError, match='role'):
        cross_module_adapters().adapt('pypique_graph', doc)
    doc = py_projection()
    doc['archive_sha256'] = 'not-a-hash'
    with pytest.raises(StructuredContractError, match='SHA-256'):
        cross_module_adapters().adapt('pypique_graph', doc)
    doc = py_projection()
    doc['immutable'] = False
    with pytest.raises(StructuredContractError, match='immutable'):
        cross_module_adapters().adapt('pypique_graph', doc)


def test_pah_shared_display_graph_keeps_coordinates_not_probabilities():
    artifact = {
        'schema': SHARED_GRAPH_SCHEMA, 'title': 'A display only',
        'nodes': [{'id': 'a', 'label': 'A', 'x': 1.0, 'y': 2},
                  {'id': 'b', 'label': 'B', 'x': -3.5, 'y': 4}],
        'edges': [{'id': 'ab', 'source': 'a', 'target': 'b',
                   'directed': True, 'label': 'display relation'}],
    }
    result = cross_module_adapters().adapt('shared_graph', artifact)
    assert result.losses
    assert result.object.nodes[0]['attributes']['x'] == 1.0
    assert result.object.edges[0]['attributes']['display_label'] == 'display relation'
    assert not result.object.observations


def test_project_scoped_ui_import_uses_real_bridge_and_no_implicit_analysis(tmp_path):
    (tmp_path / 'ast.json').write_text(json.dumps(hierarchy()), encoding='utf-8')
    source, meta = read_source(tmp_path, 'ast.json', 'hierarchy_handoff')
    assert meta['topology'] == 'tree' and meta['rows'] == 0
    assert meta['warnings']
    pinned, meta2 = read_source(tmp_path, meta['path'], meta['kind'], meta['sha256'])
    assert pinned.to_json() == source.to_json()
    report, _ = analyze(source, 'data', 'structure', {})
    assert report['support']['nodes'] == 3
    with pytest.raises(StructuredContractError):
        analyze(source, 'profiling', 'information', {'x': 'method', 'y': 'return'})
    (tmp_path / 'network.json').write_text(json.dumps(py_projection(mixed=True)), encoding='utf-8')
    with pytest.raises(StructuredContractError, match='mixed directed/undirected'):
        read_source(tmp_path, 'network.json', 'pypique_graph')
    _, meta3 = read_source(tmp_path, 'network.json', 'pypique_graph_directed')
    assert meta3['losses']
    (tmp_path / 'linked.json').symlink_to(tmp_path / 'network.json')
    with pytest.raises(StructuredContractError, match='symlink'):
        read_source(tmp_path, 'linked.json', 'pypique_graph')
