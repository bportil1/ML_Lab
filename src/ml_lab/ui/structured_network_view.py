"""Presentation adapter for exhaustive observed-variable association reports.

The PAH_Shared renderer owns all layout and interaction. Here a metric pair
never becomes a structural, learned-probabilistic, or causal edge.
"""
from __future__ import annotations

from ml_lab.structured import StructuredContractError
from ml_lab.structured.network_information import SCHEMA


def association_view(report: dict, *, source_revision: str) -> dict:
    if report.get('schema') != SCHEMA:
        raise StructuredContractError('verified information-network report required')
    if (not isinstance(source_revision, str) or len(source_revision) != 71 or
            not source_revision.startswith('sha256:') or
            any(c not in '0123456789abcdef' for c in source_revision[7:])):
        raise StructuredContractError('source revision required')
    nodes = []
    overlays_nodes = {}
    for node in report['nodes']:
        nodes.append({'id': node['id'], 'label': node['variable_id'][:240],
                      'kind': 'observed_variable', 'group': 'observed_variables'})
        measured = node['entropy_bits'] is not None
        overlays_nodes[node['id']] = {'status': 'measured' if measured else 'insufficient_support',
            'metrics': [{'name': 'empirical_entropy', 'unit': 'bits',
                         'estimator': 'empirical_discrete',
                         'status': 'measured' if measured else 'insufficient_support',
                         'value': node['entropy_bits'], 'support': node['support']}]}
    edges = []
    overlays_edges = {}
    for pair in report['pairs']:
        if pair['status'] != 'ok':
            continue
        edges.append({'id': pair['id'], 'source': pair['source_id'],
                      'target': pair['target_id'], 'kind': report['metric'],
                      'directed': False, 'traceable': False,
                      'label': f"{pair['value_bits']:.4g} bits · n={pair['support']}"})
        overlays_edges[pair['id']] = {'status': 'measured', 'metrics': [
            {'name': report['metric'], 'unit': 'bits',
             'estimator': 'empirical_discrete', 'status': 'measured',
             'value': pair['value_bits'], 'support': pair['support']}]}
    return {
        'schema': 'pah-shared.graph-view@1',
        'view_id': 'ml-lab-observed-information-network',
        'source_revision': source_revision,
        'graph_role': 'empirical_statistical_association',
        'title': 'Observed-variable information network (not source topology)',
        'nodes': nodes, 'edges': edges,
        'display': {'layout': 'circular', 'coverage': {
            'scope': 'exhaustive', 'total_nodes': len(nodes), 'total_edges': len(edges)}},
        'overlays': {'nodes': overlays_nodes, 'edges': overlays_edges},
    }
