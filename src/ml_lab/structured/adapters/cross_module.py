"""Local, read-only adapters for pyPIQUE and reusable display-network exports.

Contracts are grounded in pyPIQUE's `project_concluded_graph` and the existing
`pah-shared.graph@1` display format. No pyPIQUE or PAH imports, file/database
access, inferred probabilistic edges, or automatic graph fitting.
"""
from __future__ import annotations

from collections.abc import Mapping
import re

from ..contracts import CONTRACT_SCHEMA, StructuredContractError
from ..object import StructuredObject
from .registry import AdaptationResult

PYPIQUE_SCHEMA = 'pypique.concluded_graph'
SHARED_GRAPH_SCHEMA = 'pah-shared.graph@1'
ROLES = frozenset({'native_quality_model',
                   'recorded_signed_information_and_reporting',
                   'immutable_learned_probabilistic_backing'})


def _nonempty(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StructuredContractError(f'{where} must be a nonempty string')
    return value


def _nodes_edges(graph: Mapping, *, adapter: str, source_provenance: Mapping,
                 edge_filter: str | None = None) -> AdaptationResult:
    raw_nodes, raw_edges = graph.get('nodes'), graph.get('edges')
    if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
        raise StructuredContractError('graph projection requires nodes and edges arrays')
    if not raw_nodes:
        raise StructuredContractError('graph projection has no nodes')
    if edge_filter not in (None, 'directed', 'undirected'):
        raise StructuredContractError('unsupported graph edge projection')
    nodes = []
    for index, node in enumerate(raw_nodes):
        if not isinstance(node, Mapping):
            raise StructuredContractError(f'node {index} must be an object')
        ident = _nonempty(node.get('id'), f'node {index}.id')
        if adapter.startswith(PYPIQUE_SCHEMA) and 'kind' not in node:
            raise StructuredContractError(f'pyPIQUE node {index} lacks a recorded kind')
        kind = _nonempty(node.get('kind', 'display_node'), f'node {index}.kind')
        label = node.get('label', ident)
        if not isinstance(label, str):
            raise StructuredContractError(f'node {index}.label must be a string')
        record = {'id': ident, 'kind': kind, 'label': label,
                  'attributes': {k: v for k, v in node.items()
                                 if k not in ('id', 'kind', 'label', 'source_ref')}}
        if 'source_ref' in node:
            record['source_ref'] = node['source_ref']
        nodes.append(record)
    edges = []
    directions = set()
    for index, edge in enumerate(raw_edges):
        if not isinstance(edge, Mapping):
            raise StructuredContractError(f'edge {index} must be an object')
        direction = edge.get('directed')
        if type(direction) is not bool:
            raise StructuredContractError(f'edge {index} requires explicit boolean directedness')
        directions.add(direction)
    if len(directions) > 1 and edge_filter is None:
        raise StructuredContractError('mixed directed/undirected source: choose an explicit '
                                      'directed-only or undirected-only projection; '
                                      'no implicit edge coercion')
    kept_directions = set()
    for index, edge in enumerate(raw_edges):
        direction = edge['directed']
        if edge_filter and direction != (edge_filter == 'directed'):
            continue
        kept_directions.add(direction)
        if adapter.startswith(PYPIQUE_SCHEMA) and 'relationship' not in edge:
            raise StructuredContractError(f'pyPIQUE edge {index} lacks its relationship type')
        kind = _nonempty(edge.get('relationship', edge.get('label', 'display_edge')),
                         f'edge {index}.relationship')
        record = {'id': _nonempty(edge.get('id'), f'edge {index}.id'),
                  'kind': kind,
                  'source_id': _nonempty(edge.get('source'), f'edge {index}.source'),
                  'target_id': _nonempty(edge.get('target'), f'edge {index}.target'),
                  'attributes': {k: v for k, v in edge.items()
                                 if k not in ('id', 'source', 'target', 'relationship',
                                              'label', 'directed', 'source_ref')}}
        if 'label' in edge:
            record['attributes']['display_label'] = edge['label']
        if 'source_ref' in edge:
            record['source_ref'] = edge['source_ref']
        edges.append(record)
    if len(kept_directions) > 1:
        raise StructuredContractError('internal mixed-direction projection error')
    directed = next(iter(kept_directions)) if kept_directions else (
        edge_filter != 'undirected' if edge_filter else next(iter(directions), True))
    obj = StructuredObject.from_dict({
        'schema': CONTRACT_SCHEMA,
        'topology': {'kind': 'directed_graph' if directed else 'undirected_graph'},
        'nodes': nodes, 'edges': edges,
        'observation_unit': 'graph', 'observations': [],
        'provenance': {'adapter': adapter, 'source_graph_metadata': {
            k: v for k, v in graph.items() if k not in ('nodes', 'edges')},
            'source_export_metadata': dict(source_provenance),
            'projection': edge_filter or 'all_edges'},
    })
    removed = len(raw_edges) - len(edges)
    loss = (f'{removed}_source_edges_excluded_by_explicit_{edge_filter}_projection',) if removed else ()
    warnings = ('A network is not a set of aligned observations; no sample rows inferred.',
                'Source graph edges are not automatically probabilistic or causal links.')
    if loss:
        warnings += ('Only the selected edge-direction subset was imported.',)
    return AdaptationResult(obj, adapter, warnings=warnings, losses=loss)


class PyPIQUEConcludedAdapter:
    """Import one selected graph from a verified-format concluded projection.

    `graph_part` selects main graph or backing. Mixed-direction main graphs
    require an explicit direction filter; omitted edges are always reported.
    SHA fields are source *claims*, not independently verified archives.
    """

    def __init__(self, *, graph_part: str = 'graph', edge_filter: str | None = None):
        if graph_part not in ('graph', 'learned_probabilistic_backing'):
            raise ValueError('unsupported concluded graph part')
        self.graph_part = graph_part
        self.edge_filter = edge_filter

    def adapt(self, source: object) -> AdaptationResult:
        if not isinstance(source, Mapping) or source.get('schema') != PYPIQUE_SCHEMA:
            raise StructuredContractError('expected pyPIQUE concluded graph projection')
        if type(source.get('schema_version')) is not int or source['schema_version'] != 1 or source.get('immutable') is not True:
            raise StructuredContractError('requires immutable pyPIQUE concluded graph version 1')
        family = source.get('family')
        if not isinstance(family, str) or family not in ('standard_pique', 'mi_informed_pique'):
            raise StructuredContractError('unknown concluded-model family')
        for key in ('version_id', 'archive_sha256', 'manifest_sha256'):
            value = _nonempty(source.get(key), key)
            if key.endswith('_sha256') and not re.fullmatch(r'[a-fA-F0-9]{64}', value):
                raise StructuredContractError(key + ' must be a SHA-256 hex claim')
        part = source.get(self.graph_part)
        if not isinstance(part, Mapping):
            raise StructuredContractError('requested pyPIQUE graph part is absent')
        role = part.get('graph_role')
        expected_roles = ({'native_quality_model'} if family == 'standard_pique' else
                          {'recorded_signed_information_and_reporting'}) if self.graph_part == 'graph' else {
                              'immutable_learned_probabilistic_backing'}
        if not isinstance(role, str) or role not in expected_roles:
            raise StructuredContractError('pyPIQUE graph role conflicts with selected part/model family')
        provenance = {'schema': PYPIQUE_SCHEMA,
                      'schema_version': 1,
                      'selected_part': self.graph_part,
                      'source_export_header': {k: v for k, v in source.items()
                                               if k not in ('graph', 'learned_probabilistic_backing')}}
        result = _nodes_edges(part, adapter=f'{PYPIQUE_SCHEMA}@1',
                              source_provenance=provenance, edge_filter=self.edge_filter)
        return AdaptationResult(result.object, result.source_schema,
                                warnings=result.warnings + (
                                    'The other concluded graph part is not imported or merged.',
                                    'Source SHA-256 claims are recorded; the original archive is not reverified.'),
                                losses=result.losses)


class SharedGraphProjectionAdapter:
    """Import PAH_Shared's display-only graph@1 (not scientific provenance)."""

    def adapt(self, source: object) -> AdaptationResult:
        if not isinstance(source, Mapping) or source.get('schema') != SHARED_GRAPH_SCHEMA:
            raise StructuredContractError('expected PAH_Shared graph@1 display artifact')
        if set(source) - {'schema', 'title', 'nodes', 'edges'}:
            raise StructuredContractError('unrecognized graph@1 fields; refusing silent loss')
        result = _nodes_edges(source, adapter=SHARED_GRAPH_SCHEMA,
                              source_provenance={'schema': SHARED_GRAPH_SCHEMA})
        return AdaptationResult(result.object, result.source_schema,
                                warnings=result.warnings + (
                                    'This is a display-only projection; missing scientific metadata '
                                    'cannot be reconstructed from its nodes and edges.',),
                                losses=('original_scientific_source_metadata_not_in_display_export',))
