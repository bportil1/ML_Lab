"""Read-only, domain-neutral hierarchy handoff for syntax trees or other forests.

This is an ML_Lab interoperability envelope, NOT a claim about a Code Analyzer
export schema. The producing module explicitly exports typed nodes and edges;
no repository, source text, parser, SQLite database or module API is accessed.
"""
from __future__ import annotations

from collections.abc import Mapping

from ..contracts import CONTRACT_SCHEMA, StructuredContractError
from ..object import StructuredObject
from .registry import AdaptationResult

AST_HANDOFF_SCHEMA = 'ml-lab.hierarchy-handoff@1'
_ALLOWED = frozenset({'schema', 'source_id', 'nodes', 'edges', 'roots', 'provenance'})
_NODE_FIELDS = frozenset({'id', 'kind', 'label', 'attributes', 'source_ref', 'metadata'})
_EDGE_FIELDS = frozenset({'id', 'parent', 'child', 'kind', 'ordinal', 'attributes',
                          'source_ref', 'metadata'})


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StructuredContractError(f'{field} requires a nonempty string')
    return value


class HierarchyHandoffAdapter:
    """Import an explicitly rooted tree/forest; verify claims against real edges."""

    def adapt(self, source: object) -> AdaptationResult:
        if not isinstance(source, Mapping) or source.get('schema') != AST_HANDOFF_SCHEMA:
            raise StructuredContractError('expected explicit hierarchy handoff schema')
        if set(source) - _ALLOWED:
            raise StructuredContractError('unknown hierarchy handoff fields; refusing lossy import')
        source_id = _identifier(source.get('source_id'), 'source_id')
        raw_nodes, raw_edges, roots = (source.get('nodes'), source.get('edges'),
                                      source.get('roots'))
        if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
            raise StructuredContractError('hierarchy handoff requires node and edge arrays')
        if not isinstance(roots, list) or not roots or len(roots) != len(set(map(str, roots))):
            raise StructuredContractError('hierarchy handoff requires unique declared roots')
        roots = [_identifier(root, 'roots[]') for root in roots]
        provenance = source.get('provenance', {})
        if not isinstance(provenance, Mapping):
            raise StructuredContractError('provenance must be an object')
        nodes = []
        for index, node in enumerate(raw_nodes):
            if not isinstance(node, Mapping) or set(node) - _NODE_FIELDS:
                raise StructuredContractError(f'unsupported hierarchy node fields at {index}')
            if not {'id', 'kind'}.issubset(node):
                raise StructuredContractError(f'hierarchy node {index} needs id and kind')
            nodes.append({**node, 'attributes': node.get('attributes', {})})
        edges = []
        for index, edge in enumerate(raw_edges):
            if not isinstance(edge, Mapping) or set(edge) - _EDGE_FIELDS:
                raise StructuredContractError(f'unsupported hierarchy edge fields at {index}')
            if not {'id', 'parent', 'child', 'kind'}.issubset(edge):
                raise StructuredContractError(f'hierarchy edge {index} needs id, parent, child, kind')
            edges.append({'id': edge['id'], 'kind': edge['kind'],
                          'source_id': edge['parent'], 'target_id': edge['child'],
                          **{k: edge[k] for k in ('ordinal', 'attributes', 'source_ref', 'metadata')
                             if k in edge}})
        # Topology validation rejects cycles, multiple parents, dangling endpoints.
        obj = StructuredObject.from_dict({
            'schema': CONTRACT_SCHEMA,
            'topology': {'kind': 'tree' if len(roots) == 1 else 'forest'},
            'nodes': nodes, 'edges': edges, 'observations': [],
            'observation_unit': 'graph',
            'provenance': {'adapter': AST_HANDOFF_SCHEMA, 'source_id': source_id,
                           'source_provenance': provenance},
        })
        derived_roots = {node['id'] for node in obj.nodes} - {
            edge['target_id'] for edge in obj.edges
        }
        if derived_roots != set(roots):
            raise StructuredContractError('declared hierarchy roots do not match actual topology')
        return AdaptationResult(obj, AST_HANDOFF_SCHEMA,
                                warnings=('Source locations are provenance, not observations; '
                                          'no statistical samples were inferred.',))
