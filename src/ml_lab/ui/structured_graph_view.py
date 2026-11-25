"""SU-04: source-verified StructuredObject -> PAH_Shared graph *presentation*.

No geometry, rendering, graph interpretation, estimation or side effects live
here. The underlying graph renderer is the SU-02 PAH_Shared component.
"""
from __future__ import annotations

from collections import deque
from ml_lab.structured import StructuredContractError, StructuredObject

GRAPH_SCHEMA = 'pah-shared.graph-view@1'


def _display_groups(source: StructuredObject) -> tuple[dict[str, str], str]:
    """Display-only tiers, not inferred random-variable or causal layers."""
    kind = source.topology['kind']
    ids = [node['id'] for node in source.nodes]
    if kind not in {'tree', 'forest', 'dag'}:
        return {key: 'network' for key in ids}, 'circular'

    adjacency = {key: [] for key in ids}
    indegree = {key: 0 for key in ids}
    for edge in source.edges:
        adjacency[edge['source_id']].append(edge['target_id'])
        indegree[edge['target_id']] += 1
    depths = {key: 0 for key, count in indegree.items() if count == 0}
    queue = deque(depths)
    while queue:
        parent = queue.popleft()
        for child in adjacency[parent]:
            depths[child] = max(depths.get(child, 0), depths[parent] + 1)
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(depths) != len(ids):
        raise StructuredContractError('validated hierarchy has no complete display order')
    # Preserve exact structural depth in presentation metadata. The older
    # depth-39 display clamp merged distinct deep levels, contradicting the
    # deterministic HIA-02 scope inventory. PAH_Shared pages large overviews.
    return {key: f'depth-{depths[key]:02d}' for key in ids}, 'layers'


def make_shared_graph_view(source: StructuredObject, *, fingerprint: str) -> dict:
    """Create a detached, full source-structure view, with original identity.

    No hard cap on source node or edge totals: PAH_Shared displays large graphs
    through explicit overview and navigable pages, never silent sampling.
    Source-edge orientation remains source orientation; neither hierarchy edges
    nor association links become statistical/probabilistic relationships.
    """
    if not isinstance(source, StructuredObject):
        raise StructuredContractError('validated StructuredObject required')
    if (not isinstance(fingerprint, str) or len(fingerprint) != 64 or
            any(c not in '0123456789abcdef' for c in fingerprint)):
        raise StructuredContractError('source SHA-256 fingerprint required')
    # Display identifiers and labels preserve their complete, validated source
    # values. The shared renderer abbreviates only the visible SVG label and
    # keeps full text in its accessible tooltip and source record inspector.
    groups, layout = _display_groups(source)
    directed = source.topology['kind'] != 'undirected_graph'
    trace = 'rooted_hierarchy' in source.capabilities
    return {
        'schema': GRAPH_SCHEMA,
        'view_id': 'ml-lab-original-structure',
        'source_revision': 'sha256:' + fingerprint,
        'graph_role': 'source_structure',
        'title': 'Original structured object',
        'nodes': [
            {'id': node['id'], 'label': node.get('label') or node['id'],
             'kind': node['kind'], 'group': groups[node['id']]}
            for node in source.nodes
        ],
        'edges': [
            {'id': edge['id'], 'source': edge['source_id'], 'target': edge['target_id'],
             'kind': edge['kind'], 'directed': directed, 'traceable': bool(trace and directed),
             'label': edge['kind']}
            for edge in source.edges
        ],
        'display': {
            'layout': layout,
            'coverage': {'scope': 'exhaustive', 'total_nodes': len(source.nodes),
                         'total_edges': len(source.edges)},
        },
        'overlays': {'nodes': {}, 'edges': {}},
    }


def make_shared_graph_payload(source: StructuredObject, *, fingerprint: str) -> dict:
    """Owner-controlled inspector data, not extra fields of the shared contract."""
    view = make_shared_graph_view(source, fingerprint=fingerprint)
    # Detached original records retain provenance without altering display nodes.
    original = source.to_dict()
    return {
        'view': view,
        'inspector': {
            'nodes': {node['id']: node for node in original['nodes']},
            'edges': {edge['id']: edge for edge in original['edges']},
        },
        'notice': ('Source topology only. Graph edges are not estimated dependencies; '
                   'no variance, entropy, MI, CMI or pruning score has been computed.'),
    }
