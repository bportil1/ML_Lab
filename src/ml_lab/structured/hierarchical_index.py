"""HIA-02: deterministic, source-pinned hierarchy and scope membership.

Only *source topology* is interpreted here. No graph edge is statistical,
probabilistic, or causal. No source node is treated as an observation. The
index is detached and immutable; navigation never modifies source artifacts.

The complete graph is indexed. For memory scalability, ancestor, descendant,
and branch traversals are on demand (iterative, not recursive), rather than
materializing the transitive closure of the entire graph.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .bindings import ObservationBindingResult, source_sha256
from .contracts import StructuredContractError
from .hierarchical_contracts import (
    HierarchicalAnalysisRequest, HierarchicalScope, STRUCTURAL_METRICS,
    inspect_hierarchical_eligibility,
)
from .object import StructuredObject

HIERARCHY_SCHEMA = 'ml-lab.hierarchical-level-index@1'
SELECTION_SCHEMA = 'ml-lab.hierarchical-scope-selection@1'


def _fail(message: str) -> None:
    raise StructuredContractError(message)


def _index_policy(topology: str) -> str:
    if topology in ('tree', 'forest'):
        return 'rooted_depth'
    if topology == 'dag':
        return 'dag_longest_path'
    _fail('hierarchical levels unavailable for cyclic, undirected or general graph topology')


@dataclass(frozen=True, slots=True)
class HierarchyIndex:
    """Immutable source-level index. Identifiers are original exact node IDs.

    `.levels` stores *all* nodes exactly once. Parents/children are direct
    source-edge adjacency. Ancestors/descendants are visited on demand so the
    index itself does not require O(V^2) storage for a deep chain or dense DAG.
    """

    source_digest: str
    topology: str
    level_policy: str
    roots: tuple[str, ...]
    levels: tuple[tuple[str, ...], ...]
    edge_count: int
    edge_ids: tuple[str, ...]
    _depth: Mapping[str, int]
    _parents: Mapping[str, tuple[str, ...]]
    _children: Mapping[str, tuple[str, ...]]

    @property
    def node_count(self) -> int:
        return len(self._depth)

    @property
    def level_count(self) -> int:
        return len(self.levels)

    def _node(self, node_id: str) -> str:
        if not isinstance(node_id, str) or node_id not in self._depth:
            _fail('unknown original source node ID in hierarchy selection')
        return node_id

    def level_of(self, node_id: str) -> int:
        return self._depth[self._node(node_id)]

    def level_ids(self, depth: int) -> tuple[str, ...]:
        if type(depth) is not int or depth < 0 or depth >= len(self.levels):
            _fail('requested hierarchy level does not exist in this source')
        return self.levels[depth]

    def parents_of(self, node_id: str) -> tuple[str, ...]:
        return self._parents[self._node(node_id)]

    def children_of(self, node_id: str) -> tuple[str, ...]:
        return self._children[self._node(node_id)]

    def _walk(self, start: str, adjacency: Mapping[str, tuple[str, ...]]) -> tuple[str, ...]:
        self._node(start)
        visited: set[str] = set()
        stack = list(adjacency[start])
        while stack:
            node = stack.pop()
            if node == start or node in visited:
                continue
            visited.add(node)
            stack.extend(adjacency[node])
        return tuple(sorted(visited, key=lambda node: (self._depth[node], node)))

    def ancestors_of(self, node_id: str) -> tuple[str, ...]:
        """All ancestors, regardless of DAG parent multiplicity; excludes node."""
        return self._walk(node_id, self._parents)

    def descendants_of(self, node_id: str) -> tuple[str, ...]:
        """All descendants of this ID, including DAG reachability; excludes node."""
        return self._walk(node_id, self._children)

    def roots_for(self, node_id: str) -> tuple[str, ...]:
        """Source roots that can reach the node, not probabilistic parents."""
        node_id = self._node(node_id)
        ancestors = set(self.ancestors_of(node_id))
        return tuple(root for root in self.roots if root == node_id or root in ancestors)

    def branch_ids(self, node_id: str) -> tuple[str, ...]:
        if self.topology not in ('tree', 'forest'):
            _fail('branch membership requires verified tree/forest unique-parent topology')
        return tuple(sorted((self._node(node_id), *self.descendants_of(node_id)),
                            key=lambda node: (self._depth[node], node)))

    def prior_level_ids(self, depth: int) -> tuple[str, ...]:
        self.level_ids(depth)
        return tuple(node for level in self.levels[:depth] for node in level)

    def scope_ids(self, scope: HierarchicalScope) -> tuple[str, ...]:
        if not isinstance(scope, HierarchicalScope):
            _fail('hierarchy selection requires a validated HIA scope')
        if scope.kind == 'node':
            return (self._node(scope.node_ids[0]),)
        if scope.kind == 'branch':
            return self.branch_ids(scope.node_ids[0])
        if scope.kind == 'level':
            return self.level_ids(scope.level)
        return tuple(node for level in self.levels for node in level)

    def node_context(self, node_id: str) -> dict[str, Any]:
        """One original node's structural neighborhood, without numeric claims."""
        node_id = self._node(node_id)
        return {
            'node_id': node_id,
            'level': self._depth[node_id],
            'parent_ids': list(self._parents[node_id]),
            'child_ids': list(self._children[node_id]),
            'root_ids': list(self.roots_for(node_id)),
            'ancestor_count': len(self.ancestors_of(node_id)),
            'descendant_count': len(self.descendants_of(node_id)),
            'relationships_are_source_topology_only': True,
        }

    def to_dict(self) -> dict[str, Any]:
        """Full deterministic, JSON-compatible source-level inventory."""
        return {
            'schema': HIERARCHY_SCHEMA,
            'source_digest': self.source_digest,
            'source_topology': self.topology,
            'level_policy': self.level_policy,
            'node_count': self.node_count,
            'edge_count': self.edge_count,
            'source_edge_ids': list(self.edge_ids),
            'roots': list(self.roots),
            'levels': [{'depth': depth, 'node_ids': list(nodes), 'node_count': len(nodes)}
                       for depth, nodes in enumerate(self.levels)],
            'nodes': [{'node_id': node, 'depth': self._depth[node],
                       'parent_ids': list(self._parents[node]),
                       'child_ids': list(self._children[node])}
                      for nodes in self.levels for node in nodes],
            'structural_levels_are_probabilistic_layers': False,
            'source_graph_is_statistical_graph': False,
        }


def build_hierarchy_index(source: StructuredObject, *, level_policy: str | None = None) -> HierarchyIndex:
    """Index entire validated tree, forest or DAG without arbitrary size limits.

    Tree/forest distance is unique-parent root depth; for DAG it is the longest
    directed path from any indegree-zero root. O(V+E) traversal after sorting
    identity/adjacency for deterministic presentation. No transitive closure.
    """
    if not isinstance(source, StructuredObject):
        _fail('HIA-02 hierarchy requires a validated StructuredObject')
    topology = source.topology['kind']
    expected = _index_policy(topology)
    if level_policy is not None and level_policy != expected:
        _fail(f'{topology} hierarchy requires {expected}')
    if topology in ('tree', 'forest') and 'rooted_hierarchy' not in source.capabilities:
        _fail('hierarchical source requires verified rooted_hierarchy capability')
    if topology == 'dag' and 'acyclic' not in source.capabilities:
        _fail('DAG requires verified acyclic capability')

    ids = sorted(node['id'] for node in source.nodes)
    parents: dict[str, set[str]] = {node: set() for node in ids}
    children: dict[str, set[str]] = {node: set() for node in ids}
    for edge in source.edges:
        start, end = edge['source_id'], edge['target_id']
        # Parallel DAG edges retain their original source identities, but only
        # affect direct topology reachability once.
        children[start].add(end)
        parents[end].add(start)
    sorted_parents = {node: tuple(sorted(neighbors)) for node, neighbors in parents.items()}
    sorted_children = {node: tuple(sorted(neighbors)) for node, neighbors in children.items()}
    indegree = {node: len(parents[node]) for node in ids}
    roots = tuple(node for node in ids if not indegree[node])
    if not roots and ids:
        _fail('hierarchical source has no indegree-zero root')
    frontier = deque(roots)
    depths = {root: 0 for root in roots}
    seen = 0
    while frontier:
        node = frontier.popleft()
        seen += 1
        for child in sorted_children[node]:
            depths[child] = max(depths.get(child, 0), depths[node] + 1)
            indegree[child] -= 1
            if indegree[child] == 0:
                frontier.append(child)
    if seen != len(ids):
        _fail('hierarchy traversal failed: cycle or invalid declared DAG')
    if not ids:
        # The source may have no nodes; an empty hierarchy has no valid level.
        levels: tuple[tuple[str, ...], ...] = ()
    else:
        grouped: list[list[str]] = [[] for _ in range(max(depths.values()) + 1)]
        for node in ids:
            grouped[depths[node]].append(node)
        levels = tuple(tuple(nodes) for nodes in grouped)
    return HierarchyIndex(
        source_digest=source_sha256(source),
        topology=topology, level_policy=expected, roots=roots, levels=levels,
        edge_count=len(source.edges), edge_ids=tuple(sorted(edge['id'] for edge in source.edges)),
        _depth=MappingProxyType(depths),
        _parents=MappingProxyType(sorted_parents),
        _children=MappingProxyType(sorted_children),
    )


def inspect_hierarchical_selection(
    source: StructuredObject,
    request: HierarchicalAnalysisRequest,
    binding: ObservationBindingResult | None = None,
    *, index: HierarchyIndex | None = None,
) -> dict[str, Any]:
    """Resolve verified scope and variable IDs; never estimate information.

    A variable bound to multiple original nodes is never allocated as if it
    were independent per-node data. Level-overlapping bindings are disclosed,
    for later estimator policy decisions. Direct numeric statistics remain
    explicitly deferred or unavailable.
    """
    eligibility = inspect_hierarchical_eligibility(source, request, binding)
    if index is None:
        index = build_hierarchy_index(source)
    if not isinstance(index, HierarchyIndex) or index.source_digest != request.source_digest:
        _fail('hierarchy index source fingerprint mismatch')
    if index.topology != source.topology['kind'] or index.level_policy != _index_policy(index.topology):
        _fail('hierarchy index source topology or level policy mismatch')
    if request.scope.kind == 'level' and index.level_policy != request.level_policy:
        _fail('hierarchy request and index level policy mismatch')
    selected = index.scope_ids(request.scope)
    selected_set = set(selected)
    scope_depths = sorted({index.level_of(node) for node in selected})
    prior_ids = index.prior_level_ids(request.scope.level) if request.scope.kind == 'level' else ()
    mapped: list[str] = []
    ambiguous: list[str] = []
    level_overlap: list[str] = []
    attribution: dict[str, list[str]] = {}
    if binding is not None:
        doc = binding.to_dict()
        for variable in doc['variables']:
            original = tuple(variable['node_ids'])
            covered = sorted(selected_set.intersection(original))
            if not covered:
                continue
            name = variable['variable_id']
            mapped.append(name)
            attribution[name] = list(original)
            if variable.get('merged_node_identity_ambiguous') or len(original) > 1:
                ambiguous.append(name)
            if request.scope.kind == 'level' and any(index.level_of(node) != request.scope.level
                                                      for node in original):
                level_overlap.append(name)
    mapped.sort(); ambiguous.sort(); level_overlap.sort()
    evidence = []
    for item in eligibility['metrics']:
        result = dict(item)
        if result['status'] == 'deferred' and item['metric'] not in STRUCTURAL_METRICS:
            if binding is not None and not mapped:
                result.update(status='unavailable', reason='no_variables_bound_to_selected_scope')
            elif ambiguous:
                result.update(status='unavailable', reason='merged_source_node_attribution_ambiguous')
            elif result['reason'] == 'HIA-02_level_assignment_required':
                result['reason'] = 'HIA_scientific_estimator_not_implemented'
        evidence.append(result)
    return {
        'schema': SELECTION_SCHEMA,
        'source_digest': index.source_digest,
        'binding_digest': request.binding_digest,
        'level_policy': index.level_policy,
        'source_topology': index.topology,
        'scope': request.scope.to_dict(),
        'node_ids': list(selected),
        'node_count': len(selected),
        'levels': scope_depths,
        'roots': list(index.roots),
        'parent_edge_is_statistical_dependency': False,
        'prior_level_node_ids': list(prior_ids),
        'mapped_variable_ids': mapped,
        'variable_source_node_ids': attribution,
        'ambiguous_variable_ids': ambiguous,
        'cross_level_binding_variable_ids': level_overlap,
        'observed_rows_available': eligibility['observed_rows_available'],
        'metrics': evidence,
        'statistical_estimates_computed': False,
    }
