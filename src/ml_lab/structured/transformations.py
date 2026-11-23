"""Reproducible, source-neutral structural views and reductions (ML-UW-30).

This module operates only on validated StructuredObjects. Topology changes are
explicit, observation rows are NEVER joined to nodes, and every dropped identity
is reported. No source adapters, database operations or scientific estimators.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from typing import Any

from .contracts import StructuredContractError
from .object import StructuredObject

PLAN_SCHEMA = 'ml-lab.structure-plan@1'
REPORT_SCHEMA = 'ml-lab.structure-transformation@1'
_MODES = frozenset({'full', 'subtree', 'depth', 'aggregate', 'flat'})


def _error(message: str) -> None:
    raise StructuredContractError(message)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False)


@dataclass(frozen=True, slots=True)
class TransformationPlan:
    """Explicit, versioned plan. No inferred paths, attributes or feature joins.

    ``roots`` selects one or more complete descendant subtrees.
    ``max_depth`` is absolute from each hierarchy root (root depth is zero).
    ``variables`` projects aligned observation columns *only when requested*;
    it is not inferred from surviving graph nodes.
    """

    mode: str
    roots: tuple[str, ...] = ()
    max_depth: int | None = None
    variables: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str) or self.mode not in _MODES:
            _error(f'unsupported structural transformation mode: {self.mode!r}')
        if not isinstance(self.roots, tuple) or any(
            not isinstance(name, str) or not name.strip() for name in self.roots
        ) or len(set(self.roots)) != len(self.roots):
            _error('roots must be a tuple of unique, nonempty node IDs')
        if self.mode == 'subtree':
            if not self.roots:
                _error('subtree mode requires at least one selected root')
        elif self.roots:
            _error('selected roots are only supported for subtree mode')
        if self.mode in {'depth', 'aggregate'}:
            if type(self.max_depth) is not int or self.max_depth < 0:
                _error('depth and aggregate modes require a nonnegative max_depth')
        elif self.max_depth is not None:
            _error('max_depth is only supported for depth or aggregate mode')
        if self.variables is not None:
            if not isinstance(self.variables, tuple) or not self.variables or any(
                not isinstance(name, str) or not name.strip() for name in self.variables
            ) or len(set(self.variables)) != len(self.variables):
                _error('variables must be a nonempty tuple of unique variable IDs')

    def to_dict(self) -> dict[str, Any]:
        return {'schema': PLAN_SCHEMA, 'mode': self.mode,
                'roots': list(self.roots), 'max_depth': self.max_depth,
                'variables': None if self.variables is None else list(self.variables)}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> 'TransformationPlan':
        if not isinstance(raw, Mapping):
            _error('transformation plan must be a mapping')
        allowed = {'schema', 'mode', 'roots', 'max_depth', 'variables'}
        if set(raw) - allowed:
            _error('unsupported transformation plan fields')
        if raw.get('schema') != PLAN_SCHEMA:
            _error('unsupported transformation plan schema')
        roots = raw.get('roots', [])
        variables = raw.get('variables')
        if not isinstance(roots, list) or (variables is not None and not isinstance(variables, list)):
            _error('plan roots and variables must be arrays')
        return cls(mode=raw.get('mode'), roots=tuple(roots),
                   max_depth=raw.get('max_depth'),
                   variables=None if variables is None else tuple(variables))

    def to_json(self) -> str:
        return _canonical(self.to_dict())


@dataclass(frozen=True, slots=True)
class TransformationResult:
    """Validated output with immutable, detached-on-read transformation audit."""

    object: StructuredObject
    plan: TransformationPlan
    _audit_json: str

    @property
    def audit(self) -> dict[str, Any]:
        return json.loads(self._audit_json)

    @property
    def lossless(self) -> bool:
        return self.audit['lossless']

    @property
    def reversible(self) -> bool:
        return self.audit['reversible']

    def flat_tables(self) -> dict[str, Any]:
        """Separate node and sample tables; never multiply rows by a graph join.

        Return source-neutral records with explicit cell states. The observation
        table has exactly one row for each original observation, if any. Nodes
        and observations have unrelated identity spaces unless a study supplies
        an explicit validated mapping; we deliberately do not merge them here.
        """
        if self.plan.mode != 'flat':
            _error('flat_tables is only available for flat structural views')
        doc = self.object.to_dict()
        return {
            'nodes': [dict(node) for node in doc['nodes']],
            'observations': [dict(row) for row in doc['observations']],
            'observation_unit': doc['observation_unit'],
        }

    def to_dict(self) -> dict[str, Any]:
        return {'plan': self.plan.to_dict(), 'object': self.object.to_dict(),
                'audit': self.audit}


def _hierarchy(source: StructuredObject) -> tuple[dict[str, list[str]], dict[str, str], dict[str, int]]:
    source.require('depth_reduction')  # validated rooted tree/forest, not arbitrary DAG
    children: dict[str, list[str]] = defaultdict(list)
    parents: dict[str, str] = {}
    for edge in source.edges:
        parent, child = edge['source_id'], edge['target_id']
        children[parent].append(child)
        parents[child] = parent
    roots = sorted(node['id'] for node in source.nodes if node['id'] not in parents)
    depths: dict[str, int] = {}
    queue = deque((root, 0) for root in roots)
    while queue:
        node, depth = queue.popleft()
        depths[node] = depth
        queue.extend((child, depth + 1) for child in children[node])
    if len(depths) != len(source.nodes):
        _error('hierarchy traversal failed to visit all nodes')
    return children, parents, depths


def _subtree_nodes(selected: tuple[str, ...], children: Mapping[str, list[str]],
                   node_ids: set[str]) -> set[str]:
    invalid = set(selected) - node_ids
    if invalid:
        _error(f'unknown subtree root(s): {", ".join(sorted(invalid))}')
    retained: set[str] = set()
    queue = deque(selected)
    while queue:
        node = queue.popleft()
        if node in retained:
            continue
        retained.add(node)
        queue.extend(children.get(node, ()))
    return retained


def _project_observations(doc: dict[str, Any], variables: tuple[str, ...] | None
                          ) -> tuple[list[dict[str, Any]], list[str]]:
    rows = doc['observations']
    if variables is None:
        return rows, []
    if not rows:
        _error('observation projection requires aligned observations')
    available = set(rows[0]['values'])
    unknown = set(variables) - available
    if unknown:
        _error(f'unknown aligned observation variables: {", ".join(sorted(unknown))}')
    projected = []
    # Order source variables deterministically instead of accidentally changing
    # the sample size or joining node attributes into sample rows.
    for row in rows:
        projected_row = dict(row)
        projected_row['values'] = {name: row['values'][name] for name in variables}
        projected.append(projected_row)
    return projected, sorted(available - set(variables))


def transform(source: StructuredObject, plan: TransformationPlan) -> TransformationResult:
    """Return a validated transformed object and exact lineage / loss summary.

    No automatic aggregation of observed values is performed. ``aggregate``
    summarizes only *removed structural nodes and edges* by type/count; it does
    not imply that their attributes or sample values were statistically merged.
    """
    if not isinstance(source, StructuredObject) or not isinstance(plan, TransformationPlan):
        _error('transform requires a validated StructuredObject and TransformationPlan')
    doc = source.to_dict()  # all mutations confined to detached data
    node_ids = {node['id'] for node in doc['nodes']}
    keep = set(node_ids)
    collapsed_into: dict[str, str] = {}
    children: dict[str, list[str]] = {}
    parents: dict[str, str] = {}
    depths: dict[str, int] = {}
    if plan.mode in {'subtree', 'depth', 'aggregate'}:
        children, parents, depths = _hierarchy(source)
        if plan.mode == 'subtree':
            keep = _subtree_nodes(plan.roots, children, node_ids)
        else:
            assert plan.max_depth is not None
            keep = {node for node, depth in depths.items() if depth <= plan.max_depth}
            if plan.mode == 'aggregate':
                for node in sorted(node_ids - keep):
                    ancestor = node
                    while depths[ancestor] > plan.max_depth:
                        ancestor = parents[ancestor]
                    collapsed_into[node] = ancestor
    elif plan.mode == 'flat':
        # All node identities and attributes survive, but edges are discarded.
        # A directed_graph with no edges does not falsely claim a hierarchy.
        pass

    retained_nodes = [node for node in doc['nodes'] if node['id'] in keep]
    removed_nodes = sorted(node_ids - keep)
    if plan.mode == 'flat':
        retained_edges = []
    else:
        retained_edges = [edge for edge in doc['edges']
                          if edge['source_id'] in keep and edge['target_id'] in keep]
    retained_edge_ids = {edge['id'] for edge in retained_edges}
    removed_edges = sorted(edge['id'] for edge in doc['edges']
                           if edge['id'] not in retained_edge_ids)

    if plan.mode == 'aggregate':
        kind_by_id = {node['id']: node['kind'] for node in doc['nodes']}
        summaries: dict[str, dict[str, Any]] = defaultdict(lambda: {
            'removed_nodes': 0, 'removed_edges': 0, 'node_kinds': Counter(),
        })
        for node, ancestor in collapsed_into.items():
            summary = summaries[ancestor]
            summary['removed_nodes'] += 1
            summary['node_kinds'][kind_by_id[node]] += 1
        removed_edge_id_set = set(removed_edges)
        for edge in doc['edges']:
            if edge['id'] not in removed_edge_id_set:
                continue
            # The source is the retained boundary node or one of its descendants.
            src = edge['source_id']
            boundary = src if src in keep else collapsed_into[src]
            summaries[boundary]['removed_edges'] += 1
        for node in retained_nodes:
            if node['id'] not in summaries:
                continue
            metadata = dict(node.get('metadata', {}))
            if 'ml_lab_structural_aggregate_v1' in metadata:
                _error('structural aggregation metadata key collision')
            summary = summaries[node['id']]
            metadata['ml_lab_structural_aggregate_v1'] = {
                'removed_nodes': summary['removed_nodes'],
                'removed_edges': summary['removed_edges'],
                'node_kinds': dict(sorted(summary['node_kinds'].items())),
                'value_aggregation': 'none',
            }
            node['metadata'] = metadata

    rows, dropped_variables = _project_observations(doc, plan.variables)
    if len(rows) != len(doc['observations']) or any(
        a['observation_id'] != b['observation_id'] for a, b in zip(rows, doc['observations'])
    ):
        _error('structural transformation must preserve observation identities')

    if plan.mode == 'subtree':
        # An independent retained descendant forest is a tree iff one root
        # remains. This is an explicit validated hierarchy transformation.
        kept_roots = [node for node in keep if parents.get(node) not in keep]
        output_topology = 'tree' if len(kept_roots) == 1 else 'forest'
    elif plan.mode == 'flat':
        output_topology = 'directed_graph'
    else:
        output_topology = doc['topology']['kind']

    no_structural_loss = not removed_nodes and not removed_edges
    no_variable_loss = not dropped_variables
    identical = (plan.mode == 'full' and plan.variables is None)
    if identical:
        result_object = source
    else:
        provenance = dict(doc['provenance'])
        history = provenance.get('structural_transformations', [])
        if not isinstance(history, list):
            _error('invalid structural transformation provenance history')
        provenance['structural_transformations'] = history + [{
            'schema': REPORT_SCHEMA,
            'source_object_id': source.object_id,
            'plan': plan.to_dict(),
        }]
        output = {
            'schema': doc['schema'], 'topology': {'kind': output_topology},
            'nodes': retained_nodes, 'edges': retained_edges,
            'observation_unit': doc['observation_unit'],
            'observations': rows, 'provenance': provenance,
        }
        result_object = StructuredObject.from_dict(output)

    losses = []
    if removed_nodes:
        losses.append('node_identities_and_attributes')
    if removed_edges:
        losses.append('edge_relationships')
    if dropped_variables:
        losses.append('aligned_observation_variables')
    if output_topology != doc['topology']['kind']:
        losses.append('topology_family')
    if plan.mode == 'aggregate' and removed_nodes:
        losses.append('only_structural_counts_aggregated')
    # Do not equate a supplied source ID with an independently verified hash.
    audit = {
        'schema': REPORT_SCHEMA,
        'source_object_id': source.object_id,
        'output_object_id': result_object.object_id,
        'plan': plan.to_dict(),
        'retained_node_ids': sorted(keep),
        'removed_node_ids': removed_nodes,
        'retained_edge_ids': sorted(retained_edge_ids),
        'removed_edge_ids': removed_edges,
        'node_mapping': {name: name if name in keep else None for name in sorted(node_ids)},
        'collapsed_into': dict(sorted(collapsed_into.items())),
        'observation_unit': doc['observation_unit'],
        'input_observation_ids': [row['observation_id'] for row in doc['observations']],
        'output_observation_ids': [row['observation_id'] for row in rows],
        'retained_variables': (list(rows[0]['values']) if rows else []),
        'removed_variables': dropped_variables,
        'losses': losses,
        'lossless': not losses,
        # Only the identity view can be exactly reversed from the output alone.
        'reversible': identical,
        'observation_join_performed': False,
        'source_data_verified': False,
    }
    return TransformationResult(result_object, plan, _canonical(audit))
