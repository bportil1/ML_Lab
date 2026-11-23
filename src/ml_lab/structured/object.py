"""Validated, immutable, domain-independent structured object (ML-UW-28).

This module validates *source structure* and aligned observation cells. It does
not parse XML/ASTs, infer statistical edges, or certify probabilistic models.
No IO, global registry, SQLite access, or third-party dependencies are used.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any

from .contracts import (
    CAPABILITIES, CELL_STATES, CONTRACT_SCHEMA, NULL_REASONS, OBSERVATION_UNITS,
    TOPOLOGIES, StructuredContractError, require_capabilities,
    require_observation_alignment,
)

_TOP_FIELDS = frozenset({
    'schema', 'object_id', 'topology', 'nodes', 'edges', 'observation_unit',
    'observations', 'provenance',
})
_NODE_FIELDS = frozenset({'id', 'kind', 'attributes', 'label', 'source_ref', 'metadata'})
_EDGE_FIELDS = frozenset({
    'id', 'kind', 'source_id', 'target_id', 'ordinal', 'directed',
    'attributes', 'source_ref', 'metadata',
})
_ROW_FIELDS = frozenset({'observation_id', 'values', 'source_ref', 'lineage', 'metadata'})
_CELL_FIELDS = frozenset({
    'status', 'value', 'source_reason', 'occurrences', 'lineage', 'metadata',
})


def _fail(message: str) -> None:
    raise StructuredContractError(message)


def _mapping(value: object, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f'{where} must be a mapping')
    if any(not isinstance(key, str) for key in value):
        _fail(f'{where} keys must be strings')
    return value


def _sequence(value: object, where: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        _fail(f'{where} must be an array')
    return value


def _fields(value: Mapping[str, Any], allowed: frozenset[str],
            required: frozenset[str], where: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        _fail(f'{where} contains unsupported fields: {", ".join(sorted(unknown))}')
    missing = required - set(value)
    if missing:
        _fail(f'{where} is missing fields: {", ".join(sorted(missing))}')


def _identifier(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f'{where} must be a nonempty string')
    return value


def _json_data(value: object, where: str, depth: int = 0) -> Any:
    """Clone a JSON value without accepting NaN, implicit coercion or objects."""
    if depth > 128:
        _fail(f'{where} nesting exceeds 128 levels')
    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            _fail(f'{where} must not contain NaN or Infinity')
        return value
    if isinstance(value, Mapping):
        _mapping(value, where)
        return {k: _json_data(v, f'{where}.{k}', depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_data(v, f'{where}[{i}]', depth + 1) for i, v in enumerate(value)]
    _fail(f'{where} must contain JSON-compatible values')


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(val) for key, val in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(val) for val in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(val) for key, val in value.items()}
    if isinstance(value, tuple):
        return [_thaw(val) for val in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False)


def _validate_cell(value: object, where: str, *, occurrence: bool = False) -> dict:
    cell = _mapping(value, where)
    _fields(cell, _CELL_FIELDS, frozenset({'status', 'value'}), where)
    status = cell['status']
    if not isinstance(status, str) or status not in CELL_STATES:
        _fail(f'{where}: invalid cell state')
    cloned = _json_data(cell, where)
    v = cloned['value']
    if status in NULL_REASONS and v is not None:
        _fail(f'{where}: null state must have a null value')
    if status == 'empty' and v != '':
        _fail(f'{where}: empty state must have an empty-string value')
    if status == 'present' and v is None:
        _fail(f'{where}: present value cannot be null')
    if status == 'present' and v == '':
        _fail(f'{where}: an empty string requires the empty state')
    if 'source_reason' in cloned:
        _identifier(cloned['source_reason'], f'{where}.source_reason')
    if status == 'multiple':
        if occurrence:
            _fail(f'{where}: nested multiple states are not supported')
        if v is not None:
            _fail(f'{where}: multiple must use null value with explicit occurrences')
        occurrences = _sequence(cloned.get('occurrences'), f'{where}.occurrences')
        if not occurrences:
            _fail(f'{where}: multiple requires nonempty occurrences')
        cloned['occurrences'] = [
            _validate_cell(item, f'{where}.occurrences[{i}]', occurrence=True)
            for i, item in enumerate(occurrences)
        ]
    elif 'occurrences' in cloned:
        _fail(f'{where}: occurrences require the multiple state')
    return cloned


def _validate_nodes(raw: object) -> tuple[list[dict], set[str]]:
    nodes: list[dict] = []
    seen: set[str] = set()
    for index, value in enumerate(_sequence(raw, 'nodes')):
        where = f'nodes[{index}]'
        node = _mapping(value, where)
        _fields(node, _NODE_FIELDS, frozenset({'id', 'kind', 'attributes'}), where)
        node_id = _identifier(node['id'], f'{where}.id')
        _identifier(node['kind'], f'{where}.kind')
        _mapping(node['attributes'], f'{where}.attributes')
        if 'label' in node and node['label'] is not None and not isinstance(node['label'], str):
            _fail(f'{where}.label must be a string or null')
        if 'source_ref' in node:
            _mapping(node['source_ref'], f'{where}.source_ref')
        if 'metadata' in node:
            _mapping(node['metadata'], f'{where}.metadata')
        if node_id in seen:
            _fail(f'duplicate node ID: {node_id}')
        seen.add(node_id)
        nodes.append(_json_data(node, where))
    if not nodes:
        _fail('a StructuredObject requires at least one node')
    return nodes, seen


def _validate_edges(raw: object, node_ids: set[str], directed: bool) -> list[dict]:
    edges: list[dict] = []
    seen: set[str] = set()
    for index, value in enumerate(_sequence(raw, 'edges')):
        where = f'edges[{index}]'
        edge = _mapping(value, where)
        _fields(edge, _EDGE_FIELDS,
                frozenset({'id', 'kind', 'source_id', 'target_id'}), where)
        edge_id = _identifier(edge['id'], f'{where}.id')
        _identifier(edge['kind'], f'{where}.kind')
        if edge_id in seen:
            _fail(f'duplicate edge ID: {edge_id}')
        seen.add(edge_id)
        for endpoint in ('source_id', 'target_id'):
            name = _identifier(edge[endpoint], f'{where}.{endpoint}')
            if name not in node_ids:
                _fail(f'{where}: unknown {endpoint} {name!r}')
        ordinal = edge.get('ordinal')
        if ordinal is not None and (type(ordinal) is not int or ordinal < 0):
            _fail(f'{where}.ordinal must be a nonnegative integer or null')
        if 'directed' in edge and (type(edge['directed']) is not bool
                                   or edge['directed'] is not directed):
            _fail(f'{where}.directed conflicts with topology direction')
        for key in ('attributes', 'source_ref', 'metadata'):
            if key in edge:
                _mapping(edge[key], f'{where}.{key}')
        edges.append(_json_data(edge, where))
    return edges


def _validate_observations(raw: object, observation_unit: str) -> tuple[list[dict], bool]:
    observations: list[dict] = []
    observed_ids: set[str] = set()
    previous_variables: frozenset[str] | None = None
    for index, value in enumerate(_sequence(raw, 'observations')):
        where = f'observations[{index}]'
        row = _mapping(value, where)
        _fields(row, _ROW_FIELDS, frozenset({'observation_id', 'values'}), where)
        obs_id = _identifier(row['observation_id'], f'{where}.observation_id')
        if obs_id in observed_ids:
            _fail(f'duplicate observation ID: {obs_id}')
        observed_ids.add(obs_id)
        values = _mapping(row['values'], f'{where}.values')
        if not values:
            _fail(f'{where}.values must not be empty; missing values require explicit null cells')
        for variable_id, cell in values.items():
            _identifier(variable_id, f'{where}.values variable identity')
            _validate_cell(cell, f'{where}.values[{variable_id}]')
        variable_ids = frozenset(values)
        if previous_variables is not None and variable_ids != previous_variables:
            _fail(f'{where}: observations lack aligned variables; use explicit missing cells')
        previous_variables = variable_ids
        if 'source_ref' in row:
            _mapping(row['source_ref'], f'{where}.source_ref')
        if 'metadata' in row:
            _mapping(row['metadata'], f'{where}.metadata')
        observations.append(_json_data(row, where))
    if observations:
        # Reuse the original ML-UW-27 alignment gate for independent consistency.
        require_observation_alignment(observation_unit, observations, previous_variables or ())
    return observations, bool(observations)


def _directed_acyclic(node_ids: set[str], edges: list[dict]) -> bool:
    indegrees = {name: 0 for name in node_ids}
    children: dict[str, list[str]] = {name: [] for name in node_ids}
    for edge in edges:
        indegrees[edge['target_id']] += 1
        children[edge['source_id']].append(edge['target_id'])
    queue = deque(node for node, degree in indegrees.items() if degree == 0)
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for child in children[node]:
            indegrees[child] -= 1
            if indegrees[child] == 0:
                queue.append(child)
    return visited == len(node_ids)


def _undirected_acyclic(node_ids: set[str], edges: list[dict]) -> bool:
    parent = {key: key for key in node_ids}
    def root(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node
    for edge in edges:
        left = root(edge['source_id'])
        right = root(edge['target_id'])
        if left == right:
            return False
        parent[left] = right
    return True


def _derived_capabilities(kind: str, node_ids: set[str], edges: list[dict],
                          aligned: bool) -> frozenset[str]:
    directed = kind != 'undirected_graph'
    caps = {'graph_edges'}
    if directed:
        caps.add('directed_edges')
    acyclic = (_directed_acyclic(node_ids, edges) if directed
               else _undirected_acyclic(node_ids, edges))
    if acyclic:
        caps.add('acyclic')
    if directed:
        indegrees = {key: 0 for key in node_ids}
        ordinals: dict[str, set[int]] = {key: set() for key in node_ids}
        all_ordered = bool(edges)
        for edge in edges:
            indegrees[edge['target_id']] += 1
            ordinal = edge.get('ordinal')
            source = edge['source_id']
            if ordinal is None or ordinal in ordinals[source]:
                all_ordered = False
            else:
                ordinals[source].add(ordinal)
        unique_parent = max(indegrees.values()) <= 1
        if unique_parent and acyclic:
            caps.add('unique_parent')
        if all_ordered:
            caps.add('ordered_children')
        if kind in {'tree', 'forest'}:
            roots = sum(degree == 0 for degree in indegrees.values())
            if not acyclic or not unique_parent:
                _fail(f'{kind} requires an acyclic directed hierarchy with unique parents')
            if kind == 'tree' and roots != 1:
                _fail('tree requires exactly one root')
            if kind == 'forest' and roots < 1:
                _fail('forest requires one or more roots')
            caps.add('rooted_hierarchy')
        if kind == 'dag' and not acyclic:
            _fail('dag topology cannot contain cycles')
    if aligned:
        caps.add('aligned_observations')
    # This describes only the *source* graph. PGM evidence is a separate artifact.
    return frozenset(caps)


def _normalized_payload(raw: object) -> dict:
    payload = _mapping(raw, 'StructuredObject')
    _fields(payload, _TOP_FIELDS,
            frozenset({'schema', 'topology', 'nodes', 'edges',
                       'observation_unit', 'observations', 'provenance'}), 'StructuredObject')
    if payload['schema'] != CONTRACT_SCHEMA:
        _fail(f'unsupported structured schema: {payload["schema"]!r}')
    topology = _mapping(payload['topology'], 'topology')
    _fields(topology, frozenset({'kind', 'capabilities'}),
            frozenset({'kind'}), 'topology')
    kind = topology['kind']
    if not isinstance(kind, str) or kind not in TOPOLOGIES:
        _fail(f'unsupported topology: {kind!r}')
    observation_unit = payload['observation_unit']
    if not isinstance(observation_unit, str) or observation_unit not in OBSERVATION_UNITS:
        _fail('unsupported or missing observation unit')
    provenance = _mapping(payload['provenance'], 'provenance')
    nodes, node_ids = _validate_nodes(payload['nodes'])
    edges = _validate_edges(payload['edges'], node_ids, kind != 'undirected_graph')
    observations, aligned = _validate_observations(payload['observations'], observation_unit)
    derived = _derived_capabilities(kind, node_ids, edges, aligned)
    declared = topology.get('capabilities')
    if declared is not None:
        declaration = _sequence(declared, 'topology.capabilities')
        if any(not isinstance(cap, str) or cap not in CAPABILITIES for cap in declaration):
            _fail('topology.capabilities includes unsupported capability')
        if len(declaration) != len(set(declaration)):
            _fail('topology.capabilities includes duplicate claims')
        if frozenset(declaration) != derived:
            absent = sorted(derived - set(declaration))
            unjustified = sorted(set(declaration) - derived)
            _fail('topology capability mismatch (derived missing: '
                  + ', '.join(absent) + '; unjustified claims: ' + ', '.join(unjustified) + ')')
    if 'object_id' in payload:
        object_id = _identifier(payload['object_id'], 'object_id')
    else:
        # Source identities are independent of input dict order, but respect node,
        # edge and observation list ordering and occurrence multiplicity.
        digest_input = {
            'schema': CONTRACT_SCHEMA, 'topology': {'kind': kind},
            'nodes': nodes, 'edges': edges, 'observation_unit': observation_unit,
            'observations': observations, 'provenance': _json_data(provenance, 'provenance'),
        }
        object_id = 'sha256:' + hashlib.sha256(_canonical_json(digest_input).encode('utf-8')).hexdigest()
    normalized = {
        'schema': CONTRACT_SCHEMA, 'object_id': object_id,
        'topology': {'kind': kind, 'capabilities': sorted(derived)},
        'nodes': nodes, 'edges': edges, 'observation_unit': observation_unit,
        'observations': observations, 'provenance': _json_data(provenance, 'provenance'),
    }
    # Also certify the normalized, complete result is JSON-safe.
    return _json_data(normalized, 'StructuredObject')


def _unique_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            _fail(f'duplicate JSON field: {name}')
        result[name] = value
    return result


def _reject_json_constant(value: str) -> None:
    _fail(f'invalid JSON constant: {value}')


@dataclass(frozen=True, slots=True)
class StructuredObject:
    """Immutable validated graph and observation record, with no source assumptions.

    Construct via :meth:`from_dict` or :meth:`from_json`.  The only constructor
    argument is a JSON-compatible mapping, so callers cannot inject unverified
    capability flags through a dataclass field initializer.
    """

    _payload: Mapping[str, Any] = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, '_payload', _freeze(_normalized_payload(self._payload)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> 'StructuredObject':
        return cls(payload)

    @classmethod
    def from_json(cls, source: str | bytes | bytearray) -> 'StructuredObject':
        if not isinstance(source, (str, bytes, bytearray)):
            _fail('serialized object must be JSON text')
        try:
            payload = json.loads(source, object_pairs_hook=_unique_json_pairs,
                                 parse_constant=_reject_json_constant)
        except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
            raise StructuredContractError(f'invalid structured JSON: {exc}') from exc
        return cls(payload)

    @property
    def object_id(self) -> str:
        return self._payload['object_id']

    @property
    def schema(self) -> str:
        return self._payload['schema']

    @property
    def topology(self) -> Mapping[str, Any]:
        return self._payload['topology']

    @property
    def capabilities(self) -> frozenset[str]:
        return frozenset(self._payload['topology']['capabilities'])

    @property
    def nodes(self) -> tuple[Mapping[str, Any], ...]:
        return self._payload['nodes']

    @property
    def edges(self) -> tuple[Mapping[str, Any], ...]:
        return self._payload['edges']

    @property
    def observation_unit(self) -> str:
        return self._payload['observation_unit']

    @property
    def observations(self) -> tuple[Mapping[str, Any], ...]:
        return self._payload['observations']

    @property
    def provenance(self) -> Mapping[str, Any]:
        return self._payload['provenance']

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, writable JSON-compatible copy."""
        return _thaw(self._payload)

    def to_json(self, *, indent: int | None = None) -> str:
        """Deterministic JSON; no writes or external side effects."""
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False,
                          separators=(',', ':') if indent is None else None,
                          indent=indent, allow_nan=False)

    def require(self, operation: str) -> None:
        """Gate a scientific operation using *validated* capabilities."""
        require_capabilities(self.capabilities, operation)

    def require_aligned(self, variables: Sequence[str]) -> None:
        """Gate paired observations; does not certify independence or sample size."""
        require_observation_alignment(self.observation_unit, self.to_dict()['observations'],
                                      variables)
