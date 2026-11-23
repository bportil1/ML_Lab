"""Explicit, source-independent Bayesian-network *specification* (ML-UW-33).

Only validates a finite categorical DAG and optionally supplied normalized CPTs.
No probabilities are estimated, no inference is performed, and source edges are
never converted into probabilistic arcs unless the caller names them explicitly.
No IO, global registry, third-party dependencies or database access.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
from itertools import product
import json
import math
from types import MappingProxyType
from typing import Any

from .contracts import StructuredContractError
from .object import StructuredObject

MODEL_SCHEMA = 'ml-lab.probabilistic-model@1'
MODEL_TYPE = 'discrete_bayesian_network'
MISSING_POLICIES = frozenset({'complete_case', 'reject'})
MAX_VARIABLES = 64
MAX_STATES = 64
MAX_CPT_ROWS = 4096
_TOL = 1e-9


def _error(message: str) -> None:
    raise StructuredContractError(message)


def _object(value: object, where: str) -> dict:
    if not isinstance(value, Mapping) or any(type(k) is not str for k in value):
        _error(f'{where} must be an object with string keys')
    return dict(value)


def _array(value: object, where: str) -> list:
    if not isinstance(value, (tuple, list)):
        _error(f'{where} must be an array')
    return list(value)


def _only(raw: Mapping, allowed: set[str], required: set[str], where: str) -> None:
    if set(raw) - allowed:
        _error(f'{where} has unsupported fields: {sorted(set(raw) - allowed)}')
    if required - set(raw):
        _error(f'{where} is missing fields: {sorted(required - set(raw))}')


def _name(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _error(f'{where} must be a nonempty string')
    return value


def _json_copy(raw: object, where: str, depth: int = 0) -> Any:
    """Strict clone; never coerce arbitrary Python objects or non-finite floats."""
    if depth > 128:
        _error(f'{where} nesting exceeds 128')
    if raw is None or isinstance(raw, (str, bool)) or type(raw) is int:
        return raw
    if type(raw) is float:
        if not math.isfinite(raw):
            _error(f'{where} contains a nonfinite number')
        return raw
    if isinstance(raw, Mapping):
        return {k: _json_copy(v, f'{where}.{k}', depth + 1)
                for k, v in _object(raw, where).items()}
    if isinstance(raw, (list, tuple)):
        return [_json_copy(v, f'{where}[{i}]', depth + 1)
                for i, v in enumerate(raw)]
    _error(f'{where} is not JSON-compatible')


def _freeze(raw: Any) -> Any:
    if isinstance(raw, dict):
        return MappingProxyType({k: _freeze(v) for k, v in raw.items()})
    if isinstance(raw, list):
        return tuple(_freeze(v) for v in raw)
    return raw


def _thaw(raw: Any) -> Any:
    if isinstance(raw, Mapping):
        return {k: _thaw(v) for k, v in raw.items()}
    if isinstance(raw, tuple):
        return [_thaw(v) for v in raw]
    return raw


def _encode(raw: object) -> str:
    return json.dumps(raw, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False)


def source_reference(source: StructuredObject) -> dict[str, str]:
    """Reference both stable identity and exact verified source content."""
    if not isinstance(source, StructuredObject):
        _error('a validated StructuredObject is required for model specification')
    return {'object_id': source.object_id,
            'sha256': hashlib.sha256(source.to_json().encode('utf-8')).hexdigest()}


def _variables(raw: object, source: StructuredObject) -> tuple[list[dict], dict[str, dict]]:
    variables: list[dict] = []
    ids: set[str] = set()
    observation_keys: set[str] = set()
    source_nodes = {node['id'] for node in source.nodes}
    for i, item in enumerate(_array(raw, 'variables')):
        where = f'variables[{i}]'
        var = _object(item, where)
        _only(var, {'id', 'states', 'observed_variable', 'source_node_ids', 'metadata'},
              {'id', 'states', 'observed_variable'}, where)
        vid = _name(var['id'], where + '.id')
        observed = _name(var['observed_variable'], where + '.observed_variable')
        if vid in ids or observed in observation_keys:
            _error('model variables and observed-variable mappings must be unique')
        ids.add(vid)
        observation_keys.add(observed)
        states = _array(var['states'], where + '.states')
        if not 2 <= len(states) <= MAX_STATES or any(type(s) is not str for s in states):
            _error(f'{where}.states requires 2 to {MAX_STATES} string labels')
        if len(set(states)) != len(states):
            _error(f'{where}.states must be unique (including the empty-string label)')
        refs = _array(var.get('source_node_ids', []), where + '.source_node_ids')
        if any(type(x) is not str for x in refs) or len(refs) != len(set(refs)) or any(x not in source_nodes for x in refs):
            _error(f'{where}.source_node_ids must reference distinct source nodes')
        metadata = _json_copy(_object(var.get('metadata', {}), where + '.metadata'), where + '.metadata')
        variables.append({'id': vid, 'states': states, 'observed_variable': observed,
                          'source_node_ids': refs, 'metadata': metadata})
    if not 1 <= len(variables) <= MAX_VARIABLES:
        _error(f'model requires 1 to {MAX_VARIABLES} variables')
    if not source.observations:
        _error('model specification requires explicitly aligned source observations')
    source.require_aligned(tuple(observation_keys))
    return variables, {v['id']: v for v in variables}


def _arcs(raw: object, variables: dict[str, dict], source: StructuredObject
          ) -> tuple[list[dict], dict[str, tuple[str, ...]], list[str]]:
    arcs: list[dict] = []
    seen: set[tuple[str, str]] = set()
    parents: dict[str, list[str]] = {name: [] for name in variables}
    children: dict[str, list[str]] = {name: [] for name in variables}
    source_edges = {e['id']: e for e in source.edges}
    for i, item in enumerate(_array(raw, 'arcs')):
        where = f'arcs[{i}]'
        arc = _object(item, where)
        _only(arc, {'parent', 'child', 'source_edge_ids'}, {'parent', 'child'}, where)
        a, b = _name(arc['parent'], where + '.parent'), _name(arc['child'], where + '.child')
        if a not in variables or b not in variables or a == b or (a, b) in seen:
            _error(f'{where} has unknown, duplicate or self-loop model arc')
        seen.add((a, b))
        refs = _array(arc.get('source_edge_ids', []), where + '.source_edge_ids')
        if any(type(x) is not str for x in refs) or len(refs) != len(set(refs)):
            _error(f'{where}.source_edge_ids must be distinct string IDs')
        for ref in refs:
            if type(ref) is not str or ref not in source_edges:
                _error(f'{where} references an unknown source edge')
            source_edge = source_edges[ref]
            from_nodes = variables[a]['source_node_ids']
            to_nodes = variables[b]['source_node_ids']
            left, right = source_edge['source_id'], source_edge['target_id']
            matches = left in from_nodes and right in to_nodes
            if source.topology['kind'] == 'undirected_graph':
                matches = matches or (right in from_nodes and left in to_nodes)
            if not matches:
                _error(f'{where} source edge does not connect the mapped model variables')
        parents[b].append(a)
        children[a].append(b)
        arcs.append({'parent': a, 'child': b, 'source_edge_ids': refs})
    # Kahn validation over the *model* graph; the source graph is irrelevant here.
    degrees = {k: len(v) for k, v in parents.items()}
    ready = deque(k for k in variables if degrees[k] == 0)
    order: list[str] = []
    while ready:
        name = ready.popleft()
        order.append(name)
        for child in children[name]:
            degrees[child] -= 1
            if degrees[child] == 0:
                ready.append(child)
    if len(order) != len(variables):
        _error('probabilistic model arcs must form a directed acyclic graph (DAG)')
    position = {k: i for i, k in enumerate(variables)}
    return sorted(arcs, key=lambda e: (position[e['parent']], position[e['child']])), {
        key: tuple(sorted(values, key=lambda name: position[name]))
        for key, values in parents.items()
    }, order


def _conditionals(raw: object, variables: dict[str, dict], parents: dict[str, tuple[str, ...]]) -> list[dict]:
    given: dict[str, dict] = {}
    for i, item in enumerate(_array(raw, 'conditionals')):
        where = f'conditionals[{i}]'
        cond = _object(item, where)
        _only(cond, {'variable_id', 'family', 'status', 'rows'},
              {'variable_id', 'family', 'status', 'rows'}, where)
        name = _name(cond['variable_id'], where + '.variable_id')
        if name not in variables or name in given:
            _error(f'{where} references an unknown or duplicate model variable')
        if cond['family'] != 'categorical':
            _error(f'{where}: unsupported conditional family; only categorical is specified in v1')
        if cond['status'] not in ('unfitted', 'specified'):
            _error(f'{where}: conditional status must be unfitted or specified')
        configurations = math.prod(len(variables[parent]['states']) for parent in parents[name])
        if configurations > MAX_CPT_ROWS:
            _error(f'{name}: conditional table requires {configurations} rows; limit is {MAX_CPT_ROWS}')
        raw_rows = _array(cond['rows'], where + '.rows')
        if cond['status'] == 'unfitted':
            if raw_rows:
                _error(f'{name}: unfitted conditional must not carry probabilities')
            rows = []
        else:
            if len(raw_rows) != configurations:
                _error(f'{name}: specified table requires all {configurations} parent assignments')
            indexed: dict[tuple[str, ...], dict] = {}
            for j, item_row in enumerate(raw_rows):
                row_where = f'{where}.rows[{j}]'
                row = _object(item_row, row_where)
                _only(row, {'given', 'probabilities'}, {'given', 'probabilities'}, row_where)
                assignments = _object(row['given'], row_where + '.given')
                if set(assignments) != set(parents[name]):
                    _error(f'{row_where}: parent assignment must match model parents exactly')
                if any(type(assignments[parent]) is not str or
                       assignments[parent] not in variables[parent]['states']
                       for parent in parents[name]):
                    _error(f'{row_where}: parent assignment is outside declared state domains')
                key = tuple(assignments[parent] for parent in parents[name])
                if key in indexed:
                    _error(f'{row_where}: duplicate parent assignment')
                probabilities = _object(row['probabilities'], row_where + '.probabilities')
                if set(probabilities) != set(variables[name]['states']):
                    _error(f'{row_where}: probabilities must cover exactly the child states')
                if any(type(v) not in (int, float) or v < 0 or v > 1 or not math.isfinite(v)
                       for v in probabilities.values()):
                    _error(f'{row_where}: probabilities must be finite numbers in [0, 1]')
                if not math.isclose(math.fsum(probabilities.values()), 1.0,
                                    rel_tol=0, abs_tol=_TOL):
                    _error(f'{row_where}: conditional probabilities must sum to 1')
                indexed[key] = {'given': {p: assignments[p] for p in parents[name]},
                                'probabilities': {s: float(probabilities[s])
                                                  for s in variables[name]['states']}}
            states = [variables[p]['states'] for p in parents[name]]
            rows = [indexed[key] for key in product(*states)]
        given[name] = {'variable_id': name, 'family': 'categorical',
                       'status': cond['status'], 'rows': rows}
    if set(given) != set(variables):
        _error('every model variable requires an explicit conditional specification')
    return [given[name] for name in variables]


def _support(source: StructuredObject, variables: list[dict], missing_policy: str) -> dict:
    counts = {var['id']: {'observed': 0, 'unavailable': 0} for var in variables}
    complete = 0
    for observation in source.observations:
        full = True
        for var in variables:
            cell = observation['values'][var['observed_variable']]
            state = cell['status']
            value = cell['value']
            if state in ('present', 'empty'):
                if type(value) is not str or value not in var['states']:
                    _error(f"observation {observation['observation_id']}: value for {var['id']} "
                           'is outside the finite string domain (explicit binning required)')
                counts[var['id']]['observed'] += 1
            else:
                full = False
                counts[var['id']]['unavailable'] += 1
                if missing_policy == 'reject':
                    _error(f"observation {observation['observation_id']}: missing or multiple "
                           f"value for {var['id']} with reject missingness policy")
        if full:
            complete += 1
    return {'rows': len(source.observations), 'complete_case_rows': complete,
            'excluded_rows': len(source.observations) - complete,
            'per_variable': counts, 'sample_independence_verified': False}


def _normalize(raw: object, source: StructuredObject) -> dict:
    record = _object(raw, 'probabilistic model')
    _only(record, {'schema', 'model_type', 'model_id', 'source', 'variables', 'arcs',
                   'conditionals', 'missing_policy', 'provenance'},
          {'schema', 'model_type', 'variables', 'arcs', 'conditionals', 'missing_policy'},
          'probabilistic model')
    if record['schema'] != MODEL_SCHEMA or record['model_type'] != MODEL_TYPE:
        _error('unsupported probabilistic model schema or type')
    reference = source_reference(source)
    if 'source' in record and record['source'] != reference:
        _error('model source reference does not match the validated StructuredObject')
    if record['missing_policy'] not in MISSING_POLICIES:
        _error('missing_policy must be complete_case or reject')
    variables, by_id = _variables(record['variables'], source)
    arcs, parents, order = _arcs(record['arcs'], by_id, source)
    conditionals = _conditionals(record['conditionals'], by_id, parents)
    support = _support(source, variables, record['missing_policy'])
    provenance = _json_copy(_object(record.get('provenance', {}), 'provenance'), 'provenance')
    normalized = {'schema': MODEL_SCHEMA, 'model_type': MODEL_TYPE,
                  'source': reference, 'variables': variables, 'arcs': arcs,
                  'conditionals': conditionals,
                  'missing_policy': record['missing_policy'], 'provenance': provenance}
    identifier = 'sha256:' + hashlib.sha256(_encode(normalized).encode('utf-8')).hexdigest()
    if 'model_id' in record and record['model_id'] != identifier:
        _error('model_id conflicts with content-derived model identity')
    normalized['model_id'] = identifier
    # Support is an observation-specific audit, not another model parameter or prior.
    normalized['audit'] = {
        'support': support, 'model_topological_order': order,
        'root_variables': [v for v in by_id if not parents[v]],
        'parent_variables': {v: list(parents[v]) for v in by_id},
        'parameter_status': ('specified' if all(c['status'] == 'specified' for c in conditionals)
                             else 'unfitted' if all(c['status'] == 'unfitted' for c in conditionals)
                             else 'partial'),
        'assumptions': {
            'factorization': 'directed_acyclic_categorical',
            'source_graph_is_probability_model': False,
            'source_edges_mapped_explicitly': sum(bool(arc['source_edge_ids']) for arc in arcs),
            'observation_independence': 'not_verified',
            'missingness_mechanism': 'not_inferred',
            'inference_implemented': False,
            'fitting_implemented': False,
        },
    }
    return normalized


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict:
    result: dict = {}
    for key, val in pairs:
        if key in result:
            _error(f'duplicate JSON field: {key}')
        result[key] = val
    return result


def _reject_constant(value: str) -> None:
    _error(f'nonfinite JSON constant: {value}')


@dataclass(frozen=True, slots=True)
class ProbabilisticModelSpec:
    """Immutable, separately validated Bayesian-network contract.

    The source StructuredObject is mandatory at construction and reload, so
    model specifications cannot silently switch to a different dataset. This
    is a *specification*, not a fitted model or a statistical inference engine.
    """

    _payload: Mapping[str, Any] = field(repr=False)
    _source: StructuredObject = field(repr=False, compare=False)
    _audit: Mapping[str, Any] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        normalized = _normalize(self._payload, self._source)
        audit = normalized.pop('audit')
        object.__setattr__(self, '_payload', _freeze(normalized))
        object.__setattr__(self, '_audit', _freeze(audit))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, source: StructuredObject) -> 'ProbabilisticModelSpec':
        return cls(payload, source)

    @classmethod
    def from_json(cls, serialized: str | bytes | bytearray, *, source: StructuredObject) -> 'ProbabilisticModelSpec':
        if not isinstance(serialized, (str, bytes, bytearray)):
            _error('model JSON must be text')
        try:
            parsed = json.loads(serialized, object_pairs_hook=_unique_pairs,
                                parse_constant=_reject_constant)
        except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
            raise StructuredContractError(f'invalid probabilistic model JSON: {exc}') from exc
        return cls.from_dict(parsed, source=source)

    @property
    def model_id(self) -> str:
        return self._payload['model_id']

    @property
    def source_reference(self) -> dict[str, str]:
        return _thaw(self._payload['source'])

    @property
    def audit(self) -> dict:
        return _thaw(self._audit)

    def to_dict(self) -> dict:
        return _thaw(self._payload)

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False,
                          separators=(',', ':') if indent is None else None,
                          indent=indent, allow_nan=False)

    def require_fully_specified(self) -> None:
        """Only checks table completeness; does not perform/authorize inference."""
        if self._audit['parameter_status'] != 'specified':
            _error('all categorical conditionals must be specified before inference')
