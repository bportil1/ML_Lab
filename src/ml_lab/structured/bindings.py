"""SU-05: source-pinned, domain-independent observation-to-structure binding.

A source graph never constitutes a sample population. Adapters must first supply
explicit, aligned observation rows; this module binds their variable identities
back to original node identities without reading files or interpreting domains.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

from .contracts import OBSERVATION_UNITS, StructuredContractError
from .object import StructuredObject

PLAN_SCHEMA = 'ml-lab.observation-binding-plan@1'
RESULT_SCHEMA = 'ml-lab.observation-binding@1'
_POLICIES = frozenset({'preserve', 'scalar_only', 'mean_numeric'})
_MAX_ROWS = 200_000


def _error(message: str) -> None:
    raise StructuredContractError(message)


def _name(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _error(f'{where} must be a nonempty string')
    return value


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def source_sha256(source: StructuredObject) -> str:
    """Fingerprint complete validated source bytes, not a caller-chosen object_id."""
    if not isinstance(source, StructuredObject):
        _error('source must be a validated StructuredObject')
    return 'sha256:' + hashlib.sha256(source.to_json().encode('utf-8')).hexdigest()


@dataclass(frozen=True, slots=True)
class VariableBinding:
    """One aligned observation column explicitly mapped to source node ID(s).

    ``merged_sources`` explicitly authorizes many source nodes representing a
    single observed variable; no occurrence-to-node allocation is inferred.
    """

    variable_id: str
    node_ids: tuple[str, ...]
    occurrence_policy: str = 'preserve'
    merged_sources: bool = False

    def __post_init__(self) -> None:
        _name(self.variable_id, 'variable_id')
        if (not isinstance(self.node_ids, tuple) or not self.node_ids or
                any(not isinstance(node, str) or not node.strip() for node in self.node_ids) or
                len(set(self.node_ids)) != len(self.node_ids)):
            _error('node_ids must be nonempty, unique, exact source node identities')
        if not isinstance(self.occurrence_policy, str) or self.occurrence_policy not in _POLICIES:
            _error('unsupported repeated-occurrence policy')
        if type(self.merged_sources) is not bool:
            _error('merged_sources must be boolean')
        if len(self.node_ids) > 1 and not self.merged_sources:
            _error('multiple node IDs require explicit merged_sources=true')

    def to_dict(self) -> dict[str, Any]:
        return {'variable_id': self.variable_id, 'node_ids': list(self.node_ids),
                'occurrence_policy': self.occurrence_policy,
                'merged_sources': self.merged_sources}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> 'VariableBinding':
        if not isinstance(raw, Mapping) or set(raw) - {
            'variable_id', 'node_ids', 'occurrence_policy', 'merged_sources'
        } or not {'variable_id', 'node_ids'}.issubset(raw):
            _error('invalid variable binding fields')
        ids = raw['node_ids']
        if not isinstance(ids, list):
            _error('node_ids must be an array')
        return cls(raw['variable_id'], tuple(ids),
                   raw.get('occurrence_policy', 'preserve'),
                   raw.get('merged_sources', False))


@dataclass(frozen=True, slots=True)
class ObservationBindingPlan:
    """Explicit, reproducible binding specification, separate from source data."""

    source_digest: str
    observation_unit: str
    variables: tuple[VariableBinding, ...]
    group_variables: tuple[str, ...] = ()
    weight_variable: str | None = None

    def __post_init__(self) -> None:
        if (not isinstance(self.source_digest, str) or
                len(self.source_digest) != 71 or not self.source_digest.startswith('sha256:') or
                any(c not in '0123456789abcdef' for c in self.source_digest[7:])):
            _error('source_digest must be a lowercase sha256 fingerprint')
        if self.observation_unit not in OBSERVATION_UNITS:
            _error('invalid observation unit')
        if (not isinstance(self.variables, tuple) or not self.variables or
                any(not isinstance(v, VariableBinding) for v in self.variables) or
                len({v.variable_id for v in self.variables}) != len(self.variables)):
            _error('variable bindings must be nonempty and have unique column IDs')
        if not isinstance(self.group_variables, tuple):
            _error('group_variables must be a tuple')
        for name in self.group_variables:
            _name(name, 'group variable')
        if len(set(self.group_variables)) != len(self.group_variables):
            _error('group_variables must be unique')
        if self.weight_variable is not None:
            _name(self.weight_variable, 'weight variable')

    def to_dict(self) -> dict[str, Any]:
        return {'schema': PLAN_SCHEMA, 'source_digest': self.source_digest,
                'observation_unit': self.observation_unit,
                'variables': [v.to_dict() for v in self.variables],
                'group_variables': list(self.group_variables),
                'weight_variable': self.weight_variable}

    def to_json(self) -> str:
        return _canonical(self.to_dict())

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> 'ObservationBindingPlan':
        if (not isinstance(raw, Mapping) or
                set(raw) != {'schema', 'source_digest', 'observation_unit',
                             'variables', 'group_variables', 'weight_variable'} or
                raw.get('schema') != PLAN_SCHEMA):
            _error('unsupported binding plan schema or fields')
        if not isinstance(raw['variables'], list) or not isinstance(raw['group_variables'], list):
            _error('binding plan variables and group_variables must be arrays')
        return cls(raw['source_digest'], raw['observation_unit'],
                   tuple(VariableBinding.from_dict(v) for v in raw['variables']),
                   tuple(raw['group_variables']), raw['weight_variable'])

    @classmethod
    def from_json(cls, text: str) -> 'ObservationBindingPlan':
        if not isinstance(text, str):
            _error('binding plan JSON must be text')
        try:
            # Duplicate keys are rejected; normal json.loads silently overwrites them.
            def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
                data: dict[str, Any] = {}
                for key, value in items:
                    if key in data:
                        _error(f'duplicate binding JSON field: {key}')
                    data[key] = value
                return data
            raw = json.loads(text, object_pairs_hook=pairs,
                             parse_constant=lambda c: _error(f'invalid JSON constant: {c}'))
        except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
            raise StructuredContractError(f'invalid binding plan JSON: {exc}') from exc
        return cls.from_dict(raw)


@dataclass(frozen=True, slots=True)
class ObservationBindingResult:
    """Detached JSON-compatible report and projected rows; no source mutation."""

    _json: str

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._json)

    def to_json(self) -> str:
        return self._json

    @property
    def report(self) -> dict[str, Any]:
        doc = self.to_dict()
        doc.pop('rows')
        return doc

    @property
    def rows(self) -> list[dict[str, Any]]:
        return self.to_dict()['rows']


def _resolved_cell(cell: Mapping[str, Any], rule: VariableBinding,
                   observation_id: str) -> dict[str, Any]:
    result = json.loads(_canonical(cell))
    if result['status'] != 'multiple':
        if rule.occurrence_policy == 'mean_numeric' and result['status'] == 'present':
            number = result['value']
            if type(number) not in (int, float):
                _error(f'{observation_id}: mean_numeric requires numeric observations')
            try:
                finite = math.isfinite(float(number))
            except OverflowError as exc:
                raise StructuredContractError(f'{observation_id}: numeric aggregate overflow') from exc
            if not finite:
                _error(f'{observation_id}: mean_numeric requires finite numbers')
        return result
    if rule.occurrence_policy == 'preserve':
        return result
    if rule.occurrence_policy == 'scalar_only':
        _error(f'{observation_id}: {rule.variable_id} has repeated values; scalar_only rejects them')
    occurrences = result['occurrences']
    # Do not silently turn missing/nil/unknown occurrences into zeros or drop
    # them during aggregation. A caller can first define a different explicit
    # source-level missingness/aggregation policy in an adapter.
    if any(part['status'] != 'present' or type(part['value']) not in (int, float)
           for part in occurrences):
        _error(f'{observation_id}: {rule.variable_id} mean_numeric requires all occurrences numeric and present')
    try:
        mean = math.fsum(float(part['value']) / len(occurrences) for part in occurrences)
    except OverflowError as exc:
        raise StructuredContractError(f'{observation_id}: numeric aggregate overflow') from exc
    if not math.isfinite(mean):
        _error(f'{observation_id}: nonfinite mean_numeric aggregate')
    # Preserve exact source cells (including occurrence order and null lineage)
    # in the derived result as well as in the pinned immutable source artifact.
    return {'status': 'present', 'value': mean,
            'lineage': {'original_multiple_cell': result},
            'metadata': {'aggregation': 'mean_numeric',
                         'occurrence_count': len(occurrences),
                         'source_status': 'multiple'}}


def _branches(source: StructuredObject, rules: tuple[VariableBinding, ...]) -> list[dict[str, Any]] | None:
    if 'rooted_hierarchy' not in source.capabilities:
        return None
    children: dict[str, list[str]] = {node['id']: [] for node in source.nodes}
    parent: dict[str, str] = {}
    for edge in source.edges:
        children[edge['source_id']].append(edge['target_id'])
        parent[edge['target_id']] = edge['source_id']
    # Accumulate mapped variable identities upward in one pass. Avoid recursion
    # so very deep hierarchies cannot overflow Python's recursion limit.
    own: dict[str, set[str]] = {key: set() for key in children}
    for binding in rules:
        for node_id in binding.node_ids:
            own[node_id].add(binding.variable_id)
    roots = sorted(set(children) - set(parent))
    stack = [(root, False) for root in roots]
    descendant: dict[str, set[str]] = {}
    cumulative_assignments = 0
    while stack:
        node, done = stack.pop()
        if not done:
            stack.append((node, True))
            stack.extend((child, False) for child in children[node])
        else:
            variables = set(own[node])
            for child in children[node]:
                variables.update(descendant[child])
            cumulative_assignments += len(variables)
            if cumulative_assignments > 2_000_000:
                _error('branch inventory exceeds supported node-variable assignment limit')
            descendant[node] = variables
    return [{'node_id': node_id, 'direct_variable_ids': sorted(own[node_id]),
             'subtree_variable_ids': sorted(descendant[node_id]),
             'subtree_variables_are_statistical_samples': False}
            for node_id in sorted(children)]


def bind_observations(source: StructuredObject, plan: ObservationBindingPlan) -> ObservationBindingResult:
    """Validate explicit binding and project observations with no inferred joins.

    The result is a *binding inventory*, not a branch variance/MI estimate.
    Source nodes never become additional independent observations.
    """
    if not isinstance(source, StructuredObject) or not isinstance(plan, ObservationBindingPlan):
        _error('binding requires a StructuredObject and ObservationBindingPlan')
    if source_sha256(source) != plan.source_digest:
        _error('source fingerprint mismatch: binding plan refers to a stale structure')
    if source.observation_unit != plan.observation_unit:
        _error('observation unit mismatch')
    node_ids = {node['id'] for node in source.nodes}
    if len(plan.variables) > 2_000:
        _error('binding exceeds 2000 mapped observation variables')
    for rule in plan.variables:
        missing = set(rule.node_ids) - node_ids
        if missing:
            _error(f'{rule.variable_id}: unknown source node IDs: {", ".join(sorted(missing))}')
    if len(source.observations) > _MAX_ROWS:
        _error(f'binding exceeds {_MAX_ROWS} observation rows')
    observed_names = set(source.observations[0]['values']) if source.observations else set()
    required = {v.variable_id for v in plan.variables} | set(plan.group_variables)
    if plan.weight_variable:
        required.add(plan.weight_variable)
    if source.observations and not required.issubset(observed_names):
        _error(f'unknown aligned observation columns: {", ".join(sorted(required - observed_names))}')
    status_counts = {v.variable_id: Counter() for v in plan.variables}
    repeated = Counter()
    occurrence_counts = Counter()
    null_reasons: dict[str, Counter] = {v.variable_id: Counter() for v in plan.variables}
    group_counts: Counter[str] = Counter()
    unavailable_groups = 0
    weights_sum = 0.0
    projected_rows = []
    # Work on a detached JSON-compatible copy. StructuredObject exposes frozen
    # mapping proxies intentionally; never mutate or serialize those directly.
    for row in source.to_dict()['observations']:
        observation_id = row['observation_id']
        values = row['values']
        projected = {}
        for rule in plan.variables:
            raw = values[rule.variable_id]
            status_counts[rule.variable_id][raw['status']] += 1
            if raw.get('source_reason'):
                null_reasons[rule.variable_id][raw['source_reason']] += 1
            if raw['status'] == 'multiple':
                repeated[rule.variable_id] += 1
                occurrence_counts[rule.variable_id] += len(raw['occurrences'])
                for occurrence in raw['occurrences']:
                    if occurrence.get('source_reason'):
                        null_reasons[rule.variable_id][occurrence['source_reason']] += 1
            projected[rule.variable_id] = _resolved_cell(raw, rule, observation_id)
        group_key: list[dict[str, Any]] | None = []
        for name in plan.group_variables:
            cell = values[name]
            if cell['status'] not in ('present', 'empty'):
                group_key = None
                break
            if type(cell['value']) not in (str, int, float, bool):
                _error(f'{observation_id}: group values must be scalar')
            if type(cell['value']) is float and not math.isfinite(cell['value']):
                _error(f'{observation_id}: group values must be finite')
            group_key.append({'variable_id': name, 'status': cell['status'], 'value': cell['value']})
        if group_key is None:
            unavailable_groups += 1
        elif plan.group_variables:
            group_counts[_canonical(group_key)] += 1
        weight = 1.0
        if plan.weight_variable is not None:
            cell = values[plan.weight_variable]
            if cell['status'] != 'present' or type(cell['value']) not in (int, float):
                _error(f'{observation_id}: sample weight must be finite positive numeric')
            try:
                weight = float(cell['value'])
            except OverflowError as exc:
                raise StructuredContractError(f'{observation_id}: sample weight overflow') from exc
            if not math.isfinite(weight) or weight <= 0:
                _error(f'{observation_id}: sample weight must be finite positive numeric')
        weights_sum += weight
        if not math.isfinite(weights_sum):
            _error('nonfinite cumulative sample weights')
        projected_rows.append({'observation_id': observation_id,
                               'values': projected, 'group_key': group_key,
                               'weight': weight})
    variable_reports = []
    for rule in plan.variables:
        variable_reports.append({**rule.to_dict(),
            'aligned_rows': len(projected_rows),
            'original_status_counts': dict(sorted(status_counts[rule.variable_id].items())),
            'null_source_reason_counts': dict(sorted(null_reasons[rule.variable_id].items())),
            'repeated_rows': repeated[rule.variable_id],
            'recorded_occurrences_in_repeated_rows': occurrence_counts[rule.variable_id],
            'merged_node_identity_ambiguous': len(rule.node_ids) > 1})
    branches = _branches(source, plan.variables)
    payload = {
        'schema': RESULT_SCHEMA,
        'source_object_id': source.object_id,
        'source_digest': plan.source_digest,
        'plan': plan.to_dict(),
        'observation_unit': source.observation_unit,
        'support': {'aligned_rows': len(projected_rows),
                    'distinct_observation_ids': len(projected_rows),
                    'sample_weight_total': weights_sum,
                    'group_count': len(group_counts) if plan.group_variables else None,
                    'group_support': [
                        {'group_key': json.loads(key), 'row_count': count}
                        for key, count in sorted(group_counts.items())
                    ],
                    'unavailable_group_rows': unavailable_groups if plan.group_variables else 0,
                    'independence_verified': False,
                    'sampling_unit_not_inferred_from_nodes': True},
        'variables': variable_reports,
        'branches': branches,
        'limitations': [
            'bindings_are_declared_not_inferred_from_source_edges',
            'source_graph_nodes_are_not_statistical_sample_units_without_explicit_observations',
            'source_structure_does_not_establish_independent_sampling_or_causal_dependence',
            'mapping_is_explicit_not_empirically_proven_by_source_lineage',
            'grouping_and_weights_do_not_automatically_make_a_leakage_safe_study',
            'branch_scope_is_structural_membership_not_explained_variance_or_information',
        ] + (['no_empirical_observations_supplied'] if not projected_rows else [])
          + (['branch_descendants_not_defined_for_nonhierarchy'] if branches is None else [])
          + (['merged_source_nodes_have_no_occurrence_level_allocation']
             if any(v.merged_sources for v in plan.variables) else []),
        'rows': projected_rows,
    }
    return ObservationBindingResult(_canonical(payload))
