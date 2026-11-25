"""SU-06: descriptive, source-linked uncertainty over SU-05 aligned observations.

No source edge is a sampling unit, a statistical grouping assumption, or a
variance allocation. All estimates are complete-case descriptive quantities;
no confidence intervals, independence, or predictive utility are implied.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

from .bindings import ObservationBindingResult, RESULT_SCHEMA as BINDING_SCHEMA
from .contracts import StructuredContractError

PLAN_SCHEMA = 'ml-lab.branch-uncertainty-plan@1'
RESULT_SCHEMA = 'ml-lab.branch-uncertainty@1'
_SCALES = frozenset({'raw', 'zscore', 'declared'})
_OBJECTIVES = frozenset({'per_target', 'weighted_variance', 'reconstruction_mse'})
_MAX_DISTINCT = 5_000
_MAX_GROUP_DISPLAY = 2_000


def _fail(message: str) -> None:
    raise StructuredContractError(message)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False)


def binding_sha256(binding: ObservationBindingResult) -> str:
    if not isinstance(binding, ObservationBindingResult):
        _fail('branch analysis requires a verified SU-05 ObservationBindingResult')
    return 'sha256:' + hashlib.sha256(binding.to_json().encode('utf-8')).hexdigest()


def _digest(value: object) -> None:
    if (not isinstance(value, str) or len(value) != 71 or
            not value.startswith('sha256:') or
            any(c not in '0123456789abcdef' for c in value[7:])):
        _fail('binding_digest must be a lowercase SHA-256 fingerprint')


def _names(value: object, where: str, *, required: bool = False) -> None:
    if (not isinstance(value, tuple) or
            (required and not value) or
            any(not isinstance(x, str) or not x.strip() for x in value) or
            len(set(value)) != len(value)):
        _fail(f'{where} must be a tuple of unique nonempty variable names')


def _positive_pairs(value: object, where: str) -> dict[str, float]:
    if not isinstance(value, tuple):
        _fail(f'{where} must be a tuple of (variable, positive number) pairs')
    result: dict[str, float] = {}
    for item in value:
        if (not isinstance(item, tuple) or len(item) != 2 or
                not isinstance(item[0], str) or not item[0].strip() or
                item[0] in result or type(item[1]) not in (float, int)):
            _fail(f'{where} has invalid or duplicate variable entries')
        try:
            number = float(item[1])
        except (OverflowError, ValueError) as exc:
            raise StructuredContractError(f'{where} contains unrepresentable number') from exc
        if not math.isfinite(number) or number <= 0:
            _fail(f'{where} values must be positive and finite')
        result[item[0]] = number
    return result


@dataclass(frozen=True, slots=True)
class BranchUncertaintyPlan:
    """Bound-data fingerprint, explicit targets, group levels, objective and scale.

    Targets must refer to *observed numeric columns*. For a multi-target
    weighted objective, either standardize each target by its own complete-case
    weighted population SD or supply a declared positive scale for each target.
    """

    binding_digest: str
    targets: tuple[str, ...]
    group_levels: tuple[str, ...] = ()
    scale: str = 'raw'
    declared_scales: tuple[tuple[str, float], ...] = ()
    objective: str = 'per_target'
    target_weights: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        _digest(self.binding_digest)
        _names(self.targets, 'targets', required=True)
        _names(self.group_levels, 'group_levels')
        if not isinstance(self.scale, str) or self.scale not in _SCALES:
            _fail(f'scale must be one of {sorted(_SCALES)}')
        if not isinstance(self.objective, str) or self.objective not in _OBJECTIVES:
            _fail(f'objective must be one of {sorted(_OBJECTIVES)}')
        scales = _positive_pairs(self.declared_scales, 'declared_scales')
        weights = _positive_pairs(self.target_weights, 'target_weights')
        if (self.scale == 'declared') != bool(scales):
            _fail('declared scale requires one positive scale per target; other scales forbid declared_scales')
        if scales and set(scales) != set(self.targets):
            _fail('declared_scales must cover exactly the selected targets')
        if self.objective == 'per_target' and weights:
            _fail('per_target does not accept target_weights')
        if self.objective in ('weighted_variance', 'reconstruction_mse'):
            if set(weights) != set(self.targets):
                _fail('weighted objective requires one positive target weight per selected target')
            if len(self.targets) > 1 and self.scale == 'raw':
                _fail('multi-target weighted variance requires zscore or declared scales')

    def to_dict(self) -> dict[str, Any]:
        return {'schema': PLAN_SCHEMA, 'binding_digest': self.binding_digest,
                'targets': list(self.targets), 'group_levels': list(self.group_levels),
                'scale': self.scale,
                'declared_scales': [[k, v] for k, v in self.declared_scales],
                'objective': self.objective,
                'target_weights': [[k, v] for k, v in self.target_weights]}

    def to_json(self) -> str:
        return _canonical(self.to_dict())

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> 'BranchUncertaintyPlan':
        names = {'schema', 'binding_digest', 'targets', 'group_levels', 'scale',
                 'declared_scales', 'objective', 'target_weights'}
        if not isinstance(raw, Mapping) or set(raw) != names or raw.get('schema') != PLAN_SCHEMA:
            _fail('unsupported SU-06 plan schema or fields')
        if any(not isinstance(raw[name], list) for name in
               ('targets', 'group_levels', 'declared_scales', 'target_weights')):
            _fail('targets, group_levels and pair tables must be arrays')
        for field in ('declared_scales', 'target_weights'):
            if any(not isinstance(item, list) or len(item) != 2 for item in raw[field]):
                _fail(f'{field} must contain two-element arrays')
        return cls(raw['binding_digest'], tuple(raw['targets']), tuple(raw['group_levels']),
                   raw['scale'], tuple(tuple(x) for x in raw['declared_scales']),
                   raw['objective'], tuple(tuple(x) for x in raw['target_weights']))

    @classmethod
    def from_json(cls, value: str) -> 'BranchUncertaintyPlan':
        if not isinstance(value, str):
            _fail('SU-06 plan must be JSON text')
        def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
            out = {}
            for key, value in items:
                if key in out:
                    _fail(f'duplicate SU-06 JSON field: {key}')
                out[key] = value
            return out
        try:
            return cls.from_dict(json.loads(value, object_pairs_hook=pairs,
                              parse_constant=lambda v: _fail(f'invalid JSON constant {v}')))
        except (ValueError, TypeError, RecursionError) as exc:
            raise StructuredContractError(f'invalid SU-06 plan JSON: {exc}') from exc


@dataclass(frozen=True, slots=True)
class BranchUncertaintyResult:
    _json: str

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._json)

    def to_json(self) -> str:
        return self._json


def _finite(value: Any, where: str) -> float:
    if type(value) not in (int, float):
        _fail(f'{where}: numeric target requires finite numeric values, not bool or text')
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise StructuredContractError(f'{where}: numeric value exceeds floating point range') from exc
    if not math.isfinite(number):
        _fail(f'{where}: nonfinite numeric value')
    return number


def _sum(values: Any, where: str) -> float:
    try:
        value = math.fsum(values)
    except (ValueError, OverflowError) as exc:
        raise StructuredContractError(f'{where}: numeric overflow') from exc
    if not math.isfinite(value):
        _fail(f'{where}: numeric overflow')
    return value


def _mean(values: list[float], weights: list[float], where: str) -> float:
    total = _sum(weights, f'{where} total weight')
    return _sum((v * (w / total) for v, w in zip(values, weights)), f'{where} mean')


def _ss(values: list[float], weights: list[float], mean: float, where: str) -> float:
    return _sum((w * ((v - mean) ** 2) for v, w in zip(values, weights)), where)


def _weighted_summary(values: list[float], weights: list[float]) -> dict[str, Any]:
    n = len(values)
    if not n:
        return {'status': 'no_observations', 'rows': 0, 'total_weight': 0.0,
                'effective_n': None, 'mean': None, 'min': None, 'max': None,
                'population_variance': None, 'population_std': None,
                'sample_variance_unweighted': None, 'sum_squares': None}
    total = _sum(weights, 'weight total')
    effective = 1.0 / _sum(((w / total)**2 for w in weights), 'normalized squared weights')
    mean = _mean(values, weights, 'outcome')
    ss = _ss(values, weights, mean, 'variance sum of squares') if n >= 2 else None
    pop_var = ss / total if ss is not None else None
    plain_mean = _sum((v / n for v in values), 'unweighted mean')
    plain_ss = _sum(((v - plain_mean)**2 for v in values), 'unweighted variance') if n >= 2 else None
    return {'status': ('no_observations' if n == 0 else 'insufficient' if n == 1 else
                       'constant' if ss == 0 else 'sufficient'),
            'rows': n, 'total_weight': total, 'effective_n': effective,
            'mean': mean, 'min': min(values), 'max': max(values),
            'population_variance': pop_var,
            'population_std': math.sqrt(pop_var) if pop_var is not None else None,
            'sample_variance_unweighted': plain_ss / (n - 1) if plain_ss is not None else None,
            'sum_squares': ss}


def _original(cell: dict[str, Any]) -> dict[str, Any]:
    lineage = cell.get('lineage')
    if isinstance(lineage, dict) and 'original_multiple_cell' in lineage:
        return lineage['original_multiple_cell']
    return cell


def _profile(rows: list[dict[str, Any]], variable_id: str) -> dict[str, Any]:
    original = Counter()
    resolved = Counter()
    reasons = Counter()
    nested = Counter()
    occurrence_cardinality = Counter()
    occurrence_states = Counter()
    seen: set[str] = set()
    truncated = False
    repeated_rows = 0
    occurrences = 0
    present_occurrences = 0
    for row in rows:
        resolved_cell = row['values'][variable_id]
        cell = _original(resolved_cell)
        original[cell['status']] += 1
        resolved[resolved_cell['status']] += 1
        if cell.get('source_reason') and cell['status'] in ('missing', 'explicit_null', 'unknown'):
            reasons[cell['source_reason']] += 1
        if cell['status'] == 'multiple':
            repeated_rows += 1
            items = cell['occurrences']
        elif cell['status'] in ('present', 'empty'):
            items = [cell]
        else:
            items = []
        occurrences += len(items)
        occurrence_cardinality[len(items)] += 1
        for item in items:
            occurrence_states[item['status']] += 1
            if item['status'] in ('present', 'empty'):
                present_occurrences += 1
                if not truncated:
                    key = _canonical(item['value'])
                    if len(seen) >= _MAX_DISTINCT and key not in seen:
                        truncated = True
                        seen.clear()
                    else:
                        seen.add(key)
            elif item.get('source_reason'):
                nested[item['source_reason']] += 1
    n = len(rows)
    observed = original['present'] + original['empty'] + original['multiple']
    return {'support': {'rows': n, 'observed_rows': observed,
                        'unavailable_rows': n-observed, 'repeated_rows': repeated_rows,
                        'total_occurrences': occurrences,
                        'present_occurrences': present_occurrences},
            'occurrence': {'observed_row_rate': observed/n if n else None,
                           'occurrences_per_row': occurrences/n if n else None,
                           'per_record_cardinality': {str(k): v for k, v in sorted(occurrence_cardinality.items())},
                           'distinct_observed_values': None if truncated else len(seen),
                           'distinct_cardinality_status': 'above_limit' if truncated else 'exact'},
            'original_status_counts': dict(sorted(original.items())),
            'resolved_status_counts': dict(sorted(resolved.items())),
            'null_reasons': dict(sorted(reasons.items())),
            'repeated_null_reasons': dict(sorted(nested.items())),
            'occurrence_status_counts': dict(sorted(occurrence_states.items()))}


def _partition(values: list[float], weights: list[float], keys: list[tuple[str, ...]],
               levels: tuple[str, ...]) -> dict[str, Any]:
    outcome = _weighted_summary(values, weights)
    if outcome['rows'] < 2:
        return {'status': 'insufficient', 'total_sum_squares': None, 'components': None}
    total = outcome['sum_squares']
    previous = {(): outcome['mean']}
    components = []
    for index, level in enumerate(levels, start=1):
        groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for i, key in enumerate(keys):
            groups[key[:index]].append(i)
        means = {key: _mean([values[i] for i in indices], [weights[i] for i in indices],
                            f'group {level}') for key, indices in groups.items()}
        between = _sum((_sum((weights[i] for i in indices), 'group weight') *
                        (means[key] - previous[key[:-1]])**2
                        for key, indices in groups.items()), f'{level} between-group SS')
        components.append({'level': level, 'group_count': len(groups),
                           'sum_squares': between,
                           'proportion_total_ss': between/total if total > 0 else None,
                           'min_rows_per_group': min(len(v) for v in groups.values()),
                           'max_rows_per_group': max(len(v) for v in groups.values())})
        previous = means
    within = _sum((w*(value - previous[key])**2
                   for value, w, key in zip(values, weights, keys)), 'within-group SS')
    components.append({'level': 'within_leaf_groups', 'group_count': len(previous),
                       'sum_squares': within,
                       'proportion_total_ss': within/total if total > 0 else None})
    if not math.isclose(_sum((c['sum_squares'] for c in components), 'partition total'),
                        total, rel_tol=1e-9, abs_tol=1e-9):
        _fail('nested weighted sum-of-squares partition is numerically unstable')
    return {'status': 'constant' if total == 0 else 'sufficient',
            'total_sum_squares': total, 'components': components}


def analyze_branch_uncertainty(binding: ObservationBindingResult,
                               plan: BranchUncertaintyPlan) -> BranchUncertaintyResult:
    """Return detached descriptive evidence, with node/subtree *reference* overlays.

    Group levels refer only to explicitly bound observation grouping columns.
    Source node/branch membership only determines where to display existing
    variable evidence; it does not allocate explained variance across a tree.
    """
    if not isinstance(plan, BranchUncertaintyPlan):
        _fail('analysis requires a BranchUncertaintyPlan')
    if binding_sha256(binding) != plan.binding_digest:
        _fail('binding fingerprint mismatch: stale or different SU-05 observations')
    data = binding.to_dict()
    if data['schema'] != BINDING_SCHEMA:
        _fail('unsupported observation binding result schema')
    bound = {v['variable_id']: v for v in data['variables']}
    if not set(plan.targets).issubset(bound):
        _fail('target variables must be explicitly bound to structural source nodes')
    groups = tuple(data['plan']['group_variables'])
    if plan.group_levels != groups[:len(plan.group_levels)]:
        _fail('group_levels must be an ordered prefix of SU-05 declared group variables')
    rows = data['rows']
    evidence = {name: _profile(rows, name) for name in bound}
    exclusions: Counter[str] = Counter()
    selected = []
    for row in rows:
        scalars = {}
        failure = None
        for name in plan.targets:
            cell = row['values'][name]
            if cell['status'] != 'present':
                failure = f'target_{name}_{cell["status"]}'
                break
            scalars[name] = _finite(cell['value'], f'{name}/{row["observation_id"]}')
        key = ()
        if failure is None and plan.group_levels:
            if row['group_key'] is None:
                failure = 'group_unavailable'
            else:
                key = tuple(_canonical(part) for part in row['group_key'][:len(plan.group_levels)])
                if len(key) != len(plan.group_levels):
                    _fail('group key is shorter than declared levels')
        if failure:
            exclusions[failure] += 1
        else:
            selected.append((scalars, row['weight'], key))
    values = {name: [record[0][name] for record in selected] for name in plan.targets}
    weights = [record[1] for record in selected]
    keys = [record[2] for record in selected]
    # Group support is computed on exactly the cohort used for variance, not
    # SU-05's broader row inventory. Bound the potentially huge identity list.
    group_support = []
    for level_index, level in enumerate(plan.group_levels, start=1):
        buckets: dict[tuple[str, ...], list[float]] = defaultdict(list)
        for key, weight in zip(keys, weights):
            buckets[key[:level_index]].append(weight)
        displayed = []
        for key in sorted(buckets)[:_MAX_GROUP_DISPLAY]:
            group_weights = buckets[key]
            displayed.append({'key': [json.loads(part) for part in key],
                              'rows': len(group_weights),
                              'weight_total': _sum(group_weights, 'group support weights')})
        group_support.append({'level': level, 'group_count': len(buckets),
                              'displayed_group_count': len(displayed),
                              'groups_truncated': len(buckets) > len(displayed),
                              'groups': displayed})
    summaries = {name: _weighted_summary(values[name], weights) for name in plan.targets}
    declared = dict(plan.declared_scales)
    transformed = {}
    normalizations = {}
    for name in plan.targets:
        divisor = (1.0 if plan.scale == 'raw' else declared[name] if plan.scale == 'declared'
                   else summaries[name]['population_std'])
        if divisor is None or divisor == 0:
            transformed[name] = None
            normalizations[name] = {'status': 'unavailable_constant_or_sparse', 'divisor': None}
        else:
            transformed[name] = [v / divisor for v in values[name]]
            if any(not math.isfinite(v) for v in transformed[name]):
                _fail(f'{name}: normalized target exceeds floating point range')
            normalizations[name] = {'status': 'available', 'divisor': divisor}
    target_results = {}
    for name in plan.targets:
        series = transformed[name]
        entry = {'raw': summaries[name], 'normalization': normalizations[name],
                 'scaled': _weighted_summary(series, weights) if series is not None else None,
                 'group_variance': (_partition(series, weights, keys, plan.group_levels)
                                    if series is not None and plan.group_levels else None)}
        target_results[name] = entry
        evidence[name]['numeric'] = entry
    objective = None
    if plan.objective == 'reconstruction_mse':
        objective = {'status': 'unavailable', 'reason': 'paired_reconstruction_required',
                     'mean_squared_error': None, 'components': None}
    if plan.objective == 'weighted_variance':
        objective_weights = dict(plan.target_weights)
        if any(transformed[name] is None or len(selected) < 2 for name in plan.targets):
            objective = {'status': 'unavailable', 'reason': 'sparse_or_unscalable_target',
                         'weighted_population_variance': None, 'components': None}
        else:
            denom = _sum(objective_weights.values(), 'target-weight total')
            mix = {name: objective_weights[name]/denom for name in plan.targets}
            value = _sum((mix[name]*target_results[name]['scaled']['population_variance']
                          for name in plan.targets), 'weighted objective')
            components = None
            if plan.group_levels:
                components = []
                for i, part in enumerate(target_results[plan.targets[0]]['group_variance']['components']):
                    # SS / total observation weight yields comparable weighted population variance.
                    quantity = _sum((mix[name] * target_results[name]['group_variance']['components'][i]['sum_squares']
                                    / target_results[name]['scaled']['total_weight']
                                    for name in plan.targets), 'objective variance component')
                    components.append({'level': part['level'], 'weighted_population_variance': quantity,
                                       'proportion_objective': quantity/value if value > 0 else None})
                if not math.isclose(_sum((p['weighted_population_variance'] for p in components),
                                         'objective component accounting'), value,
                                    rel_tol=1e-9, abs_tol=1e-9):
                    _fail('weighted objective components fail variance accounting')
            objective = {'status': 'constant' if value == 0 else 'sufficient',
                         'weighted_population_variance': value, 'components': components,
                         'normalized_target_weights': mix}
    # Direct source mappings are safe only when a variable maps to one node.
    direct: dict[str, set[str]] = defaultdict(set)
    ambiguous: dict[str, set[str]] = defaultdict(set)
    for variable in bound.values():
        for node_id in variable['node_ids']:
            (ambiguous if variable['merged_node_identity_ambiguous'] else direct)[node_id].add(variable['variable_id'])
    branches = data['branches']
    branch_by_id = {entry['node_id']: entry for entry in branches} if branches is not None else {}
    ambiguous_ids = {v['variable_id'] for v in bound.values() if v['merged_node_identity_ambiguous']}
    nodes = []
    source_nodes = ([entry['node_id'] for entry in branches]
                    if branches is not None else sorted(set(direct) | set(ambiguous)))
    for node_id in source_nodes:
        branch = branch_by_id.get(node_id)
        scoped = set(branch['subtree_variable_ids']) if branch is not None else set(direct[node_id])
        ambiguous_scope = sorted(scoped & ambiguous_ids)
        nodes.append({'node_id': node_id, 'scope': 'subtree' if branch is not None else 'direct_only',
                      'direct_variable_ids': sorted(direct[node_id]),
                      'scope_variable_ids': sorted(scoped - set(ambiguous_scope)),
                      'ambiguous_merged_variable_ids': sorted(set(ambiguous_scope) | ambiguous[node_id]),
                      'selected_target_ids': sorted((scoped - set(ambiguous_scope)) & set(plan.targets)),
                      'variance_is_attributed_to_graph_edge': False})
    payload = {'schema': RESULT_SCHEMA, 'source_object_id': data['source_object_id'],
               'source_digest': data['source_digest'], 'binding_digest': plan.binding_digest,
               'plan': plan.to_dict(), 'observation_unit': data['observation_unit'],
               'support': {'aligned_rows': len(rows), 'complete_case_rows': len(selected),
                           'excluded_rows': len(rows) - len(selected),
                           'exclusion_reasons': dict(sorted(exclusions.items())),
                           'complete_case_weight_total': _sum(weights, 'complete-case weights'),
                           'independence_verified': False},
               'group_support': group_support,
               'variables': evidence, 'targets': target_results, 'objective': objective,
               'nodes': nodes,
               'limitations': [
                   'descriptive_record_variation_not_sampling_independence_or_confidence_interval',
                   'variance_group_levels_are_observed_labels_not_tree_edges',
                   'node_branch_scopes_reference_variables_and_do_not_partition_variance',
                   'source_nodes_and_repeated_occurrences_are_not_independent_records',
                   'missing_outcomes_and_group_labels_are_complete_case_exclusions',
                   'weighted_variance_uses_observation_weighted_population_denominator',
                   'no_source_graph_causal_or_statistical_edge_inference',
                   'no_multivariable_variance_summation_without_explicit_scale_and_weights',
                   'reconstruction_mse_requires_future_explicit_paired_candidate_values',
               ]}
    return BranchUncertaintyResult(_canonical(payload))
