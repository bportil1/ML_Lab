"""SU-09: matched, replayable empirical comparisons of SU-08 candidates.

The source topology supplies no training rows or stochastic dependencies.  All
scoring uses the *same* SU-05 resolved records and a fixed train/holdout split.
Binning and lookup estimators are fitted only on training rows.  This is an
exploratory comparison, not a selected model's independent final evaluation.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

from .bindings import ObservationBindingPlan, bind_observations, source_sha256
from .branch_uncertainty import binding_sha256
from .candidates import StructuralCandidatePlan, apply_structural_candidate
from .contracts import StructuredContractError
from .object import StructuredObject

PLAN_SCHEMA = 'ml-lab.structural-retention-study@1'
REPORT_SCHEMA = 'ml-lab.structural-retention-comparison@1'
KINDS = frozenset({'target_free', 'categorical_target', 'numeric_target'})
METRICS = frozenset({'retained_entropy_fraction', 'retained_variance_fraction',
                     'holdout_accuracy', 'holdout_r2', 'reconstruction_r2'})
_MAX_CANDIDATES = 24
_MAX_ROWS = 50_000
_MAX_STATES = 4096


def _fail(reason: str) -> None:
    raise StructuredContractError(reason)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def _hash(value: Any) -> str:
    return 'sha256:' + hashlib.sha256(_json(value).encode('utf-8')).hexdigest()


def _names(values: object, field: str) -> None:
    if (not isinstance(values, tuple) or len(set(v for v in values if isinstance(v, str))) != len(values)
            or any(not isinstance(v, str) or not v.strip() for v in values)):
        _fail(f'{field} must be a tuple of unique nonempty variable IDs')


def _fingerprint(value: str, field: str) -> None:
    if (not isinstance(value, str) or len(value) != 71 or not value.startswith('sha256:')
            or any(ch not in '0123456789abcdef' for ch in value[7:])):
        _fail(f'{field} must be lowercase SHA-256')


def _strict_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    answer = {}
    for k, v in items:
        if k in answer:
            _fail(f'duplicate SU-09 JSON key: {k}')
        answer[k] = v
    return answer


@dataclass(frozen=True, slots=True)
class StructuralRetentionStudy:
    """Explicit source/binding-pinned candidate set and leakage-safe evaluation.

    ``predictors`` is the original feature vocabulary; a SU-08 projection is
    the only way to remove a variable from a candidate's empirical score.
    Optional variance/reconstruction variables must be explicitly numeric.
    """

    source_digest: str
    binding_digest: str
    candidates: tuple[StructuralCandidatePlan, ...]
    predictors: tuple[str, ...]
    kind: str = 'target_free'
    target: str | None = None
    variance_variables: tuple[str, ...] = ()
    reconstruction_variables: tuple[str, ...] = ()
    group_variable: str | None = None
    acknowledge_independent_rows: bool = False
    seed: int = 0
    holdout_fraction: float = 0.25
    numeric_bins: int = 8
    min_train_rows: int = 4
    min_test_rows: int = 2
    max_joint_states: int = _MAX_STATES
    threshold_metric: str | None = None
    minimum_threshold: float | None = None

    def __post_init__(self) -> None:
        _fingerprint(self.source_digest, 'source_digest')
        _fingerprint(self.binding_digest, 'binding_digest')
        if (not isinstance(self.candidates, tuple) or not 2 <= len(self.candidates) <= _MAX_CANDIDATES
                or any(not isinstance(p, StructuralCandidatePlan) for p in self.candidates)):
            _fail('study requires 2–24 validated SU-08 candidate plans')
        if len(set(p.to_json() for p in self.candidates)) != len(self.candidates):
            _fail('duplicate candidate plans')
        if sum(p.operation == 'preserve' for p in self.candidates) != 1:
            _fail('study requires exactly one preserve reference')
        if any(p.source_digest != self.source_digest or
               p.binding_digest != self.binding_digest for p in self.candidates):
            _fail('candidate source/binding digest differs from study')
        for name, value in [('predictors', self.predictors),
                            ('variance_variables', self.variance_variables),
                            ('reconstruction_variables', self.reconstruction_variables)]:
            _names(value, name)
        if not self.predictors:
            _fail('declare at least one predictor; empty candidate representations are permitted')
        if not set(self.variance_variables) <= set(self.predictors):
            _fail('variance_variables must be selected predictors')
        if not set(self.reconstruction_variables) <= set(self.predictors):
            _fail('reconstruction_variables must be selected predictors')
        if not isinstance(self.kind, str) or self.kind not in KINDS:
            _fail('unknown task kind')
        if self.kind == 'target_free':
            if self.target is not None:
                _fail('target_free requires target=None')
        else:
            if not isinstance(self.target, str) or not self.target.strip():
                _fail('predictive study requires a target')
            if self.reconstruction_variables:
                _fail('reconstruction_variables are supported only in target_free studies')
        if self.target in self.predictors or self.group_variable in self.predictors:
            _fail('target/group variable cannot be a predictor')
        if self.group_variable is not None:
            if not isinstance(self.group_variable, str) or not self.group_variable.strip():
                _fail('group_variable must be a nonempty ID')
            if self.group_variable == self.target or self.acknowledge_independent_rows:
                _fail('group split conflicts with target or independent-row acknowledgment')
        elif self.acknowledge_independent_rows is not True:
            _fail('group_variable or explicit independent-row acknowledgment required')
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            _fail('seed must be a 32-bit nonnegative integer')
        if (type(self.holdout_fraction) not in (float, int) or
                not math.isfinite(self.holdout_fraction) or
                not 0 < self.holdout_fraction < 1):
            _fail('holdout_fraction must be strictly between 0 and 1')
        for field, minimum, maximum in (('numeric_bins', 2, 64),
                                        ('min_train_rows', 1, _MAX_ROWS),
                                        ('min_test_rows', 1, _MAX_ROWS),
                                        ('max_joint_states', 1, _MAX_STATES)):
            value = getattr(self, field)
            if type(value) is not int or not minimum <= value <= maximum:
                _fail(f'{field} must be an integer within [{minimum}, {maximum}]')
        if (self.threshold_metric is None) != (self.minimum_threshold is None):
            _fail('threshold_metric and minimum_threshold must be provided together')
        if self.threshold_metric is not None:
            allowed = ({'retained_entropy_fraction', 'retained_variance_fraction',
                        'reconstruction_r2'} if self.kind == 'target_free' else
                       {'holdout_accuracy'} if self.kind == 'categorical_target' else
                       {'holdout_r2'})
            if not isinstance(self.threshold_metric, str) or self.threshold_metric not in allowed:
                _fail('threshold_metric is not defined for the requested task')
            if self.threshold_metric == 'retained_variance_fraction' and not self.variance_variables:
                _fail('retained variance threshold requires variance_variables')
            if self.threshold_metric == 'reconstruction_r2' and not self.reconstruction_variables:
                _fail('reconstruction threshold requires reconstruction_variables')
            if (type(self.minimum_threshold) not in (int, float) or
                    not math.isfinite(self.minimum_threshold)):
                _fail('minimum_threshold must be a finite number')
            if self.threshold_metric.endswith('fraction') or self.threshold_metric == 'holdout_accuracy':
                if not 0 <= self.minimum_threshold <= 1:
                    _fail('fraction/accuracy threshold must be within [0,1]')

    def to_dict(self) -> dict[str, Any]:
        return {'schema': PLAN_SCHEMA, 'source_digest': self.source_digest,
                'binding_digest': self.binding_digest,
                'candidates': [x.to_dict() for x in self.candidates],
                'predictors': list(self.predictors), 'kind': self.kind,
                'target': self.target, 'variance_variables': list(self.variance_variables),
                'reconstruction_variables': list(self.reconstruction_variables),
                'group_variable': self.group_variable,
                'acknowledge_independent_rows': self.acknowledge_independent_rows,
                'seed': self.seed, 'holdout_fraction': self.holdout_fraction,
                'numeric_bins': self.numeric_bins,
                'min_train_rows': self.min_train_rows, 'min_test_rows': self.min_test_rows,
                'max_joint_states': self.max_joint_states,
                'threshold_metric': self.threshold_metric,
                'minimum_threshold': self.minimum_threshold}

    def to_json(self) -> str:
        return _json(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> 'StructuralRetentionStudy':
        required = set(cls.__dataclass_fields__) | {'schema'}
        if not isinstance(data, Mapping) or set(data) != required or data.get('schema') != PLAN_SCHEMA:
            _fail('invalid SU-09 study schema/fields')
        if (not isinstance(data['candidates'], list) or
                any(not isinstance(data[k], list) for k in
                    ('predictors', 'variance_variables', 'reconstruction_variables'))):
            _fail('study candidates and variable selections must be JSON arrays')
        return cls(**{k: (tuple(StructuralCandidatePlan.from_dict(p) for p in data[k]) if k == 'candidates'
                          else tuple(data[k]) if k in {'predictors', 'variance_variables',
                                                      'reconstruction_variables'} else data[k])
                      for k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, raw: str) -> 'StructuralRetentionStudy':
        if not isinstance(raw, str):
            _fail('SU-09 study must be JSON text')
        try:
            parsed = json.loads(raw, object_pairs_hook=_strict_pairs,
                                parse_constant=lambda x: _fail(f'nonfinite JSON: {x}'))
        except (ValueError, TypeError, RecursionError, UnicodeError) as exc:
            raise StructuredContractError(f'invalid SU-09 study JSON: {exc}') from exc
        return cls.from_dict(parsed)


@dataclass(frozen=True, slots=True)
class StructuralRetentionResult:
    _json: str

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._json)

    def to_json(self) -> str:
        return self._json


def _metric(value: float | None, reason: str = 'ok', **details: Any) -> dict[str, Any]:
    if value is not None and not math.isfinite(value):
        _fail('nonfinite SU-09 metric')
    return {'status': reason, 'value': value, **details}


def _numeric(cell: Mapping[str, Any], name: str) -> float:
    value = cell['value']
    if type(value) not in (int, float):
        _fail(f'{name}: expected finite numeric measurement (not boolean)')
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise StructuredContractError(f'{name}: numeric value overflow') from exc
    if not math.isfinite(number):
        _fail(f'{name}: nonfinite numeric measurement')
    return number


def _category(cell: Mapping[str, Any], name: str) -> tuple[str, Any]:
    if cell['status'] == 'empty':
        return ('empty', '')
    value = cell['value']
    if type(value) not in (str, int, bool):
        _fail(f'{name}: target must be finite categorical scalar (str/int/bool or empty)')
    return (type(value).__name__, value)


def _key(value: Any) -> str:
    return _json(value)


def _variance(numbers: list[float]) -> float:
    if len(numbers) < 2:
        return 0.0
    mean = math.fsum(numbers) / len(numbers)
    try:
        return math.fsum(((x - mean)**2 / len(numbers) for x in numbers))
    except (ValueError, OverflowError) as exc:
        raise StructuredContractError('numeric variance overflow') from exc


def _bin_edges(values: list[float], count: int) -> tuple[float, ...]:
    ordered = sorted(values)
    if not ordered:
        _fail('cannot fit bins with no training observations')
    # Train-only empirical quantiles, duplicates removed, no float interpolation.
    return tuple(sorted({ordered[min(len(ordered)-1, (i * len(ordered)) // count)]
                         for i in range(1, count)}))


def _encoded_features(rows: list[dict], names: tuple[str, ...],
                      edges: Mapping[str, tuple[float, ...]]) -> list[dict[str, str]]:
    result = []
    for row in rows:
        values = {}
        for name in names:
            cell = row['cells'][name]
            if cell['status'] == 'empty':
                values[name] = _key(['empty'])
            elif name in edges:
                values[name] = _key(['numeric_bin', bisect_right(edges[name], _numeric(cell, name))])
            else:
                values[name] = _key(['category', *_category(cell, name)])
        result.append(values)
    return result


def _joint(row: Mapping[str, str], predictors: tuple[str, ...]) -> str:
    return _key([row[name] for name in predictors])


def _entropy(keys: list[str]) -> float:
    if not keys:
        return 0.0
    total = len(keys)
    return -math.fsum((count / total * math.log2(count / total)
                       for count in Counter(keys).values()))


def _information(keys: list[str], target: list[str]) -> float:
    return max(0.0, _entropy(keys) + _entropy(target) -
               _entropy([_key([x, y]) for x, y in zip(keys, target)]))


def _predict_categorical(train_keys: list[str], test_keys: list[str],
                         training: list[str], observed: list[str],
                         max_states: int) -> dict[str, Any]:
    counts = defaultdict(Counter)
    marginal: Counter[str] = Counter(training)
    for key, value in zip(train_keys, training):
        counts[key][value] += 1
    if len(counts) > max_states:
        return {'status': 'joint_state_limit', 'holdout_accuracy': _metric(None, 'joint_state_limit'),
                'holdout_log_loss_nats': _metric(None, 'joint_state_limit')}
    labels = sorted(marginal)
    if set(observed) - set(labels):
        return {'status': 'unseen_holdout_target_state',
                'holdout_accuracy': _metric(None, 'unseen_holdout_target_state'),
                'holdout_log_loss_nats': _metric(None, 'unseen_holdout_target_state')}
    successes, penalties, fallback = 0, [], 0
    for key, truth in zip(test_keys, observed):
        local = counts.get(key)
        if local is None:
            local = marginal
            fallback += 1
        total = sum(local.values()) + len(labels)  # Laplace add-one
        probabilities = {value: (local[value] + 1) / total for value in labels}
        choice = min(labels, key=lambda value: (-probabilities[value], value))
        successes += choice == truth
        penalties.append(-math.log(probabilities[truth]))
    return {'status': 'ok', 'holdout_accuracy': _metric(successes / len(observed)),
            'holdout_log_loss_nats': _metric(math.fsum(penalties) / len(penalties)),
            'unseen_configuration_fallback_rows': fallback,
            'train_feature_configurations': len(counts),
            'estimator': 'train_only_add_one_categorical_lookup_marginal_fallback'}


def _predict_numeric(train_keys: list[str], test_keys: list[str],
                     training: list[float], observed: list[float],
                     max_states: int) -> dict[str, Any]:
    sums: dict[str, list[float]] = defaultdict(list)
    for key, value in zip(train_keys, training):
        sums[key].append(value)
    if len(sums) > max_states:
        return {'status': 'joint_state_limit', 'holdout_mse': _metric(None, 'joint_state_limit'),
                'holdout_r2': _metric(None, 'joint_state_limit')}
    global_mean = math.fsum(training) / len(training)
    predictions = [math.fsum(sums[key]) / len(sums[key]) if key in sums else global_mean
                   for key in test_keys]
    mse = math.fsum(((p - y)**2 / len(observed) for p, y in zip(predictions, observed)))
    baseline = math.fsum(((y - global_mean)**2 / len(observed) for y in observed))
    status = 'ok' if baseline > 0 else 'constant_holdout_target'
    return {'status': 'ok', 'holdout_mse': _metric(mse),
            'holdout_r2': _metric(1 - mse / baseline) if baseline > 0 else _metric(None, status),
            'train_mean_baseline_mse': baseline,
            'unseen_configuration_fallback_rows': sum(key not in sums for key in test_keys),
            'train_feature_configurations': len(sums),
            'estimator': 'train_only_conditional_mean_lookup_marginal_fallback'}


def _partition(records: list[dict], study: StructuralRetentionStudy) -> tuple[list[dict], list[dict], dict]:
    clusters: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        identifier = (row['group'] if study.group_variable is not None else _key(['observation', row['id']]))
        clusters[identifier].append(row)
    if len(clusters) < 2:
        _fail('split requires at least two independent groups or explicitly acknowledged rows')
    ordered = sorted(clusters, key=lambda key: (_hash([study.seed, key]), key))
    held_count = max(1, min(len(ordered) - 1, round(len(ordered)*study.holdout_fraction)))
    held = set(ordered[:held_count])
    test = [row for row in records if
            (row['group'] if study.group_variable is not None else _key(['observation', row['id']])) in held]
    train = [row for row in records if
             (row['group'] if study.group_variable is not None else _key(['observation', row['id']])) not in held]
    if len(train) < study.min_train_rows or len(test) < study.min_test_rows:
        _fail('matched train/test split does not meet declared sample-support thresholds')
    train_ids, test_ids = sorted(x['id'] for x in train), sorted(x['id'] for x in test)
    return train, test, {
        'method': 'group_disjoint_hash_holdout' if study.group_variable else 'row_hash_holdout',
        'seed': study.seed, 'group_variable': study.group_variable,
        'requested_holdout_fraction': study.holdout_fraction,
        'train_rows': len(train), 'test_rows': len(test),
        'train_groups': len(ordered) - held_count, 'test_groups': held_count,
        'train_observation_ids': train_ids, 'test_observation_ids': test_ids,
        'train_id_digest': _hash(train_ids), 'test_id_digest': _hash(test_ids),
        'independence_verified': False,
        'group_leakage_prevented': study.group_variable is not None}


def _ratio(value: float | None, reference: float | None) -> dict[str, Any]:
    if value is None or reference is None:
        return _metric(None, 'not_available')
    if reference <= 0:
        return _metric(None, 'zero_reference_denominator')
    return _metric(value / reference)


def _complexity(result: Any, original_rows: int) -> dict[str, Any]:
    manifest = result.manifest
    loss, feature = manifest['structural_loss'], manifest['feature_impact']
    return {'retained_nodes': loss['retained_node_count'],
            'retained_edges': loss['retained_edge_count'],
            'retained_observed_columns': len(feature['output_variable_ids']),
            'removed_nodes': loss['removed_node_count'],
            'removed_edges': loss['removed_edge_count'],
            'removed_observed_columns': feature['dropped_variable_count'],
            'observation_cell_storage_proxy': original_rows * len(feature['output_variable_ids']),
            'proxy_units': 'counts_only_not_bytes_or_runtime'}


def _pareto(report: list[dict], metric: str) -> list[str]:
    eligible = [r for r in report if r['metrics'].get(metric, {}).get('status') == 'ok']
    axes = ('retained_nodes', 'retained_edges', 'retained_observed_columns')
    frontier = []
    for item in eligible:
        val = item['metrics'][metric]['value']
        dominated = False
        for other in eligible:
            if item is other:
                continue
            score = other['metrics'][metric]['value']
            costs = all(other['complexity'][key] <= item['complexity'][key] for key in axes)
            strict = score > val + 1e-12 or any(other['complexity'][key] < item['complexity'][key] for key in axes)
            if score >= val - 1e-12 and costs and strict:
                dominated = True
                break
        item['pareto_eligible'] = True
        item['pareto_nondominated'] = not dominated
        if not dominated:
            frontier.append(item['candidate_id'])
    for item in report:
        if 'pareto_eligible' not in item:
            item['pareto_eligible'] = False
            item['pareto_nondominated'] = None
    return frontier


def compare_structural_retention(source: StructuredObject, binding: ObservationBindingPlan,
                                 study: StructuralRetentionStudy) -> StructuralRetentionResult:
    """Evaluate candidate projections without changing the original or any UI.

    Complete-case alignment is chosen ONCE using all declared original variables,
    before candidate scoring. The split is identical for all candidates. A
    presentation-only graph change must therefore have identical scores.
    """
    if not isinstance(source, StructuredObject) or not isinstance(binding, ObservationBindingPlan) or not isinstance(study, StructuralRetentionStudy):
        _fail('validated source, SU-05 binding and SU-09 study are required')
    if len(source.observations) > _MAX_ROWS:
        _fail('input exceeds explicit SU-09 row budget')
    if source_sha256(source) != study.source_digest:
        _fail('source fingerprint mismatch')
    bound_result = bind_observations(source, binding)
    if binding_sha256(bound_result) != study.binding_digest:
        _fail('binding fingerprint mismatch')
    bound = bound_result.to_dict()
    if bound['plan']['weight_variable'] is not None:
        _fail('weighted binding unsupported for SU-09: no silent unweighted scoring')
    selected = set(study.predictors) | set(study.variance_variables) | set(study.reconstruction_variables)
    if study.target is not None:
        selected.add(study.target)
    variable_ids = {v['variable_id'] for v in bound['variables']}
    if selected - variable_ids:
        _fail(f'study variables require exact SU-05 bindings: {sorted(selected-variable_ids)}')
    if study.group_variable is not None and study.group_variable not in bound['plan']['group_variables']:
        _fail('group_variable must be declared in SU-05 binding group_variables')
    source_rows = {row['observation_id']: row['values'] for row in source.to_dict()['observations']}
    reasons: Counter[str] = Counter()
    records = []
    for row in bound['rows']:
        cells = row['values']
        group = None
        if study.group_variable is not None:
            group_cell = source_rows[row['observation_id']][study.group_variable]
            if group_cell['status'] not in {'present', 'empty'}:
                _fail('missing grouping key cannot be used in a leakage-safe group split')
            if type(group_cell['value']) not in (str, int, float, bool):
                _fail('group key must be a scalar value')
            if type(group_cell['value']) is float and not math.isfinite(group_cell['value']):
                _fail('group key must be finite')
            group = _key([group_cell['status'], type(group_cell['value']).__name__, group_cell['value']])
        bad = [(name, cells[name]['status']) for name in sorted(selected)
               if cells[name]['status'] not in {'present', 'empty'}]
        if bad:
            for name, status in bad:
                reasons[f'{name}:{status}'] += 1
            continue
        records.append({'id': row['observation_id'], 'cells': cells, 'group': group})
    if not records:
        _fail('no complete-case aligned observation records available')
    train, test, split = _partition(records, study)
    # Fit numeric predictor quantiles from training rows only, shared by all
    # candidates and independent of target values.
    edges = {}
    for name in study.predictors:
        present = [row['cells'][name] for row in train if row['cells'][name]['status'] == 'present']
        value_types = {type(c['value']) for c in present}
        if any(t in (float, int) for t in value_types):
            if any(t not in (float, int) for t in value_types):
                _fail(f'{name}: mixed numeric/categorical predictor types')
            edges[name] = _bin_edges([_numeric(c, name) for c in present], study.numeric_bins)
        elif any(t not in (str, bool) for t in value_types):
            _fail(f'{name}: unsupported categorical predictor type')
    encoded_train = _encoded_features(train, study.predictors, edges)
    encoded_test = _encoded_features(test, study.predictors, edges)
    result_candidates = [apply_structural_candidate(source, plan, binding=binding)
                         for plan in study.candidates]
    reference = next((r for r in result_candidates if r.plan.operation == 'preserve'))
    assert reference is not None
    train_targets, test_targets = [], []
    if study.target is not None:
        for rows, dest in ((train, train_targets), (test, test_targets)):
            for row in rows:
                dest.append(_key(_category(row['cells'][study.target], study.target))
                            if study.kind == 'categorical_target'
                            else _numeric(row['cells'][study.target], study.target))
    variance_weights = {}
    for name in study.variance_variables:
        training = [_numeric(row['cells'][name], name) for row in train]
        train_var = _variance(training)
        variance_weights[name] = (1 / train_var if train_var > 0 else None)
    holdout_var = {}
    for name in study.variance_variables:
        observed_var = _variance([_numeric(row['cells'][name], name) for row in test])
        holdout_var[name] = (observed_var * variance_weights[name]
                             if variance_weights[name] is not None else
                             (0.0 if observed_var == 0 else None))
    reference_var = (math.fsum(holdout_var.values()) if study.variance_variables
                     and all(v is not None for v in holdout_var.values()) else None)
    full_keys = [_joint(x, study.predictors) for x in encoded_test]
    reference_entropy = _entropy(full_keys) if len(set(full_keys)) <= study.max_joint_states else None
    reference_mi = None
    if (study.target is not None and study.kind == 'categorical_target'
            and reference_entropy is not None):
        reference_mi = _information(full_keys, test_targets)
    report = []
    for candidate in result_candidates:
        manifest = candidate.manifest
        surviving = tuple(name for name in study.predictors
                          if name in manifest['feature_impact']['output_variable_ids'])
        train_keys = [_joint(row, surviving) for row in encoded_train]
        test_keys = [_joint(row, surviving) for row in encoded_test]
        distinct = len(set(test_keys))
        metrics = {}
        if reference_entropy is None or distinct > study.max_joint_states:
            metrics['retained_entropy_fraction'] = _metric(None, 'joint_state_limit')
            metrics['omitted_conditional_entropy_bits'] = _metric(None, 'joint_state_limit')
        else:
            candidate_entropy = _entropy(test_keys)
            metrics['retained_entropy_fraction'] = _ratio(candidate_entropy, reference_entropy)
            # Because the retained columns are a subset of the full columns,
            # H(full | retained) = H(full) - H(retained) for matched rows.
            metrics['omitted_conditional_entropy_bits'] = _metric(max(0.0, reference_entropy - candidate_entropy))
        if not study.variance_variables:
            metrics['retained_variance_fraction'] = _metric(None, 'not_requested')
            metrics['retained_numeric_scaled_variance'] = _metric(None, 'not_requested')
            metrics['omitted_numeric_scaled_variance'] = _metric(None, 'not_requested')
        elif reference_var is None:
            metrics['retained_variance_fraction'] = _metric(None, 'train_variance_scale_unavailable')
            metrics['retained_numeric_scaled_variance'] = _metric(None, 'train_variance_scale_unavailable')
            metrics['omitted_numeric_scaled_variance'] = _metric(None, 'train_variance_scale_unavailable')
        else:
            retained = [holdout_var[n] for n in study.variance_variables if n in surviving]
            retained_amount = math.fsum(retained)
            metrics['retained_variance_fraction'] = _ratio(retained_amount, reference_var)
            metrics['retained_numeric_scaled_variance'] = _metric(retained_amount)
            metrics['omitted_numeric_scaled_variance'] = _metric(max(0.0, reference_var - retained_amount))
        if study.kind == 'categorical_target':
            relevant = (_information(test_keys, test_targets)
                        if distinct <= study.max_joint_states else None)
            metrics['heldout_plugin_mi_bits'] = (_metric(relevant) if relevant is not None
                                                 else _metric(None, 'joint_state_limit'))
            metrics['retained_target_information_fraction'] = _ratio(relevant, reference_mi)
            metrics['omitted_conditional_target_information_bits'] = (
                _metric(max(0.0, reference_mi - relevant))
                if relevant is not None and reference_mi is not None else
                _metric(None, 'joint_state_limit'))
            predicted = _predict_categorical(train_keys, test_keys, train_targets,
                                             test_targets, study.max_joint_states)
            metrics.update({k: v for k, v in predicted.items() if k in
                            ('holdout_accuracy', 'holdout_log_loss_nats')})
        elif study.kind == 'numeric_target':
            predicted = _predict_numeric(train_keys, test_keys, train_targets,
                                         test_targets, study.max_joint_states)
            metrics.update({k: v for k, v in predicted.items() if k in ('holdout_mse', 'holdout_r2')})
            metrics['heldout_plugin_mi_bits'] = _metric(None, 'numeric_target_information_not_defined')
            metrics['retained_target_information_fraction'] = _metric(None, 'numeric_target_information_not_defined')
            metrics['omitted_conditional_target_information_bits'] = _metric(None, 'numeric_target_information_not_defined')
        else:
            predicted = {'estimator': 'none_target_free'}
            metrics['heldout_plugin_mi_bits'] = _metric(None, 'no_target')
            metrics['retained_target_information_fraction'] = _metric(None, 'no_target')
            metrics['omitted_conditional_target_information_bits'] = _metric(None, 'no_target')
        if study.reconstruction_variables:
            mse_total, baseline_total, fallback_count, unavailable = 0.0, 0.0, 0, False
            for name in study.reconstruction_variables:
                tr = [_numeric(row['cells'][name], name) for row in train]
                te = [_numeric(row['cells'][name], name) for row in test]
                mean = math.fsum(tr) / len(tr)
                baseline_total += math.fsum((v-mean)**2 / len(te) for v in te)
                if name in surviving:
                    continue  # exact observed column is the identity reconstruction
                fit = _predict_numeric(train_keys, test_keys, tr, te, study.max_joint_states)
                if fit['holdout_mse']['status'] != 'ok':
                    unavailable = True
                    break
                mse_total += fit['holdout_mse']['value']
                fallback_count += fit['unseen_configuration_fallback_rows']
            divisor = len(study.reconstruction_variables)
            metrics['reconstruction_mse'] = (_metric(mse_total / divisor) if not unavailable
                                             else _metric(None, 'joint_state_limit'))
            metrics['reconstruction_r2'] = (_metric(1 - mse_total / baseline_total)
                                            if not unavailable and baseline_total > 0 else
                                            _metric(None, 'zero_reference_denominator' if not unavailable
                                                    else 'joint_state_limit'))
            predicted['reconstruction_unseen_fallback_count'] = fallback_count
        else:
            metrics['reconstruction_mse'] = _metric(None, 'not_requested')
            metrics['reconstruction_r2'] = _metric(None, 'not_requested')
        report.append({
            'candidate_id': manifest['candidate_id'], 'operation': candidate.plan.operation,
            'plan': candidate.plan.to_dict(), 'output_digest': manifest['output_digest'],
            'structural_loss': manifest['structural_loss'],
            'feature_impact': manifest['feature_impact'],
            'evaluated_predictors': list(surviving),
            'candidate_source_mapping_affected': manifest['feature_impact']['source_nodes_removed_for_bound_variables'],
            'complexity': _complexity(candidate, len(records)),
            'metrics': metrics, 'support': {'matched_complete_case_rows': len(records),
                'train_rows': len(train), 'test_rows': len(test),
                'holdout_distinct_feature_states': distinct,
                'unseen_feature_configurations_train_fallback': predicted.get('unseen_configuration_fallback_rows'),
                'train_feature_configurations': predicted.get('train_feature_configurations'),
                'reconstruction_unseen_fallback_count': predicted.get('reconstruction_unseen_fallback_count'),
                'estimator': predicted.get('estimator'),
                'source_node_count_is_not_sample_count': True},
        })
    # A fixed preserve candidate is the baseline. Deltas keep absolute task
    # scores separate from relative empirical changes; an unavailable baseline
    # makes the delta unavailable instead of fabricating a zero.
    baseline_metrics = next(row['metrics'] for row in report if row['operation'] == 'preserve')
    for row in report:
        for metric in ('holdout_accuracy', 'holdout_log_loss_nats', 'holdout_mse', 'holdout_r2'):
            current = row['metrics'].get(metric)
            baseline = baseline_metrics.get(metric)
            if current is None or baseline is None:
                continue
            if current['status'] == 'ok' and baseline['status'] == 'ok':
                row['metrics'][metric + '_delta_vs_preserve'] = _metric(current['value'] - baseline['value'])
            else:
                row['metrics'][metric + '_delta_vs_preserve'] = _metric(None, 'reference_or_candidate_unavailable')
    primary = study.threshold_metric or ({'target_free': 'retained_entropy_fraction',
                                          'categorical_target': 'holdout_accuracy',
                                          'numeric_target': 'holdout_r2'}[study.kind])
    frontier = _pareto(report, primary)
    qualified = None
    if study.threshold_metric is not None:
        qualified = [item['candidate_id'] for item in report if
                     item['metrics'][study.threshold_metric]['status'] == 'ok' and
                     item['metrics'][study.threshold_metric]['value'] >= study.minimum_threshold]
        for item in report:
            item['meets_declared_threshold'] = item['candidate_id'] in qualified
    else:
        for item in report:
            item['meets_declared_threshold'] = None
    return StructuralRetentionResult(_json({
        'schema': REPORT_SCHEMA, 'source_digest': study.source_digest,
        'binding_digest': study.binding_digest, 'study': study.to_dict(),
        'observation_unit': source.observation_unit,
        'sampling_independence_verified': False,
        'complete_case_support': {
            'source_rows': len(bound['rows']), 'included_rows': len(records),
            'excluded_rows': len(bound['rows']) - len(records),
            'excluded_statuses': dict(sorted(reasons.items())),
            'selection': 'shared_complete_case_prior_to_candidate_projection',
            'aggregation': 'SU_05_declared_occurrence_policies_only',
        }, 'split': split,
        'estimation': {'numeric_predictor_bins': study.numeric_bins,
            'train_only_bin_cutpoints': {k: list(v) for k, v in sorted(edges.items())},
            'discretization': 'train_only_empirical_quantiles',
            'categorical_predictor': 'exact_scalar',
            'categorical_target': 'explicit_categorical',
            'variance_measure': 'sum_of_holdout_population_variances_divided_by_train_population_variance_per_selected_numeric_variable',
            'variance_variables': list(study.variance_variables),
            'information': 'uncorrected_holdout_plugin_entropy_and_MI_bits_not_causal_and_not_crossvalidated',
            'omitted_information': 'H(full_features|retained_features), and I(omitted_features;target|retained_features) when target categorical; plugin identities on same holdout rows',
            'reconstruction': 'exact_retained_numeric_values_else_train_only_conditional_mean_lookup',
            'feature_information_vs_structure': 'distinct_quantities',
        }, 'pareto': {'metric': primary, 'cost_axes': ['retained_nodes', 'retained_edges',
             'retained_observed_columns'], 'nondominated_candidate_ids': frontier,
             'threshold_metric': study.threshold_metric,
             'minimum_threshold': study.minimum_threshold,
             'threshold_qualified_candidate_ids': qualified},
        'candidates': report,
        'warnings': [
            'split_does_not_establish_independent_samples',
            'unseen_joint_feature_configurations_use_train_only_marginal_fallback',
            'plugin_entropy_MI_have_finite_sample_bias_and_no_confidence_intervals',
            'no_permutation_significance_or_model_stability_uncertainty_computed',
            'variance_retention_is_a_declared_scaled_sum_not_variance_explained_by_a_model',
            'candidate_ranking_and_threshold_selection_on_this_holdout_are_exploratory_not_final_test',
            'reference_entropy_zero_makes_entropy_retention_undefined',
            'topology_only_changes_do_not_alter_observed_feature_statistics',
            'no_metric_is_inferred_from_source_node_edge_counts',
            'storage_complexity_is_a_count_proxy_not_memory_or_runtime',
            'no_probabilistic_or_causal_edges_inferred',
        ],
    }))
