"""Matched, source-neutral representation comparisons (ML-UW-35).

This deliberately does *not* infer feature values from source-graph topology.
The common holdout partition is fixed before any candidate transformation, and
all estimates use the same complete-case observation IDs. Feature selection and
structural retention are reported separately so graph removal cannot be credited
with predictive changes that never occurred in the observed feature matrix.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

from .contracts import StructuredContractError
from .information import (
    estimate_conditional_mutual_information, estimate_entropy,
    estimate_mutual_information,
)
from .object import StructuredObject
from .probabilistic import source_reference
from .transformations import TransformationPlan, transform

STUDY_SCHEMA = 'ml-lab.representation-comparison@1'
PLAN_SCHEMA = 'ml-lab.representation-study-plan@1'
MAX_CANDIDATES = 12
MAX_ROWS = 50000
MAX_JOINT_STATES = 4096


def _error(message: str) -> None:
    raise StructuredContractError(message)


def _encode(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False)


def _hash(value: Any) -> str:
    return 'sha256:' + hashlib.sha256(_encode(value).encode('utf-8')).hexdigest()


def _name(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _error(f'{where} must be a nonempty string')
    return value


def _names(values: Any, where: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        _error(f'{where} must be a tuple of variable IDs')
    for value in values:
        _name(value, where)
    if len(values) != len(set(values)):
        _error(f'{where} requires unique variables')
    return values


@dataclass(frozen=True, slots=True)
class ComparisonCandidate:
    label: str
    plan: TransformationPlan
    predictors: tuple[str, ...]

    def __post_init__(self) -> None:
        _name(self.label, 'candidate label')
        if not isinstance(self.plan, TransformationPlan):
            _error('candidate requires a validated TransformationPlan')
        _names(self.predictors, 'predictors')

    def to_dict(self) -> dict[str, Any]:
        return {'label': self.label, 'plan': self.plan.to_dict(),
                'predictors': list(self.predictors)}


@dataclass(frozen=True, slots=True)
class ComparisonStudy:
    """A predeclared, matched study on categorical observations.

    A group variable (e.g. project identity) prevents siblings from crossing
    partitions. Ungrouped rows are accepted only with an explicit independence
    acknowledgement; this does not prove statistical independence.
    """

    target: str
    candidates: tuple[ComparisonCandidate, ...]
    group_variable: str | None = None
    acknowledge_independent_rows: bool = False
    seed: int = 0
    holdout_fraction: float = 0.25
    pseudocount: float = 1.0
    min_train_rows: int = 4
    min_test_rows: int = 2
    max_joint_states: int = MAX_JOINT_STATES

    def __post_init__(self) -> None:
        _name(self.target, 'target')
        if not isinstance(self.candidates, tuple) or not 2 <= len(self.candidates) <= MAX_CANDIDATES:
            _error(f'comparison requires 2 to {MAX_CANDIDATES} candidates')
        if any(not isinstance(candidate, ComparisonCandidate) for candidate in self.candidates):
            _error('all candidates must be validated ComparisonCandidate instances')
        labels = [candidate.label for candidate in self.candidates]
        if len(set(labels)) != len(labels) or 'full' not in labels:
            _error('unique candidate labels including the full reference are required')
        baseline = next(candidate for candidate in self.candidates if candidate.label == 'full')
        if baseline.plan != TransformationPlan('full'):
            _error('the full reference must use the unprojected full transformation')
        reference_variables = set(baseline.predictors)
        for candidate in self.candidates:
            if self.target in candidate.predictors:
                _error('target must not be used as a predictor')
            if not set(candidate.predictors) <= reference_variables:
                _error('candidate predictors must be a subset of full reference predictors')
        if self.group_variable is not None:
            _name(self.group_variable, 'group_variable')
            if self.group_variable == self.target or self.group_variable in reference_variables:
                _error('group variable must not be the target or a predictor')
            if self.acknowledge_independent_rows:
                _error('independent-row acknowledgment conflicts with grouped splitting')
        elif self.acknowledge_independent_rows is not True:
            _error('provide a group_variable or explicitly acknowledge independent rows')
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            _error('seed must be a nonnegative 32-bit integer')
        if type(self.holdout_fraction) not in (float, int) or not math.isfinite(self.holdout_fraction) or not 0 < self.holdout_fraction < 1:
            _error('holdout_fraction must be strictly between zero and one')
        if type(self.pseudocount) not in (float, int) or not math.isfinite(self.pseudocount) or not 0 < self.pseudocount <= 1_000_000:
            _error('pseudocount must be finite and positive')
        for name in ('min_train_rows', 'min_test_rows', 'max_joint_states'):
            value = getattr(self, name)
            maximum = MAX_JOINT_STATES if name == 'max_joint_states' else MAX_ROWS
            if type(value) is not int or not 1 <= value <= maximum:
                _error(f'{name} must be an integer in [1, {maximum}]')

    def to_dict(self) -> dict[str, Any]:
        return {'schema': PLAN_SCHEMA, 'target': self.target,
                'candidates': [candidate.to_dict() for candidate in self.candidates],
                'group_variable': self.group_variable,
                'acknowledge_independent_rows': self.acknowledge_independent_rows,
                'seed': self.seed, 'holdout_fraction': self.holdout_fraction,
                'pseudocount': self.pseudocount, 'min_train_rows': self.min_train_rows,
                'min_test_rows': self.min_test_rows,
                'max_joint_states': self.max_joint_states}

    @classmethod
    def from_dict(cls, raw: Any) -> 'ComparisonStudy':
        if not isinstance(raw, dict) or raw.get('schema') != PLAN_SCHEMA:
            _error('invalid representation study schema')
        allowed = set(cls.__dataclass_fields__) | {'schema'}
        if set(raw) - allowed:
            _error('unknown representation study fields')
        if not {'target', 'candidates'} <= set(raw):
            _error('study requires target and candidates')
        if not isinstance(raw.get('candidates'), list):
            _error('study candidates must be an array')
        candidates = []
        for item in raw['candidates']:
            if not isinstance(item, dict) or set(item) != {'label', 'plan', 'predictors'}:
                _error('candidate has unsupported or missing fields')
            if not isinstance(item['predictors'], list):
                _error('candidate predictors must be an array')
            candidates.append(ComparisonCandidate(item['label'],
                TransformationPlan.from_dict(item['plan']), tuple(item['predictors'])))
        kwargs = {key: value for key, value in raw.items()
                  if key not in {'schema', 'candidates'}}
        return cls(candidates=tuple(candidates), **kwargs)

    def to_json(self) -> str:
        return _encode(self.to_dict())


def _category(cell: Any, where: str) -> str | None:
    """Represent source state and scalar type; never alias bool 1 with int 1."""
    state = cell['status']
    if state in {'missing', 'explicit_null', 'unknown', 'multiple'}:
        return None
    if state == 'empty':
        return _encode(['empty', 'string', ''])
    value = cell['value']
    if type(value) not in (str, int, bool):
        _error(f'{where}: comparison requires categorical scalars (discretize other types first)')
    return _encode([state, type(value).__name__, value])


def _partition(records: list[dict], study: ComparisonStudy) -> tuple[list[dict], list[dict], dict]:
    keyed: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        key = (record['categories'][study.group_variable] if study.group_variable
               else _encode(['row', record['id']]))
        keyed[key].append(record)
    if len(keyed) < 2:
        _error('split requires at least two independent groups or rows')
    order = sorted(keyed, key=lambda key: (_hash([study.seed, key]), key))
    test_count = max(1, min(len(order) - 1,
                            int(round(len(order) * study.holdout_fraction))))
    test_keys = set(order[:test_count])
    train = [record for record in records if (
        record['categories'][study.group_variable] if study.group_variable
        else _encode(['row', record['id']])) not in test_keys]
    test = [record for record in records if (
        record['categories'][study.group_variable] if study.group_variable
        else _encode(['row', record['id']])) in test_keys]
    if len(train) < study.min_train_rows or len(test) < study.min_test_rows:
        _error('partition has insufficient train/test rows under declared thresholds')
    train_ids = sorted(record['id'] for record in train)
    test_ids = sorted(record['id'] for record in test)
    return train, test, {
        'method': 'group_disjoint_hash_partition' if study.group_variable else 'row_hash_partition',
        'group_variable': study.group_variable, 'seed': study.seed,
        'requested_holdout_fraction': study.holdout_fraction,
        'actual_holdout_fraction': len(test) / len(records),
        'train_rows': len(train), 'test_rows': len(test),
        'train_group_count': len(keyed) - len(test_keys),
        'test_group_count': len(test_keys),
        'train_observation_ids': train_ids, 'test_observation_ids': test_ids,
        'train_id_digest': _hash(train_ids), 'test_id_digest': _hash(test_ids),
        'disjoint_groups': True if study.group_variable else None,
        'independence_assumption': 'unverified' if not study.group_variable else 'within_group_dependence_held_out_together',
    }


def _joint(record: dict, names: tuple[str, ...]) -> str:
    return _encode([record['categories'][name] for name in names])


def _temporary_information_object(source: StructuredObject,
                                  train: list[dict], target: str,
                                  features: tuple[str, ...],
                                  omitted: tuple[str, ...]) -> StructuredObject:
    doc = source.to_dict()
    original = {row['observation_id']: row for row in doc['observations']}
    doc.pop('object_id', None)
    doc['observations'] = [{
        'observation_id': record['id'], 'values': {
            '__study_target__': original[record['id']]['values'][target],
            '__study_features__': {'status': 'present', 'value': _joint(record, features)},
            '__study_omitted__': {'status': 'present', 'value': _joint(record, omitted)},
        },
    } for record in train]
    # The three temporary variables are analysis-only categorical encodings.
    doc['provenance'] = {'study_temporary_observations': True,
                         'source_object_id': source.object_id}
    return StructuredObject.from_dict(doc)


def _score(train: list[dict], test: list[dict], predictors: tuple[str, ...],
           alpha: float, max_states: int) -> dict[str, Any]:
    rows: dict[str, Counter] = defaultdict(Counter)
    marginal: Counter = Counter()
    for record in train:
        label = record['target_category']
        rows[_joint(record, predictors)][label] += 1
        marginal[label] += 1
    if len(rows) > max_states:
        _error(f'feature configurations exceed max_joint_states={max_states}')
    labels = sorted(marginal)
    n_labels = len(labels)
    accuracy = 0
    penalties: list[float] = []
    unseen = 0
    fallback = 0
    for record in test:
        actual = record['target_category']
        if actual not in marginal:
            unseen += 1
            continue
        key = _joint(record, predictors)
        counts = rows.get(key)
        if counts is None:
            counts = marginal
            fallback += 1
        denominator = sum(counts.values()) + alpha * n_labels
        probs = {label: (counts[label] + alpha) / denominator for label in labels}
        predicted = min(labels, key=lambda label: (-probs[label], label))
        accuracy += predicted == actual
        penalties.append(-math.log(probs[actual]))
    status = 'ok' if unseen == 0 else 'unseen_holdout_target_state'
    # Do not silently discard unscorable outcomes to inflate accuracy.
    return {
        'status': status,
        'holdout_log_loss_nats': math.fsum(penalties) / len(test) if status == 'ok' else None,
        'holdout_accuracy': accuracy / len(test) if status == 'ok' else None,
        'train_target_states': n_labels,
        'train_feature_configurations': len(rows),
        'unseen_feature_fallback_rows': fallback,
        'unseen_target_rows': unseen,
        'smoothing_per_state': alpha,
        'scoring': 'categorical_conditional_frequency_with_smoothed_marginal_fallback',
        'score_direction': {'holdout_log_loss_nats': 'lower', 'holdout_accuracy': 'higher'},
    }


def compare_representations(source: StructuredObject, study: ComparisonStudy) -> dict[str, Any]:
    """Return a deterministic, JSON-compatible, matched representation study.

    Holdout log loss is an *assessment* of a fixed, predeclared candidate, not
    an unbiased final test after selecting candidates on those same scores.
    Joint categorical information is calculated on training rows only; it is
    not an out-of-sample information estimator.
    """
    if not isinstance(source, StructuredObject) or not isinstance(study, ComparisonStudy):
        _error('validated StructuredObject and ComparisonStudy are required')
    source.require_aligned((study.target,))
    names = {study.target}
    for candidate in study.candidates:
        names.update(candidate.predictors)
    if study.group_variable:
        names.add(study.group_variable)
    source.require_aligned(tuple(sorted(names)))
    if len(source.observations) > MAX_ROWS:
        _error(f'comparison input exceeds {MAX_ROWS} rows')
    for candidate in study.candidates:
        if candidate.plan.variables is not None and not {
            study.target, *candidate.predictors
        } <= set(candidate.plan.variables):
            _error(f'{candidate.label}: transformation projection removes target or requested predictor')

    # Enforce the common complete-case intersection *before* partitioning.
    records = []
    excluded = Counter()
    for row in source.observations:
        categories = {}
        unavailable = []
        for name in sorted(names):
            category = _category(row['values'][name], name)
            if category is None:
                unavailable.append(name)
            else:
                categories[name] = category
        if unavailable:
            excluded['incomplete_row'] += 1
            continue
        records.append({'id': row['observation_id'], 'categories': categories,
                        'target_category': categories[study.target]})
    if len(records) < study.min_train_rows + study.min_test_rows:
        _error('common complete-case intersection has insufficient observations')

    train, test, split = _partition(records, study)
    candidates = []
    baseline = next(item for item in study.candidates if item.label == 'full')
    baseline_predictors = tuple(baseline.predictors)
    base_nodes = len(source.nodes)
    base_edges = len(source.edges)
    source_json_bytes = len(source.to_json().encode('utf-8'))
    for candidate in study.candidates:
        view = transform(source, candidate.plan)
        if tuple(row['observation_id'] for row in view.object.observations) != tuple(
                row['observation_id'] for row in source.observations):
            _error('candidate transformation changed observation alignment')
        if candidate.plan.variables is not None and not set(candidate.predictors) <= set(
                view.object.observations[0]['values']):
            _error('candidate predictors are missing from its transformed view')
        omitted = tuple(name for name in baseline_predictors
                        if name not in candidate.predictors)
        transient = _temporary_information_object(source, train, study.target,
                                                   candidate.predictors, omitted)
        options = {'missing_policy': 'error', 'min_samples': study.min_train_rows}
        h_target = estimate_entropy(transient, '__study_target__', **options)
        mutual = estimate_mutual_information(
            transient, '__study_target__', '__study_features__', **options)
        if omitted:
            conditional = estimate_conditional_mutual_information(
                transient, '__study_target__', '__study_omitted__',
                given=['__study_features__'], min_stratum_samples=2, **options)
            incremental = {'status': conditional['status'],
                           'bits': conditional['value'],
                           'conditioning_strata': conditional['support']['conditioning_strata'],
                           'under_supported_strata': conditional['support']['under_supported_strata']}
        else:
            incremental = {'status': 'exact_identity', 'bits': 0.0,
                           'conditioning_strata': None, 'under_supported_strata': None}
        scores = _score(train, test, candidate.predictors, float(study.pseudocount),
                        study.max_joint_states)
        structure = {
            'source_node_count': base_nodes, 'retained_node_count': len(view.object.nodes),
            'node_identity_fraction': len(view.object.nodes) / base_nodes,
            'source_edge_count': base_edges, 'retained_edge_count': len(view.object.edges),
            'edge_identity_fraction': len(view.object.edges) / base_edges if base_edges else None,
            'retained_predictors': list(candidate.predictors),
            'omitted_baseline_predictors': list(omitted),
            'topology': view.object.topology['kind'],
            'view_json_bytes': len(view.object.to_json().encode('utf-8')),
            'source_json_bytes': source_json_bytes,
            'structure_affects_predictors': False,
        }
        candidates.append({
            'label': candidate.label, 'plan': candidate.plan.to_dict(),
            'output_object_id': view.object.object_id,
            'structural_audit': view.audit, 'retention': structure,
            'information_train_only': {
                'method': 'empirical_discrete_plugin', 'unit': 'bits',
                'target_entropy_bits': h_target['value'],
                'target_predictor_mi_bits': mutual['value'],
                'omitted_predictor_cmi_given_retained': incremental,
                'support_rows': len(train),
                'shared_training_rows': True,
                'finite_sample_bias_warning': True,
            },
            'predictive_holdout': scores,
        })
    ref = next(item for item in candidates if item['label'] == 'full')
    reference_score = ref['predictive_holdout']['holdout_log_loss_nats']
    for candidate in candidates:
        value = candidate['predictive_holdout']['holdout_log_loss_nats']
        candidate['delta_holdout_log_loss_vs_full_nats'] = (
            value - reference_score if value is not None and reference_score is not None
            else None)
    comparable = [item for item in candidates
                  if item['predictive_holdout']['status'] == 'ok']
    best = (min(comparable, key=lambda item: (item['predictive_holdout']['holdout_log_loss_nats'],
                                           item['label']))['label'] if comparable else None)
    report = {
        'schema': STUDY_SCHEMA, 'plan': study.to_dict(), 'source': source_reference(source),
        'observation_unit': source.observation_unit,
        'objective': 'minimize_matched_holdout_categorical_log_loss',
        'selection_warning': 'Choosing a candidate by these holdout scores requires an independent final test.',
        'support': {
            'source_rows': len(source.observations), 'common_complete_rows': len(records),
            'excluded_rows': len(source.observations) - len(records),
            'exclusion_reasons': dict(sorted(excluded.items())),
            'same_observation_ids_for_all_candidates': True,
            'no_post_transform_observation_joins': True,
        },
        'split': split, 'candidates': candidates,
        'advisory': {
            'lowest_observed_holdout_log_loss': best,
            'structural_loss_without_feature_change_is_not_predictive_evidence': True,
            'feature_reduction_and_structural_reduction_are_separate_interventions': True,
            'probabilistic_graph_structure_not_inferred': True,
            'not_a_causal_structure_effect': True,
            'uncertainty_intervals_or_significance_tests': 'not_available',
            'sample_independence': 'not_proven',
            'execution_time_or_peak_memory': 'not_measured',
            'view_json_bytes_are_storage_proxy_not_execution_cost': True,
            'reconstruction_accuracy_not_measured': True,
        },
    }
    report['study_id'] = _hash(report)
    return report
