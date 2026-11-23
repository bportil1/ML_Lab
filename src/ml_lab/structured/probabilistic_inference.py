"""ML-UW-34: bounded fitting and exact queries for explicit finite categorical DAGs.

This module consumes ML-UW-33 probabilistic *model* specifications, not source
hierarchies, statistical-association graphs, or inferred causal structures.
Functions are pure, in-memory, domain-neutral and require explicit invocation.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from itertools import product
import json
import math
from types import MappingProxyType
from typing import Any

from .contracts import StructuredContractError
from .object import StructuredObject
from .probabilistic import ProbabilisticModelSpec, source_reference

FIT_SCHEMA = 'ml-lab.probabilistic-fit@1'
QUERY_SCHEMA = 'ml-lab.probabilistic-query@1'
MAX_ENUMERATED_ASSIGNMENTS = 131072
DEFAULT_ENUMERATED_ASSIGNMENTS = 65536
MAX_QUERY_OUTCOMES = 4096
MAX_PSEUDOCOUNT = 1_000_000.0


def _fail(message: str) -> None:
    raise StructuredContractError(message)


def _integer(value: object, where: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(f'{where} must be an integer between {minimum} and {maximum}')
    return value


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False)


def _digest(value: object) -> str:
    return 'sha256:' + hashlib.sha256(_canonical(value).encode('utf-8')).hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class FittingResult:
    """A separately retained fitted specification and detached fit diagnostics."""

    model: ProbabilisticModelSpec
    _report: Mapping[str, Any]

    @property
    def report(self) -> dict:
        return _thaw(self._report)

    def to_dict(self) -> dict:
        return {'model': self.model.to_dict(), 'report': self.report}


def fit_categorical_model(
    model: ProbabilisticModelSpec,
    *,
    source: StructuredObject,
    pseudocount: float = 0.0,
    min_complete_rows: int = 1,
    min_parent_rows: int = 1,
    replace_specified: bool = False,
) -> FittingResult:
    """Fit unfitted CPTs with frequency or symmetric Dirichlet-smoothed counts.

    With ``replace_specified=False``, fully specified tables are preserved; an
    entirely specified model must be explicitly refitted to estimate it anew.
    ``min_parent_rows=0`` permits prior-only rows only with positive smoothing.
    Missing/repeated cells follow the specification's *global* complete-case
    policy; no imputation, EM, learned arcs, or independence claims.
    """
    if not isinstance(model, ProbabilisticModelSpec):
        _fail('a validated ProbabilisticModelSpec is required')
    if not isinstance(source, StructuredObject) or source_reference(source) != model.source_reference:
        _fail('source reference does not match the probabilistic model')
    if type(pseudocount) not in (int, float) or not math.isfinite(pseudocount) or not 0 <= pseudocount <= MAX_PSEUDOCOUNT:
        _fail(f'pseudocount must be finite and in [0, {MAX_PSEUDOCOUNT:g}]')
    pseudocount = float(pseudocount)
    _integer(min_complete_rows, 'min_complete_rows', minimum=1, maximum=1_000_000_000)
    _integer(min_parent_rows, 'min_parent_rows', minimum=0, maximum=1_000_000_000)
    if type(replace_specified) is not bool:
        _fail('replace_specified must be a boolean')
    if min_parent_rows == 0 and pseudocount == 0:
        _fail('min_parent_rows=0 requires positive pseudocount for unobserved assignments')

    raw = model.to_dict()
    variables = {v['id']: v for v in raw['variables']}
    parents = model.audit['parent_variables']
    targets = [cond['variable_id'] for cond in raw['conditionals']
               if replace_specified or cond['status'] == 'unfitted']
    if not targets:
        _fail('no unfitted conditional tables; explicitly enable replace_specified to refit')

    complete_rows = []
    names = {v['id']: v['observed_variable'] for v in raw['variables']}
    for observation in source.observations:
        assignment: dict[str, str] = {}
        for variable_id, name in names.items():
            cell = observation['values'][name]
            if cell['status'] not in ('present', 'empty'):
                break
            assignment[variable_id] = cell['value']
        if len(assignment) == len(names):
            complete_rows.append(assignment)
    support = model.audit['support']
    if len(complete_rows) != support['complete_case_rows']:
        _fail('model observation support changed unexpectedly')
    if len(complete_rows) < min_complete_rows:
        _fail('insufficient complete-case rows for categorical fitting')

    diagnostics = []
    for cond in raw['conditionals']:
        name = cond['variable_id']
        if name not in targets:
            diagnostics.append({'variable_id': name, 'action': 'preserved', 'rows': []})
            continue
        state_labels = variables[name]['states']
        parent_ids = parents[name]
        counts: dict[tuple[str, ...], Counter] = {}
        for row in complete_rows:
            key = tuple(row[parent] for parent in parent_ids)
            counts.setdefault(key, Counter())[row[name]] += 1
        rows = []
        row_diagnostics = []
        for configuration in product(*(variables[parent]['states'] for parent in parent_ids)):
            counter = counts.get(configuration, Counter())
            observed = sum(counter.values())
            given = dict(zip(parent_ids, configuration))
            if observed < min_parent_rows:
                _fail(f'{name}: parent assignment {given} has {observed} observations; '
                      f'min_parent_rows={min_parent_rows}')
            denominator = observed + pseudocount * len(state_labels)
            if denominator <= 0 or not math.isfinite(denominator):
                _fail(f'{name}: conditional denominator is not finite and positive')
            probabilities = {state: (counter[state] + pseudocount) / denominator
                             for state in state_labels}
            rows.append({'given': given, 'probabilities': probabilities})
            row_diagnostics.append({
                'given': given, 'observed_rows': observed,
                'state_counts': {state: counter[state] for state in state_labels},
                'prior_only': observed == 0,
            })
        action = 'refitted' if cond['status'] == 'specified' else 'fitted'
        cond.update(status='specified', rows=rows)
        diagnostics.append({'variable_id': name, 'action': action,
                            'rows': row_diagnostics})

    # Provenance is copied, never written to the user's source model. A
    # previously fitted model may be deliberately refitted with a new run.
    raw.pop('model_id')
    raw['provenance'] = dict(raw['provenance'])
    raw['provenance']['ml_uw_34_fit'] = {
        'input_model_id': model.model_id,
        'method': 'symmetric_dirichlet_posterior_mean' if pseudocount else 'empirical_frequency',
        'pseudocount_per_state': pseudocount,
        'min_complete_rows': min_complete_rows,
        'min_parent_rows': min_parent_rows,
        'replace_specified': replace_specified,
    }
    fitted = ProbabilisticModelSpec.from_dict(raw, source=source)
    fitted.require_fully_specified()
    fitted_conditionals = {c['variable_id']: c for c in fitted.to_dict()['conditionals']}
    fit_parents = fitted.audit['parent_variables']
    fit_tables = {name: {
        tuple(r['given'][parent] for parent in fit_parents[name]): r['probabilities']
        for r in cond['rows']} for name, cond in fitted_conditionals.items()}
    log_likelihood_terms = []
    zero_likelihood_rows = 0
    for assignment in complete_rows:
        term = 0.0
        for name in variables:
            given = tuple(assignment[parent] for parent in fit_parents[name])
            probability = fit_tables[name][given][assignment[name]]
            if probability == 0:
                zero_likelihood_rows += 1
                term = None
                break
            term += math.log(probability)
        if term is not None:
            log_likelihood_terms.append(term)
    log_likelihood = (math.fsum(log_likelihood_terms)
                      if not zero_likelihood_rows else None)
    if log_likelihood is not None and not math.isfinite(log_likelihood):
        log_likelihood = None
    report = {
        'schema': FIT_SCHEMA, 'input_model_id': model.model_id,
        'fitted_model_id': fitted.model_id, 'source': model.source_reference,
        'configuration': raw['provenance']['ml_uw_34_fit'],
        'support': support, 'conditionals': diagnostics,
        'training_diagnostics': {
            'complete_case_log_likelihood': log_likelihood,
            'zero_likelihood_rows': zero_likelihood_rows,
            'evaluated_complete_case_rows': len(complete_rows),
            'evaluation_scope': 'in_sample_only_not_holdout_validation',
        },
        'diagnostics': {
            'observation_independence': 'not_verified',
            'missingness_mechanism': 'not_inferred',
            'excluded_rows_are_not_imputed': True,
            'source_edges_used_as_model_arcs': False,
            'prior_only_parent_rows': sum(
                row['prior_only'] for d in diagnostics for row in d['rows']),
            'smoothing_is_user_specified': True,
        },
    }
    report['fit_id'] = _digest(report)
    return FittingResult(fitted, _freeze(report))


def _log_add(left: float, right: float) -> float:
    if left == -math.inf:
        return right
    if right == -math.inf:
        return left
    high, low = (left, right) if left >= right else (right, left)
    return high + math.log1p(math.exp(low - high))


def query_exact(
    model: ProbabilisticModelSpec,
    *,
    variables: Sequence[str],
    evidence: Mapping[str, str] | None = None,
    max_assignments: int = DEFAULT_ENUMERATED_ASSIGNMENTS,
) -> dict:
    """Exact, bounded categorical BN posterior by evidence-restricted enumeration.

    Complexity is the product of the *unobserved* model state-domain sizes.
    No Monte Carlo approximation or structure learning occurs. Probability
    accumulation uses log-sum-exp to avoid underflow for rare evidence.
    """
    if not isinstance(model, ProbabilisticModelSpec):
        _fail('a validated ProbabilisticModelSpec is required')
    model.require_fully_specified()
    raw = model.to_dict()
    by_id = {v['id']: v for v in raw['variables']}
    if not isinstance(variables, (tuple, list)) or not variables:
        _fail('query variables must be a nonempty array')
    if any(type(v) is not str or v not in by_id for v in variables):
        _fail('query contains an unknown model variable')
    if len(set(variables)) != len(variables):
        _fail('query variables must be distinct')
    if evidence is None:
        evidence = {}
    if not isinstance(evidence, Mapping) or any(type(k) is not str for k in evidence):
        _fail('evidence must map variable IDs to finite categorical labels')
    evidence = dict(evidence)
    if set(evidence) - set(by_id):
        _fail('evidence contains an unknown model variable')
    if set(variables) & set(evidence):
        _fail('query variables and evidence must be disjoint')
    for key, state in evidence.items():
        if type(state) is not str or state not in by_id[key]['states']:
            _fail(f'evidence for {key} is outside its declared state domain')
    _integer(max_assignments, 'max_assignments', minimum=1,
             maximum=MAX_ENUMERATED_ASSIGNMENTS)

    queried = list(variables)
    query_state_axes = [by_id[name]['states'] for name in queried]
    result_size = math.prod(len(axis) for axis in query_state_axes)
    if result_size > MAX_QUERY_OUTCOMES:
        _fail(f'query has {result_size} outcomes; limit is {MAX_QUERY_OUTCOMES}')
    model_names = list(by_id)
    iterables = [[evidence[name]] if name in evidence else by_id[name]['states']
                 for name in model_names]
    assignments = math.prod(len(axis) for axis in iterables)
    if assignments > max_assignments:
        _fail(f'exact query would enumerate {assignments} assignments; '
              f'max_assignments={max_assignments}')

    parents = model.audit['parent_variables']
    tables: dict[str, dict[tuple[str, ...], dict[str, float]]] = {}
    for cond in raw['conditionals']:
        name = cond['variable_id']
        p = parents[name]
        tables[name] = {tuple(row['given'][parent] for parent in p): row['probabilities']
                        for row in cond['rows']}

    log_total = -math.inf
    log_outcomes: dict[tuple[str, ...], float] = {}
    nonzero_assignments = 0
    for states in product(*iterables):
        assignment = dict(zip(model_names, states))
        log_joint = 0.0
        for name in model_names:
            parent_state = tuple(assignment[parent] for parent in parents[name])
            probability = tables[name][parent_state][assignment[name]]
            if probability == 0:
                log_joint = -math.inf
                break
            log_joint += math.log(probability)
        if log_joint == -math.inf:
            continue
        nonzero_assignments += 1
        log_total = _log_add(log_total, log_joint)
        target_key = tuple(assignment[name] for name in queried)
        log_outcomes[target_key] = _log_add(log_outcomes.get(target_key, -math.inf), log_joint)

    report: dict = {
        'schema': QUERY_SCHEMA, 'model_id': model.model_id,
        'source': model.source_reference,
        'variables': queried, 'evidence': evidence,
        'algorithm': 'bounded_exact_enumeration_log_domain',
        'audit': {
            'enumerated_assignments': assignments,
            'nonzero_assignments': nonzero_assignments,
            'max_assignments': max_assignments,
            'source_graph_is_probability_model': False,
            'causal_interpretation': False,
        },
    }
    if log_total == -math.inf:
        report.update(status='unavailable', reason='zero_probability_evidence',
                      probability_evidence=0.0, log_probability_evidence=None,
                      posterior=None)
        return report

    # Underflow of *reported* evidence probability must not be mistaken for
    # impossible evidence; the log probability remains informative.
    prob_evidence = math.exp(log_total)  # includes representable subnormals
    table = []
    for state_combination in product(*query_state_axes):
        log_mass = log_outcomes.get(state_combination, -math.inf)
        probability = math.exp(log_mass - log_total) if log_mass != -math.inf else 0.0
        table.append({'given': dict(zip(queried, state_combination)),
                      'probability': probability})
    total = math.fsum(row['probability'] for row in table)
    if not math.isfinite(total) or total <= 0:
        _fail('posterior normalization failed')
    for row in table:
        row['probability'] /= total
    report.update(status='ok', reason=None,
                  probability_evidence=prob_evidence,
                  log_probability_evidence=log_total, posterior=table)
    report['audit']['probability_evidence_underflow'] = prob_evidence == 0.0
    return report
