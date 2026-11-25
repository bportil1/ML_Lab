"""Observed-variable *whole-network* information, separate from source topology.

Estimate all requested unordered MI/CMI pairs over aligned observation records.
No causal, learned-probabilistic, node-to-sample, or edge direction claims.
The full matrix is reported independently of a visual edge cutoff.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from .contracts import StructuredContractError
from .object import StructuredObject
from .bindings import source_sha256
from .information import (
    _bin_index, _binning_definition, _categorical, _conditional_mi,
    _entropy, _numeric, _parameters,
)

SCHEMA = 'ml-lab.observed-information-network@1'
MAX_VARIABLES = 128
MAX_ROWS = 10_000
MAX_PAIR_ROW_WORK = 10_000_000
MAX_JOINT_STATES = 5_000


def _fail(text: str) -> None:
    raise StructuredContractError(text)


def _variable_names(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or any(not isinstance(x, str) or not x.strip() for x in value):
        _fail(f'{label} must be an array of nonempty variable names')
    if len(value) != len(set(value)):
        _fail(f'{label} contains duplicate variable names')
    return tuple(value)



def analyze_observed_network(
    source: StructuredObject, *, variables: tuple[str, ...] = (),
    conditioning: tuple[str, ...] = (), bins: Mapping[str, Mapping[str, Any]] | None = None,
    missing_policy: str = 'complete_case', min_samples: int = 5,
    min_stratum_samples: int = 2,
) -> dict[str, Any]:
    """Return exhaustive matrix entries and per-pair sample support.

    Default variables=() selects every observed column *except* the explicit
    conditioning columns. Conditioning is a single declared set Z for every
    tested pair. This is NOT an automatically inferred conditional graph.
    """
    if not isinstance(source, StructuredObject):
        _fail('validated StructuredObject required')
    requested = _variable_names(variables, 'variables')
    conditioning = _variable_names(conditioning, 'conditioning')
    if type(min_samples) is not int or not 2 <= min_samples <= MAX_ROWS:
        _fail('min_samples must be an integer in [2, 10000]')
    if type(min_stratum_samples) is not int or not 2 <= min_stratum_samples <= MAX_ROWS:
        _fail('min_stratum_samples must be an integer in [2, 10000]')
    if not source.observations:
        _fail('Whole-network information requires aligned observation records. A single XML structure is not an observation population; use an aligned collection or table.')
    if len(source.observations) > MAX_ROWS:
        _fail('Whole-network workbench exceeds 10000 rows; use headless analysis')
    source.require('conditional_information')
    all_names = tuple(sorted(source.observations[0]['values']))
    selected = requested or tuple(name for name in all_names if name not in conditioning)
    if not 2 <= len(selected) <= MAX_VARIABLES:
        _fail('Whole-network analysis requires 2–128 observed variables; choose an explicit subset for larger tables')
    if len(conditioning) > 8:
        _fail('At most 8 explicit conditioning columns are supported interactively')
    if set(selected) & set(conditioning):
        _fail('conditioning variables cannot also be graph endpoint variables')
    if not set((*selected, *conditioning)) <= set(all_names):
        _fail('selected and conditioning names must be original observed variable IDs')
    source.require_aligned((*selected, *conditioning))
    comparisons = len(selected) * (len(selected) - 1) // 2
    if comparisons * len(source.observations) > MAX_PAIR_ROW_WORK:
        _fail(f'Exhaustive {comparisons}-pair × {len(source.observations)}-row work exceeds {MAX_PAIR_ROW_WORK:,}; explicitly choose fewer variables or use headless analysis. No sampling performed.')
    bins = {} if bins is None else bins
    _parameters(bins=bins, variables=(*selected, *conditioning),
                min_samples=min_samples, min_stratum_samples=min_stratum_samples,
                log_base=2.0, missing_policy=missing_policy, estimator='empirical_discrete')
    # Fit one descriptive bin definition per selected variable, using the same
    # observed numeric values throughout the run (never per-edge refitting).
    fitted = {}
    for name, definition in bins.items():
        values = [_numeric(row['values'][name]['value'], name)
                  for row in source.observations if row['values'][name]['status'] == 'present']
        fitted[name] = _binning_definition(values, name, dict(definition))

    names = (*selected, *conditioning)
    encoded: dict[str, list[tuple[str, Any] | None]] = {name: [] for name in names}
    nonpresent = {name: Counter() for name in names}
    for row in source.observations:
        for name in names:
            cell = row['values'][name]
            state = cell['status']
            if state == 'multiple':
                _fail(f'{name}: repeated values require explicit resolution/alignment before network analysis')
            if state in ('missing', 'explicit_null', 'unknown'):
                nonpresent[name][state] += 1
                if missing_policy == 'error':
                    _fail(f'{name}: missing observation under error policy')
                encoded[name].append(None if missing_policy == 'complete_case' else ('state', state))
            elif state == 'empty':
                encoded[name].append(('state', 'empty'))
            elif name in fitted:
                spec = fitted[name]
                encoded[name].append(('bin', _bin_index(_numeric(cell['value'], name), spec)))
            else:
                encoded[name].append(_categorical(cell['value'], name))

    nodes = []
    for index, name in enumerate(selected):
        observed = [v for v in encoded[name] if v is not None]
        counts = Counter(observed)
        value = _entropy(counts, len(observed), 2.0) if len(observed) >= min_samples else None
        nodes.append({
            'id': f'var:{index:04d}', 'variable_id': name,
            'status': 'ok' if value is not None else 'insufficient_samples',
            'entropy_bits': value, 'support': len(observed),
            'distinct_states': len(counts),
            'nonpresent_counts': dict(sorted(nonpresent[name].items())),
        })

    # Multi-variable dependency requires a *common* sample and a feasible
    # joint space. This is a descriptive total-correlation statistic, not a
    # learned multivariate distribution or an allocation to structural edges.
    if len(selected) > 12:
        joint = {'status': 'joint_dimensionality_limit', 'support': None,
                 'joint_entropy_bits': None, 'total_correlation_bits': None,
                 'distinct_joint_states': None,
                 'reason': 'full joint plug-in estimate is capped at 12 variables'}
    else:
        common = [tuple(encoded[name][rownum] for name in selected)
                  for rownum in range(len(source.observations))]
        common = [record for record in common if all(cell is not None for cell in record)]
        n_common = len(common)
        states = Counter(common)
        if n_common < min_samples:
            reason = 'insufficient_common_complete_case_rows'
        elif len(states) > MAX_JOINT_STATES:
            reason = 'joint_state_limit'
        else:
            reason = None
        if reason:
            joint = {'status': reason, 'support': n_common,
                     'joint_entropy_bits': None, 'total_correlation_bits': None,
                     'distinct_joint_states': len(states), 'reason': reason}
        else:
            h_joint = _entropy(states, n_common, 2.0)
            entropy_sum = sum(_entropy(Counter(row[i] for row in common),
                                         n_common, 2.0) for i in range(len(selected)))
            total_correlation = entropy_sum - h_joint
            if total_correlation < -1e-9:
                _fail('nonnegative joint total-correlation invariant violated')
            joint = {'status': 'ok', 'support': n_common,
                     'joint_entropy_bits': h_joint,
                     'total_correlation_bits': max(0.0, total_correlation),
                     'distinct_joint_states': len(states),
                     'reason': None,
                     'warnings': ['sparse_joint_states_relative_to_support']
                         if len(states) * 5 >= n_common else []}

    pairs = []
    nrows = len(source.observations)
    for i, x in enumerate(selected):
        for j in range(i + 1, len(selected)):
            y = selected[j]
            samples = []
            for rownum in range(nrows):
                a, b = encoded[x][rownum], encoded[y][rownum]
                z = tuple(encoded[name][rownum] for name in conditioning)
                if a is not None and b is not None and all(v is not None for v in z):
                    samples.append((a, b, z))
            n = len(samples)
            strata = Counter(z for _, _, z in samples)
            joint_states = len(set(samples))
            warning = []
            if n and joint_states * 5 >= n:
                warning.append('sparse_joint_states_relative_to_support')
            if n and (len(set(a for a, _, _ in samples)) *
                      len(set(b for _, b, _ in samples))) >= n:
                warning.append('high_cardinality_relative_to_support')
            if n < min_samples:
                status = 'insufficient_samples'
            elif joint_states > MAX_JOINT_STATES:
                status = 'joint_state_limit'
            elif conditioning and any(count < min_stratum_samples for count in strata.values()):
                status = 'insufficient_conditioning_support'
            else:
                status = 'ok'
            value = _conditional_mi(samples, 2.0) if status == 'ok' else None
            # CMI is nonnegative in exact arithmetic; roundoff can yield tiny negatives.
            if value is not None and value < -1e-9:
                _fail('nonfinite or unexpectedly negative information estimate')
            if value is not None:
                value = max(0.0, value)
            pairs.append({
                'id': f'pair:{i:04d}:{j:04d}', 'source_id': nodes[i]['id'],
                'target_id': nodes[j]['id'], 'x': x, 'y': y,
                'status': status, 'value_bits': value,
                'support': n, 'excluded_rows': nrows - n,
                'joint_states': joint_states, 'conditioning_strata': len(strata),
                'under_supported_strata': sum(v < min_stratum_samples for v in strata.values()),
                'warnings': warning,
            })
    return {
        'schema': SCHEMA, 'source_digest': source_sha256(source),
        'observation_unit': source.observation_unit, 'variables': list(selected),
        'conditioning': list(conditioning), 'metric': 'empirical_conditional_mutual_information'
            if conditioning else 'empirical_mutual_information',
        'estimator': {'name': 'empirical_discrete_plugin', 'log_base': 2.0,
                      'missing_policy': missing_policy, 'min_samples': min_samples,
                      'min_stratum_samples': min_stratum_samples,
                      'discretization': fitted, 'bias_correction': None,
                      'confidence_intervals': False, 'independence_verified': False},
        'support': {'aligned_rows': nrows, 'total_variables': len(selected),
                    'pairs_requested': comparisons,
                    'pairs_measured': sum(p['status'] == 'ok' for p in pairs)},
        'nodes': nodes, 'pairs': pairs, 'joint_dependency': joint,
        'limitations': [
            'source_topology_edges_are_not_statistical_associations',
            'all_observed_variables_are_not_automatically_mapped_to_structure_nodes',
            'edges_are_symmetric_observational_associations_not_causal_or_propagation_links',
            'pairwise_complete_case_sample_support_may_differ_between_edges'
                if missing_policy == 'complete_case' else 'missing_states_modeled_as_distinct_categories',
            'plug_in_entropy_and_mi_are_finite_sample_biased',
            'numeric_looking_CSV_strings_remain_literal_categories_without_explicit_typed_conversion',
            'numeric_discretization_cutpoints_fitted_on_all_available_rows_not_for_prediction',
            'CMI_conditions_only_on_explicit_global_Z_not_automatically_on_all_other_nodes',
            'joint_total_correlation_is_unconditional_even_if_pairwise_CMI_is_requested',
            'joint_total_correlation_is_not_an_edge_weight_or_evidence_of_causality',
            'zero_or_small_pairwise_MI_does_not_rule_out_synergy_or_XOR_dependencies',
            'no_significance_tests_confidence_intervals_or_learned_joint_probabilistic_model',
        ],
    }
