"""ML-UW-32: finite-sample discrete information measures on aligned observations.

This module consumes only validated StructuredObject observation cells. A source
edge is never a statistical dependency, and a graph node is never counted as an
independent observation. Continuous measurements require explicit discretization.
No persistence, source-specific parsing, or third-party estimators are used.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from collections.abc import Mapping, Sequence
import math
from typing import Any

from .contracts import StructuredContractError
from .object import StructuredObject

INFORMATION_SCHEMA = 'ml-lab.structured-information@1'
MEASURES = frozenset({'entropy', 'conditional_entropy', 'mutual_information',
                      'conditional_mutual_information'})
MISSING_POLICIES = frozenset({'complete_case', 'as_category', 'error'})
BINNING_MODES = frozenset({'quantile', 'equal_width'})


def _names(x: str, y: str | None, given: Sequence[str], measure: str) -> tuple[str, ...]:
    if measure not in MEASURES:
        raise StructuredContractError('unsupported information measure')
    if not isinstance(x, str) or not x.strip():
        raise StructuredContractError('x must be a nonempty variable name')
    if y is not None and (not isinstance(y, str) or not y.strip()):
        raise StructuredContractError('y must be a nonempty variable name')
    if not isinstance(given, (list, tuple)):
        raise StructuredContractError('given must be a list or tuple of variable names')
    if any(not isinstance(name, str) or not name.strip() for name in given):
        raise StructuredContractError('given contains an invalid variable name')
    if measure in {'mutual_information', 'conditional_mutual_information'} and y is None:
        raise StructuredContractError(f'{measure} requires y')
    if measure in {'entropy', 'conditional_entropy'} and y is not None:
        raise StructuredContractError(f'{measure} does not use y; use given for conditioning')
    if measure in {'entropy', 'mutual_information'} and given:
        raise StructuredContractError(f'{measure} does not use conditioning variables')
    if measure in {'conditional_entropy', 'conditional_mutual_information'} and not given:
        raise StructuredContractError(f'{measure} requires conditioning variables')
    variables = (x,) + ((y,) if y is not None else ()) + tuple(given)
    if len(set(variables)) != len(variables):
        raise StructuredContractError('information analysis requires distinct variable names')
    return variables


def _parameters(
    *, bins: Mapping[str, Mapping[str, Any]] | None,
    variables: tuple[str, ...], min_samples: int, min_stratum_samples: int,
    log_base: float, missing_policy: str, estimator: str,
) -> dict[str, dict]:
    if estimator != 'empirical_discrete':
        raise StructuredContractError('only empirical_discrete estimator is implemented')
    if not isinstance(missing_policy, str) or missing_policy not in MISSING_POLICIES:
        raise StructuredContractError('unsupported missing policy')
    if type(min_samples) is not int or min_samples < 2:
        raise StructuredContractError('min_samples must be an integer >= 2')
    if type(min_stratum_samples) is not int or min_stratum_samples < 1:
        raise StructuredContractError('min_stratum_samples must be an integer >= 1')
    if type(log_base) not in (float, int) or (isinstance(log_base, int) and log_base.bit_length() > 1023) or not math.isfinite(log_base) or log_base <= 1:
        raise StructuredContractError('log_base must be finite, positive and not equal to one')
    if bins is None:
        return {}
    if not isinstance(bins, Mapping) or any(name not in variables for name in bins):
        raise StructuredContractError('discretization names must be among the requested variables')
    result = {}
    for name, definition in bins.items():
        if not isinstance(definition, Mapping) or set(definition) != {'mode', 'bins'}:
            raise StructuredContractError(f'{name}: discretization requires mode and bins')
        mode, count = definition['mode'], definition['bins']
        if not isinstance(mode, str) or mode not in BINNING_MODES or type(count) is not int or not 2 <= count <= 128:
            raise StructuredContractError(f'{name}: invalid discretization mode or bin count')
        result[name] = {'mode': mode, 'bins': count}
    return result


def _numeric(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise StructuredContractError(f'{name}: discretization requires numeric observations, not booleans or strings')
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise StructuredContractError(f'{name}: value outside finite numeric range') from exc
    if not math.isfinite(number):
        raise StructuredContractError(f'{name}: value outside finite numeric range')
    return number


def _binning_definition(numbers: list[float], name: str, definition: dict) -> dict:
    if not numbers:
        return {'mode': definition['mode'], 'requested_bins': definition['bins'],
                'actual_bins': 0, 'cutpoints': [], 'bounds': None}
    low, high = min(numbers), max(numbers)
    if definition['mode'] == 'quantile':
        ordered = sorted(numbers)
        n = len(ordered)
        thresholds = sorted(set(ordered[min(n - 1, math.ceil(i * n / definition['bins']))]
                                for i in range(1, definition['bins'])))
        thresholds = [value for value in thresholds if low < value <= high]
        return {'mode': 'quantile', 'requested_bins': definition['bins'],
                'actual_bins': len(thresholds) + 1, 'cutpoints': thresholds, 'bounds': [low, high]}
    # Equal-width binning uses a normalized [0,1] position to avoid overflow
    # for large but individually finite endpoints.
    return {'mode': 'equal_width', 'requested_bins': definition['bins'],
            'actual_bins': 1 if low == high else definition['bins'],
            'cutpoints': [], 'bounds': [low, high]}


def _bin_index(value: float, specification: dict) -> int:
    if specification['mode'] == 'quantile':
        return bisect_right(specification['cutpoints'], value)
    low, high = specification['bounds']
    if high == low:
        return 0
    # Use direct normalization where finite (including subnormal spans).
    # Halve the endpoints only if a full-width subtraction would overflow.
    span = high - low
    if math.isfinite(span) and span > 0:
        position = (value - low) / span
    else:
        position = (value / 2 - low / 2) / (high / 2 - low / 2)
    if not math.isfinite(position):
        raise StructuredContractError('equal-width discretization exceeds numerical range')
    return min(specification['actual_bins'] - 1,
               max(0, math.floor(position * specification['actual_bins'])))


def _categorical(value: object, name: str) -> tuple[str, object]:
    if type(value) is bool:
        return 'boolean', value
    if type(value) is int:
        return 'integer', value
    if type(value) is str:
        return 'string', value
    if type(value) is float:
        raise StructuredContractError(
            f'{name}: floating-point observations require explicit discretization')
    raise StructuredContractError(
        f'{name}: information estimator requires scalar categorical values or numeric discretization')


def _entropy(counts: Counter, denominator: int, base: float) -> float:
    if not denominator:
        return 0.0
    h = -math.fsum((count / denominator) * math.log(count / denominator, base)
                   for count in counts.values() if count)
    return max(0.0, h)


def _conditional_entropy(pairs: list[tuple[tuple, tuple]], base: float) -> float:
    n = len(pairs)
    if n == 0:
        return 0.0
    joint = Counter(pairs)
    strata = Counter(y for _, y in pairs)
    return max(0.0, math.fsum((count / n) * math.log(strata[y] / count, base)
                              for (_, y), count in joint.items()))


def _conditional_mi(rows: list[tuple[tuple, tuple, tuple]], base: float) -> float:
    n = len(rows)
    if n == 0:
        return 0.0
    xyz = Counter(rows)
    xz = Counter((x, z) for x, y, z in rows)
    yz = Counter((y, z) for x, y, z in rows)
    z_counts = Counter(z for x, y, z in rows)
    result = math.fsum((count / n) * math.log(
        (count / xz[x, z]) * (z_counts[z] / yz[y, z]), base)
        for (x, y, z), count in xyz.items())
    # Round-off cancellation can produce a tiny negative on independent data.
    if result < -1e-10:
        raise StructuredContractError('information estimate violated nonnegativity')
    return max(0.0, result)


def estimate_information(
    obj: StructuredObject, *, measure: str, x: str, y: str | None = None,
    given: Sequence[str] = (), bins: Mapping[str, Mapping[str, Any]] | None = None,
    estimator: str = 'empirical_discrete', missing_policy: str = 'complete_case',
    min_samples: int = 2, min_stratum_samples: int = 2,
    log_base: float = 2.0,
) -> dict[str, Any]:
    """Estimate empirical information from *paired observation rows* only.

    ``bins`` is a per-variable mapping, e.g. ``{'temperature':
    {'mode': 'quantile', 'bins': 4}}``. Floats require this explicit policy;
    integers, strings and booleans are treated as discrete categories unless
    configured otherwise. Missing states are excluded (complete_case), encoded
    as distinct generic states (as_category), or rejected (error). Multiple
    cells are unsupported; callers must align/aggregate them explicitly.

    Estimates are descriptive plug-in estimators without bias correction,
    independence guarantees, p-values, or confidence intervals.
    """
    if not isinstance(obj, StructuredObject):
        raise StructuredContractError('information estimates require a validated StructuredObject')
    variables = _names(x, y, given, measure)
    discretizers = _parameters(
        bins=bins, variables=variables, min_samples=min_samples,
        min_stratum_samples=min_stratum_samples, log_base=log_base,
        missing_policy=missing_policy, estimator=estimator)
    obj.require('conditional_information')
    obj.require_aligned(variables)

    included: list[dict[str, Mapping[str, Any]]] = []
    dropped: Counter[str] = Counter()
    observed_states = {name: Counter() for name in variables}
    reason_counts = {name: Counter() for name in variables}
    for row in obj.observations:
        values = row['values']
        incomplete = []
        for name in variables:
            cell = values[name]
            state = cell['status']
            observed_states[name][state] += 1
            if state in {'missing', 'explicit_null', 'unknown'}:
                reason_counts[name][cell.get('source_reason', 'unspecified')] += 1
                incomplete.append(name)
            elif state == 'multiple':
                raise StructuredContractError(
                    f'{name}: repeated observations require explicit alignment or aggregation')
        if incomplete and missing_policy == 'error':
            raise StructuredContractError('missing value encountered under error policy')
        if incomplete and missing_policy == 'complete_case':
            dropped['incomplete_row'] += 1
            continue
        included.append(values)

    specifications = {}
    for name, definition in discretizers.items():
        numbers = [_numeric(row[name]['value'], name) for row in included
                   if row[name]['status'] == 'present']
        # Empty strings aren't numbers; their state is a separate category.
        specifications[name] = _binning_definition(numbers, name, definition)

    encoded = []
    for values in included:
        row = {}
        for name in variables:
            cell = values[name]
            state = cell['status']
            if state in {'missing', 'explicit_null', 'unknown', 'empty'}:
                row[name] = ('state', state)
            elif name in specifications:
                row[name] = ('bin', _bin_index(_numeric(cell['value'], name), specifications[name]))
            else:
                row[name] = _categorical(cell['value'], name)
        encoded.append(row)

    n = len(encoded)
    strata_names = tuple(given)
    if measure == 'conditional_entropy':
        strata_names = tuple(given)
    if measure == 'mutual_information':
        strata_names = ()
    strata = Counter(tuple(row[name] for name in strata_names) for row in encoded)
    insufficient_strata = sum(1 for count in strata.values() if count < min_stratum_samples)
    is_conditional = bool(strata_names)
    sufficient = n >= min_samples and (not is_conditional or not insufficient_strata)
    status = 'ok' if sufficient else ('insufficient_samples' if n < min_samples
                                      else 'insufficient_conditioning_support')
    estimate: float | None = None
    if sufficient:
        if measure == 'entropy':
            estimate = _entropy(Counter(row[x] for row in encoded), n, log_base)
        elif measure == 'conditional_entropy':
            estimate = _conditional_entropy(
                [(row[x], tuple(row[name] for name in given)) for row in encoded], log_base)
        else:
            estimate = _conditional_mi([
                (row[x], row[y], tuple(row[name] for name in given)) for row in encoded
            ], log_base)
    if estimate is not None and not math.isfinite(estimate):
        raise StructuredContractError('information computation exceeded numerical range')
    n_states = {name: len(set(row[name] for row in encoded)) for name in variables}
    warnings = [
        'Empirical discrete plug-in estimate is generally biased at finite sample sizes.',
        'Source edges are not probabilistic dependencies; rows may be dependent or clustered.',
        'Discretization and complete-case selection can change the estimand.',
    ]
    if n and any(count == 1 for count in Counter(tuple(row[name] for name in variables)
                                                 for row in encoded).values()):
        warnings.append('At least one joint state has singleton support; estimate may be sparse.')
    if is_conditional and insufficient_strata:
        warnings.append('Conditioning strata below min_stratum_samples; no estimate reported.')
    return {
        'schema': INFORMATION_SCHEMA, 'object_id': obj.object_id,
        'observation_unit': obj.observation_unit, 'measure': measure,
        'variables': {'x': x, 'y': y, 'given': list(given)},
        'estimator': estimator, 'missing_policy': missing_policy,
        'discretization': specifications,
        'log_base': log_base, 'value': estimate, 'status': status,
        'support': {
            'aligned_rows': len(obj.observations), 'included_rows': n,
            'excluded_rows': len(obj.observations) - n,
            'exclusion_reasons': dict(sorted(dropped.items())),
            'min_samples': min_samples, 'min_stratum_samples': min_stratum_samples,
            'conditioning_strata': len(strata) if is_conditional else None,
            'under_supported_strata': insufficient_strata if is_conditional else None,
            'distinct_states': n_states,
            'cell_status_counts': {name: dict(sorted(counts.items())) for name, counts in observed_states.items()},
            'null_source_reasons': {name: dict(sorted(counts.items())) for name, counts in reason_counts.items()},
        },
        'warnings': warnings,
    }


def estimate_entropy(obj: StructuredObject, variable: str, **options: Any) -> dict[str, Any]:
    return estimate_information(obj, measure='entropy', x=variable, **options)


def estimate_conditional_entropy(obj: StructuredObject, target: str,
                                 given: Sequence[str], **options: Any) -> dict[str, Any]:
    return estimate_information(obj, measure='conditional_entropy', x=target,
                                given=given, **options)


def estimate_mutual_information(obj: StructuredObject, x: str, y: str,
                                **options: Any) -> dict[str, Any]:
    return estimate_information(obj, measure='mutual_information', x=x, y=y, **options)


def estimate_conditional_mutual_information(obj: StructuredObject, x: str, y: str,
                                            given: Sequence[str], **options: Any) -> dict[str, Any]:
    return estimate_information(obj, measure='conditional_mutual_information', x=x,
                                y=y, given=given, **options)
