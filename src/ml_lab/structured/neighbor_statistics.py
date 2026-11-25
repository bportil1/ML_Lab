"""HIA-05: opt-in, source-pinned observed relationships for structural neighbors.

Only original, explicitly mapped observation columns are compared. Structural
adjacency determines *which* candidates are offered, never the existence or
strength of statistical dependence. Calculations use complete, aligned rows;
source nodes/edges are not independent samples.
"""
from __future__ import annotations

from collections import Counter
import math
from typing import Any

from .bindings import ObservationBindingPlan, VariableBinding, bind_observations, source_sha256
from .branch_uncertainty import binding_sha256
from .contracts import StructuredContractError
from .hierarchical_index import build_hierarchy_index
from .information import (_bin_index, _binning_definition, _categorical,
                          _conditional_entropy, _conditional_mi, _entropy,
                          _numeric, _parameters)
from .object import StructuredObject

NEIGHBOR_SCHEMA = 'ml-lab.hia-neighbor-relationships@1'


def _mapping(raw: Any, label: str) -> dict[str, str]:
    if not isinstance(raw, dict) or set(raw) != {'node_id', 'variable_id'}:
        raise StructuredContractError(f'{label} requires exact node_id and variable_id')
    if any(not isinstance(value, str) or not value.strip() for value in raw.values()):
        raise StructuredContractError(f'{label} requires nonempty original node and observation column IDs')
    return raw


def _ranks(values: list[float]) -> list[float]:
    # Average mid-ranks for ties; no SciPy dependency or inferred significance.
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        midrank = (start + 1 + end) / 2
        for pos in range(start, end):
            ranks[order[pos]] = midrank
        start = end
    return ranks


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    # Normalize by max absolute magnitude to avoid overflow in covariance.
    scale_x, scale_y = max(map(abs, x)), max(map(abs, y))
    if not scale_x or not scale_y:
        return None
    a, b = [v / scale_x for v in x], [v / scale_y for v in y]
    mean_x, mean_y = math.fsum(a) / len(a), math.fsum(b) / len(b)
    dx = [v - mean_x for v in a]
    dy = [v - mean_y for v in b]
    xx, yy = math.fsum(v * v for v in dx), math.fsum(v * v for v in dy)
    if xx <= 0 or yy <= 0:
        return None
    return max(-1.0, min(1.0, math.fsum(u * v for u, v in zip(dx, dy)) / math.sqrt(xx * yy)))


def _correlations(rows: list[dict], x: str, y: str, minimum: int) -> dict[str, Any]:
    numbers: list[tuple[float, float]] = []
    nonnumeric = 0
    for row in rows:
        left, right = row['values'][x], row['values'][y]
        if left['status'] not in ('present', 'empty') or right['status'] not in ('present', 'empty'):
            continue
        if left['status'] != 'present' or right['status'] != 'present' or type(left['value']) not in (int, float) or type(right['value']) not in (int, float):
            nonnumeric += 1
            continue
        numbers.append((_numeric(left['value'], x), _numeric(right['value'], y)))
    out: dict[str, Any] = {'support': len(numbers), 'excluded_non_numeric_complete_pairs': nonnumeric,
                           'pearson': None, 'spearman': None, 'status': 'ok'}
    if len(numbers) < minimum:
        out['status'] = 'insufficient_numeric_pairs'
        return out
    a, b = [item[0] for item in numbers], [item[1] for item in numbers]
    out['pearson'] = _pearson(a, b)
    out['spearman'] = _pearson(_ranks(a), _ranks(b))
    if out['pearson'] is None or out['spearman'] is None:
        out['status'] = 'constant_or_undefined_variance'
    return out


def _encode(cell: dict, name: str, fitted: dict[str, dict]) -> tuple:
    if cell['status'] == 'empty':
        return ('state', 'empty')
    if name in fitted:
        return ('bin', _bin_index(_numeric(cell['value'], name), fitted[name]))
    return _categorical(cell['value'], name)


def _joint(rows: list[dict], variables: tuple[str, ...], fitted: dict[str, dict],
           minimum: int, min_stratum: int, conditioned: bool) -> dict[str, Any]:
    selected: list[dict] = []
    unavailable: Counter[str] = Counter()
    for row in rows:
        values = row['values']
        statuses = [values[name]['status'] for name in variables]
        if 'multiple' in statuses:
            unavailable['unresolved_repeated_values'] += 1
            continue
        if any(status not in ('present', 'empty') for status in statuses):
            unavailable['incomplete_case'] += 1
            continue
        try:
            selected.append({name: _encode(values[name], name, fitted) for name in variables})
        except StructuredContractError as exc:
            # Unsupported encodings (not an estimator estimate) remain explicit.
            message = str(exc)
            if 'require explicit discretization' in message:
                unavailable['float_requires_explicit_binning'] += 1
            elif 'numeric' in message or 'discretization' in message:
                unavailable['binning_requires_numeric_values'] += 1
            else:
                unavailable['unsupported_non_scalar_values'] += 1
    reasons = ('unresolved_repeated_values', 'float_requires_explicit_binning',
               'binning_requires_numeric_values', 'unsupported_non_scalar_values')
    invalid = next((reason for reason in reasons if unavailable[reason]), None)
    result: dict[str, Any] = {
        'status': invalid or ('insufficient_samples' if len(selected) < minimum else 'ok'),
        'support': len(selected), 'excluded_rows': len(rows) - len(selected),
        'excluded_by_reason': dict(sorted(unavailable.items())),
        'conditional_strata': None, 'under_supported_strata': None,
        'mi_bits': None, 'h_x_given_y_bits': None, 'h_y_given_x_bits': None,
    }
    if result['status'] != 'ok':
        return result
    if conditioned:
        strata = Counter(tuple(record[z] for z in variables[2:]) for record in selected)
        short = sum(count < min_stratum for count in strata.values())
        result['conditional_strata'] = len(strata)
        result['under_supported_strata'] = short
        if short:
            result['status'] = 'insufficient_conditioning_support'
            return result
    x, y = variables[:2]
    if conditioned:
        samples = [(record[x], record[y], tuple(record[z] for z in variables[2:])) for record in selected]
        result['mi_bits'] = _conditional_mi(samples, 2.0)
        result['h_x_given_y_bits'] = _conditional_entropy([(record[x], (record[y], *(record[z] for z in variables[2:]))) for record in selected], 2.0)
        result['h_y_given_x_bits'] = _conditional_entropy([(record[y], (record[x], *(record[z] for z in variables[2:]))) for record in selected], 2.0)
    else:
        samples = [(record[x], record[y], ()) for record in selected]
        result['mi_bits'] = _conditional_mi(samples, 2.0)
        result['h_x_given_y_bits'] = _conditional_entropy([(record[x], record[y]) for record in selected], 2.0)
        result['h_y_given_x_bits'] = _conditional_entropy([(record[y], record[x]) for record in selected], 2.0)
    if len({(record[x], record[y], *(record[z] for z in variables[2:])) for record in selected}) * 5 >= len(selected):
        result['warning'] = 'sparse_joint_states_relative_to_support'
    return result


def analyze_neighbor_relationships(
    source: StructuredObject, *, node_id: str, variable_id: str,
    neighbors: list[dict[str, str]], conditioning: list[dict[str, str]] | None = None,
    occurrence_policy: str = 'preserve', bins: dict[str, dict[str, Any]] | None = None,
    min_samples: int = 2, min_stratum_samples: int = 2,
) -> dict[str, Any]:
    """Compare declared original structural neighbors using aligned rows only.

    Statistical edges are never asserted by this API. Each neighbor is paired
    independently with X (different pairwise support may result). A requested
    conditioner is explicit and can be nonadjacent; it must not reuse X/Y.
    """
    if not isinstance(source, StructuredObject):
        raise StructuredContractError('HIA-05 requires a validated StructuredObject')
    index = build_hierarchy_index(source)
    context = index.node_context(node_id)
    if not isinstance(neighbors, list) or not neighbors:
        raise StructuredContractError('select at least one explicitly mapped structural neighbor')
    if not isinstance(conditioning, list) and conditioning is not None:
        raise StructuredContractError('conditioning must be an explicit array')
    conditioning = conditioning or []
    if len(conditioning) > 8:
        raise StructuredContractError('conditioning is limited to 8 explicitly selected columns per estimate')
    if len(neighbors) + len(conditioning) > 1999:
        raise StructuredContractError('too many variables for the SU-05 binding contract')
    central = _mapping({'node_id': node_id, 'variable_id': variable_id}, 'selected node')
    neighbor_map = [_mapping(value, 'neighbor') for value in neighbors]
    conditioners = [_mapping(value, 'conditioner') for value in conditioning]
    candidate_ids = set(context['parent_ids']) | set(context['child_ids'])
    if any(value['node_id'] not in candidate_ids for value in neighbor_map):
        raise StructuredContractError('a selected neighbor is not directly adjacent in original source topology')
    assignments = [central, *neighbor_map, *conditioners]
    if (len({entry['node_id'] for entry in assignments}) != len(assignments) or
            len({entry['variable_id'] for entry in assignments}) != len(assignments)):
        raise StructuredContractError('each selected node and observation column must be mapped one-to-one')
    rules = tuple(VariableBinding(entry['variable_id'], (entry['node_id'],), occurrence_policy)
                  for entry in assignments)
    plan = ObservationBindingPlan(source_sha256(source), source.observation_unit, rules)
    bound = bind_observations(source, plan)
    rows = bound.rows
    variables = tuple(entry['variable_id'] for entry in assignments)
    bins = {} if bins is None else bins
    _parameters(bins=bins, variables=variables, min_samples=min_samples,
                min_stratum_samples=min_stratum_samples, log_base=2.0,
                missing_policy='complete_case', estimator='empirical_discrete')
    fitted = {}
    for name, definition in bins.items():
        numbers = [_numeric(row['values'][name]['value'], name) for row in rows
                   if row['values'][name]['status'] == 'present' and type(row['values'][name]['value']) in (int, float)]
        if any(row['values'][name]['status'] == 'present' and type(row['values'][name]['value']) not in (int, float) for row in rows):
            raise StructuredContractError(f'{name}: explicit numeric binning requires numeric observed values')
        fitted[name] = _binning_definition(numbers, name, definition)
    x = variable_id
    z = tuple(entry['variable_id'] for entry in conditioners)
    relationships = []
    for item in neighbor_map:
        y = item['variable_id']
        ordinary = _joint(rows, (x, y), fitted, min_samples, min_stratum_samples, False)
        conditional = _joint(rows, (x, y, *z), fitted, min_samples, min_stratum_samples, True) if z else None
        edges = [{'edge_id': edge['id'], 'source_id': edge['source_id'],
                  'target_id': edge['target_id'], 'kind': edge['kind']}
                 for edge in source.edges if
                 (edge['source_id'] == node_id and edge['target_id'] == item['node_id']) or
                 (edge['source_id'] == item['node_id'] and edge['target_id'] == node_id)]
        relationships.append({
            'node_id': item['node_id'], 'variable_id': y,
            'structural_edges': edges, 'structural_edges_are_not_statistical_dependence': True,
            'mutual_information': ordinary, 'conditional_mutual_information': conditional,
            'correlations': _correlations(rows, x, y, min_samples),
        })
    return {
        'schema': NEIGHBOR_SCHEMA, 'source_digest': source_sha256(source),
        'binding_digest': binding_sha256(bound), 'binding_plan': plan.to_dict(),
        'node_id': node_id, 'variable_id': variable_id, 'structural_context': context,
        'observation_unit': source.observation_unit, 'aligned_rows': len(rows),
        'occurrence_policy': occurrence_policy, 'discretization': fitted,
        'conditioning': conditioners, 'relationships': relationships,
        'scientific_cautions': [
            'Source edges select candidates but do not imply statistical or causal dependence.',
            'MI/CMI are empirical discrete plug-in estimates, generally biased in finite samples.',
            'Missing or repeated values alter complete-case support; no imputation or silent aggregation.',
            'Pearson and Spearman apply only to aligned numeric pairs, with no p-values or inference.',
            'Rows may be dependent; observation counts are not independent-sample guarantees.',
        ],
    }
