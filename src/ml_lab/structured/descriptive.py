"""ML-UW-31: domain-neutral structural and aligned-observation descriptions.

Structural nodes are NOT independent samples. Occurrence rates require an
explicit common identity policy over multiple independently supplied objects.
Numeric summaries and nested-group decompositions require actual, aligned
observation rows. No domain adapters, estimators, databases, filesystem IO,
or inferred probabilistic dependencies are used here.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import json
import math
from statistics import fmean
from typing import Any

from .contracts import StructuredContractError
from .object import StructuredObject

STRUCTURE_SCHEMA = 'ml-lab.descriptive-structure@1'
COLLECTION_SCHEMA = 'ml-lab.descriptive-collection@1'
OBSERVATION_SCHEMA = 'ml-lab.descriptive-observations@1'
VARIANCE_SCHEMA = 'ml-lab.descriptive-nested-variance@1'


def _object(obj: StructuredObject) -> StructuredObject:
    if not isinstance(obj, StructuredObject):
        raise StructuredContractError('descriptive analyses require a validated StructuredObject')
    return obj


def _sum_finite(terms, where: str) -> float:
    try:
        result = math.fsum(terms)
    except (OverflowError, ValueError) as exc:
        raise StructuredContractError(f'{where}: numerical range exceeded') from exc
    if not math.isfinite(result):
        raise StructuredContractError(f'{where}: numerical range exceeded')
    return result


def _square_difference(value: float, mean: float) -> float:
    delta = value - mean
    square = delta * delta
    if not math.isfinite(square):
        raise StructuredContractError('variance: numerical range exceeded')
    return square


def _stats(numbers: Sequence[int | float]) -> dict[str, Any]:
    """Descriptive, not inferential; small-sample variance stays unavailable."""
    data = list(numbers)
    if not data:
        return {'count': 0, 'mean': None, 'min': None, 'max': None,
                'sample_variance': None, 'variance_support': 'no_observations'}
    try:
        mu = fmean(data)
    except (OverflowError, ValueError) as exc:
        raise StructuredContractError('mean: numerical range exceeded') from exc
    if not math.isfinite(mu):
        raise StructuredContractError('mean: numerical range exceeded')
    variance = (_sum_finite((_square_difference(x, mu) for x in data), 'variance')
                / (len(data) - 1) if len(data) >= 2 else None)
    if variance is not None and not math.isfinite(variance):
        raise StructuredContractError('variance: numerical range exceeded')
    return {
        'count': len(data), 'mean': mu, 'min': min(data), 'max': max(data),
        'sample_variance': variance,
        'variance_support': 'sufficient' if len(data) >= 2 else 'insufficient',
    }


def describe_structure(obj: StructuredObject) -> dict[str, Any]:
    """Describe a single observed graph; never interpret counts as probabilities."""
    obj = _object(obj)
    nodes, edges = obj.nodes, obj.edges
    directed = 'directed_edges' in obj.capabilities
    node_ids = [node['id'] for node in nodes]
    children: dict[str, list[str]] = {key: [] for key in node_ids}
    parent_counts = dict.fromkeys(node_ids, 0)
    kinds = Counter(node['kind'] for node in nodes)
    edge_kinds = Counter(edge['kind'] for edge in edges)
    for edge in edges:
        source, target = edge['source_id'], edge['target_id']
        children[source].append(target)
        parent_counts[target] += 1
        if not directed:
            children[target].append(source)
            parent_counts[source] += 1
    out_degree = [len(children[node]) for node in node_ids]
    result: dict[str, Any] = {
        'schema': STRUCTURE_SCHEMA,
        'object_id': obj.object_id,
        'topology': obj.topology['kind'],
        'observation_unit': obj.observation_unit,
        'support': {'objects': 1, 'nodes': len(nodes), 'edges': len(edges),
                    'aligned_observation_rows': len(obj.observations)},
        'node_kinds': dict(sorted(kinds.items())),
        'edge_kinds': dict(sorted(edge_kinds.items())),
        'degree': {
            'out' if directed else 'undirected': _stats(out_degree),
            **({'in': _stats(list(parent_counts.values()))} if directed else {}),
        },
        'hierarchy': None,
        'limitations': [
            'Counts describe one object; no node/path occurrence probability is estimated.',
            'Node and edge counts are not observation sample sizes.',
        ],
    }
    if 'rooted_hierarchy' in obj.capabilities:
        obj.require('hierarchy_summary')
        roots = sorted(name for name, count in parent_counts.items() if count == 0)
        depths = {root: 0 for root in roots}
        queue = list(roots)
        for name in queue:
            for child in children[name]:
                depths[child] = depths[name] + 1
                queue.append(child)
        child_counts = [len(children[name]) for name in node_ids]
        result['hierarchy'] = {
            'root_ids': roots,
            'root_count': len(roots),
            'leaves': sum(count == 0 for count in child_counts),
            'maximum_depth': max(depths.values()),
            'depth_counts': dict(sorted(Counter(depths.values()).items())),
            'child_cardinality': _stats(child_counts),
            'child_count_distribution': dict(sorted(Counter(child_counts).items())),
            'cardinality_unit': 'outgoing structural edges per parent node, including leaves',
        }
    return result


def _identity(node_or_edge: Mapping[str, Any], selector: str) -> str:
    if selector == 'id':
        return node_or_edge['id']
    if selector.startswith('attributes:') or selector.startswith('metadata:'):
        scope, key = selector.split(':', 1)
        if not key:
            raise StructuredContractError('identity selector requires an attribute key')
        value = node_or_edge.get(scope, {}).get(key)
        if not isinstance(value, str) or not value.strip():
            raise StructuredContractError(f'identity {selector!r} missing or not a string')
        return value
    raise StructuredContractError('identity must be id, attributes:<key>, or metadata:<key>')


def describe_collection(
    objects: Sequence[StructuredObject], *, node_identity: str,
    edge_identity: str,
) -> dict[str, Any]:
    """Empirical presence across explicitly aligned objects, NOT a population CI.

    Callers must assert that identical extracted identity keys denote the same
    scientific variable/relationship across objects; no fuzzy matching occurs.
    Duplicate logical identities within one object contribute one *presence*
    but all their multiplicities are separately reported.
    """
    if not isinstance(objects, (list, tuple)) or not objects:
        raise StructuredContractError('collection must contain one or more objects')
    if not isinstance(node_identity, str) or not isinstance(edge_identity, str):
        raise StructuredContractError('node and edge identity selectors must be strings')
    # Check selector syntax even on empty-edge objects.
    for name in (node_identity, edge_identity):
        if name != 'id' and not ((name.startswith('attributes:') or name.startswith('metadata:'))
                                 and name.partition(':')[2]):
            raise StructuredContractError('identity must be id, attributes:<key>, or metadata:<key>')
    node_presence: Counter[str] = Counter()
    node_multiplicity: Counter[str] = Counter()
    edge_presence: Counter[str] = Counter()
    edge_multiplicity: Counter[str] = Counter()
    ids: list[str] = []
    for obj in objects:
        obj = _object(obj)
        ids.append(obj.object_id)
        n_counts = Counter(_identity(n, node_identity) for n in obj.nodes)
        e_counts = Counter(_identity(e, edge_identity) for e in obj.edges)
        node_presence.update(n_counts.keys())
        node_multiplicity.update(n_counts)
        edge_presence.update(e_counts.keys())
        edge_multiplicity.update(e_counts)
    n = len(objects)
    def rows(presence: Counter[str], multiplicities: Counter[str]) -> list[dict]:
        return [
            {'identity': key, 'present_objects': count, 'total_objects': n,
             'absent_objects': n - count, 'empirical_presence_rate': count / n,
             'total_occurrences': multiplicities[key]}
            for key, count in sorted(presence.items())
        ]
    return {
        'schema': COLLECTION_SCHEMA,
        'object_ids': ids,
        'support': {'objects': n, 'node_identities': len(node_presence),
                    'edge_identities': len(edge_presence)},
        'identity_policy': {'nodes': node_identity, 'edges': edge_identity,
                            'caller_asserts_cross_object_equivalence': True},
        'node_occurrence': rows(node_presence, node_multiplicity),
        'edge_occurrence': rows(edge_presence, edge_multiplicity),
        'limitations': [
            'Rates describe empirical object coverage, not uncertainty intervals or independence.',
            'Identity equivalence is explicitly asserted by the caller, not inferred.',
            'Repeated equal-content objects remain separate supplied observations.',
        ],
    }


def _names(variables: Sequence[str], where: str) -> tuple[str, ...]:
    if not isinstance(variables, (list, tuple)) or not variables:
        raise StructuredContractError(f'{where} requires nonempty variable names')
    if any(not isinstance(name, str) or not name.strip() for name in variables):
        raise StructuredContractError(f'{where} contains invalid names')
    if len(set(variables)) != len(variables):
        raise StructuredContractError(f'{where} requires unique names')
    return tuple(variables)


def _numeric(cell: Mapping[str, Any], variable: str) -> float | None:
    if cell['status'] != 'present':
        return None
    value = cell['value']
    if type(value) not in (int, float):
        raise StructuredContractError(f'{variable}: numeric analysis requires finite numeric observed values')
    try:
        numeric = float(value)
    except (ValueError, OverflowError) as exc:
        raise StructuredContractError(f'{variable}: cannot represent value as finite float') from exc
    if not math.isfinite(numeric):
        raise StructuredContractError(f'{variable}: numeric analysis requires finite values')
    return numeric


def describe_observations(
    obj: StructuredObject, *, variables: Sequence[str],
    numeric_variables: Sequence[str] = (),
) -> dict[str, Any]:
    """Count statuses per paired *observation*, not per element/node/occurrence."""
    obj = _object(obj)
    variables = _names(variables, 'variables')
    numeric_variables = tuple(numeric_variables)
    if any(name not in variables for name in numeric_variables):
        raise StructuredContractError('numeric variables must be included in requested variables')
    if len(set(numeric_variables)) != len(numeric_variables):
        raise StructuredContractError('numeric variables must be unique')
    obj.require_aligned(variables)
    rows = obj.observations
    result: dict[str, Any] = {
        'schema': OBSERVATION_SCHEMA,
        'object_id': obj.object_id,
        'observation_unit': obj.observation_unit,
        'support': {'aligned_rows': len(rows), 'node_count_not_used_as_sample_size': len(obj.nodes)},
        'variables': {},
        'limitations': [
            'Structural edges never create independent sample rows.',
            'Multiple-valued cells are counted as one record but never silently averaged.',
            'Sample variance requires at least two observed numeric values.',
        ],
    }
    for name in variables:
        counts = Counter(row['values'][name]['status'] for row in rows)
        reason_counts = Counter(
            row['values'][name].get('source_reason', 'unspecified')
            for row in rows if row['values'][name]['status'] in ('missing', 'explicit_null', 'unknown')
        )
        occurrence_counts = Counter(
            subcell['status']
            for row in rows if row['values'][name]['status'] == 'multiple'
            for subcell in row['values'][name]['occurrences']
        )
        nested_reasons = Counter(
            subcell.get('source_reason', 'unspecified')
            for row in rows if row['values'][name]['status'] == 'multiple'
            for subcell in row['values'][name]['occurrences']
            if subcell['status'] in ('missing', 'explicit_null', 'unknown')
        )
        observed = counts['present'] + counts['empty']
        values = [_numeric(row['values'][name], name) for row in rows] if name in numeric_variables else []
        numeric_values = [value for value in values if value is not None]
        entry: dict[str, Any] = {
            'support': {'rows': len(rows), 'observed_rows': observed,
                        'present_rows': counts['present'], 'empty_rows': counts['empty'],
                        'unavailable_rows': len(rows) - observed},
            'status_counts': dict(sorted(counts.items())),
            'status_rates': {key: count / len(rows) for key, count in sorted(counts.items())},
            'null_source_reasons': dict(sorted(reason_counts.items())),
            'repeated_occurrence_status_counts': dict(sorted(occurrence_counts.items())),
            'repeated_occurrence_null_reasons': dict(sorted(nested_reasons.items())),
            'numeric': _stats(numeric_values) if name in numeric_variables else None,
        }
        if name in numeric_variables:
            entry['support']['numeric_rows'] = len(numeric_values)
            entry['support']['excluded_from_numeric'] = len(rows) - len(numeric_values)
        result['variables'][name] = entry
    return result


def _group_key(cell: Mapping[str, Any], name: str) -> str | None:
    # Empty is an observed category; null/unknown/multiple are not usable labels.
    if cell['status'] not in ('present', 'empty'):
        return None
    value = cell['value']
    if type(value) not in (str, bool, int, float):
        raise StructuredContractError(f'{name}: grouping values must be scalar categories')
    return json.dumps([type(value).__name__, value], allow_nan=False,
                      ensure_ascii=False, separators=(',', ':'))


def describe_nested_variance(
    obj: StructuredObject, *, value_variable: str, group_levels: Sequence[str],
) -> dict[str, Any]:
    """Exact descriptive SS decomposition for explicit, nested grouping columns.

    For nested keys (g1, ..., gk), SS_total = SS_g1 + SS_g2|g1 + ...
    + SS_within_leaf. This is observational, not random-effects estimation,
    a causal result, or an inference about the source graph's parent edges.
    """
    obj = _object(obj)
    groups = _names(group_levels, 'group_levels')
    if value_variable in groups:
        raise StructuredContractError('outcome and grouping variable must be different')
    obj.require_aligned((value_variable, *groups))
    complete: list[tuple[tuple[str, ...], float]] = []
    exclusions: Counter[str] = Counter()
    numeric_rows = 0
    for row in obj.observations:
        values = row['values']
        y = _numeric(values[value_variable], value_variable)
        if y is None:
            exclusions[f'outcome_{values[value_variable]["status"]}'] += 1
            continue
        numeric_rows += 1
        keys = tuple(_group_key(values[name], name) for name in groups)
        if any(key is None for key in keys):
            for name, key in zip(groups, keys):
                if key is None:
                    exclusions[f'group_{name}_{values[name]["status"]}'] += 1
                    break
            continue
        complete.append((keys, y))
    n = len(complete)
    result: dict[str, Any] = {
        'schema': VARIANCE_SCHEMA,
        'object_id': obj.object_id,
        'observation_unit': obj.observation_unit,
        'value_variable': value_variable,
        'group_levels': list(groups),
        'support': {'total_aligned_rows': len(obj.observations),
                    'numeric_outcome_rows': numeric_rows, 'complete_case_rows': n,
                    'excluded_rows': len(obj.observations) - n,
                    'exclusion_reasons': dict(sorted(exclusions.items()))},
        'outcome': _stats([v for _, v in complete]),
        'components': None,
        'status': 'sufficient' if n >= 2 else 'insufficient',
        'limitations': [
            'Components are descriptive sums of squares, not fitted random effects or causal effects.',
            'Nested observation-group labels are explicit columns; graph edges are not grouping assumptions.',
            'Missing group labels and non-observed outcomes are excluded, with counts reported.',
            'Sampling dependence, confidence intervals and predictive utility are not estimated.',
        ],
    }
    if n < 2:
        return result
    overall = fmean([v for _, v in complete])
    total_ss = _sum_finite((_square_difference(value, overall) for _, value in complete),
                           'total sum of squares')
    previous = {(): (n, overall)}
    components = []
    for index, level in enumerate(groups, start=1):
        buckets: defaultdict[tuple[str, ...], list[float]] = defaultdict(list)
        for key, value in complete:
            buckets[key[:index]].append(value)
        means = {key: (len(values), fmean(values)) for key, values in buckets.items()}
        ss = _sum_finite((count * _square_difference(mean, previous[key[:-1]][1])
                          for key, (count, mean) in means.items()), 'between-group sum of squares')
        components.append({'level': level, 'level_index': index,
                           'group_count': len(means), 'sum_squares': ss,
                           'proportion_total_ss': ss / total_ss if total_ss > 0 else None,
                           'support': {'complete_case_rows': n},
                           'group_sizes': _stats([len(values) for values in buckets.values()])})
        previous = means
    within = _sum_finite((_square_difference(value, previous[key][1])
                          for key, value in complete), 'within-group sum of squares')
    components.append({'level': 'within_leaf_groups', 'level_index': len(groups) + 1,
                       'group_count': len(previous), 'sum_squares': within,
                       'proportion_total_ss': within / total_ss if total_ss > 0 else None,
                       'support': {'complete_case_rows': n}})
    # Floating-point roundoff is allowed, but never silently return a broken partition.
    if not math.isclose(math.fsum(part['sum_squares'] for part in components), total_ss,
                        rel_tol=1e-9, abs_tol=1e-9):
        raise StructuredContractError('nested sum-of-squares partition is numerically unstable')
    result['total_sum_squares'] = total_ss
    result['components'] = components
    result['status'] = 'constant' if total_ss == 0 else 'sufficient'
    return result
