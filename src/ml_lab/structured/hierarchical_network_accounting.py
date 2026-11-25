"""HIA-10: matched-cohort information accounting across structural depths.

Each displayed result applies only to explicitly mapped observed variables.
All cumulative and cross-level comparisons use the *same* complete-case sample;
no pairwise MI totals are added, and topology is never treated as dependence.
"""
from __future__ import annotations

from typing import Any

from .bindings import ObservationBindingPlan, VariableBinding, bind_observations, source_sha256
from .branch_uncertainty import binding_sha256
from .contracts import StructuredContractError
from .hierarchical_branch_statistics import _entry, _joint_entropy, _rows
from .hierarchical_index import build_hierarchy_index
from .information import _binning_definition, _numeric, _parameters
from .object import StructuredObject

NETWORK_ACCOUNTING_SCHEMA = 'ml-lab.hia-network-information-accounting@1'


def analyze_hierarchical_network_accounting(
    source: StructuredObject, *, mappings: list[dict[str, str]],
    target: dict[str, str] | None = None, occurrence_policy: str = 'preserve',
    bins: dict[str, dict[str, Any]] | None = None, min_samples: int = 2,
) -> dict[str, Any]:
    """Return matched-cohort level increments, redundancy and target relevance.

    The caller must explicitly map every observed source node. Unmapped members
    are retained in inventories but never assigned made-up random variables.
    If Y is requested, its source node must be outside the mapped level set.
    All information figures then use the *single complete-case cohort X,Y*;
    no incompatible cohort subtraction is permitted.
    """
    if not isinstance(source, StructuredObject):
        raise StructuredContractError('HIA-10 requires a validated StructuredObject')
    index = build_hierarchy_index(source)
    if not isinstance(mappings, list) or not mappings:
        raise StructuredContractError('declare at least one mapped original source node')
    declared = [_entry(raw, 'network mapping') for raw in mappings]
    outside = _entry(target, 'target mapping') if target is not None else None
    assigned = [*declared, *([outside] if outside is not None else [])]
    if len(set(item['node_id'] for item in assigned)) != len(assigned):
        raise StructuredContractError('source node IDs must be distinct, including the target')
    if len(set(item['variable_id'] for item in assigned)) != len(assigned):
        raise StructuredContractError('observed columns must be distinct, including the target')
    for item in assigned:
        index.level_of(item['node_id'])
    ordered = sorted(declared, key=lambda m: (index.level_of(m['node_id']), m['node_id']))
    level_groups: dict[int, list[dict[str, str]]] = {}
    for item in ordered:
        level_groups.setdefault(index.level_of(item['node_id']), []).append(item)
    names = tuple(m['variable_id'] for m in [*ordered, *([outside] if outside else [])])
    definitions = _parameters(bins=bins, variables=names, min_samples=min_samples,
                              min_stratum_samples=1, log_base=2.0,
                              missing_policy='complete_case', estimator='empirical_discrete')
    plan = ObservationBindingPlan(source_sha256(source), source.observation_unit,
                                  tuple(VariableBinding(m['variable_id'], (m['node_id'],), occurrence_policy)
                                        for m in [*ordered, *([outside] if outside else [])]))
    bound = bind_observations(source, plan)
    fitted = {}
    for name, definition in definitions.items():
        values = [row['values'][name]['value'] for row in bound.rows
                  if row['values'][name]['status'] == 'present']
        if any(type(value) not in (int, float) for value in values):
            raise StructuredContractError(f'{name}: numeric discretization requires numeric observations')
        fitted[name] = _binning_definition([_numeric(value, name) for value in values], name, definition)
    cohort = _rows(bound.rows, names, fitted, min_samples)
    encoded = cohort.pop('_encoded')
    status = cohort['status']
    information: dict[str, Any] = {
        'status': status, 'support': cohort['support'],
        'excluded_rows': cohort['excluded_rows'],
        'excluded_by_reason': cohort['excluded_by_reason'],
        'joint_entropy_bits': None, 'total_correlation_bits': None,
        'target_mutual_information_bits': None, 'levels': [], 'cross_level_pairs': [],
        'finite_sample_bias_warning': True,
        'sparse_joint_states_relative_to_support': None,
    }
    n = len(ordered)
    if status == 'ok':
        x = [row[:n] for row in encoded]
        h_all = _joint_entropy(x)
        h_y = _joint_entropy([row[n:] for row in encoded]) if outside else None
        h_xy = _joint_entropy(encoded) if outside else None
        previous = 0
        prior_names: list[str] = []
        old_h = 0.0
        old_target_mi = 0.0
        sum_marginal = 0.0
        group_samples = {}
        group_entropy = {}
        for depth, members in level_groups.items():
            width = len(members)
            next_prefix = previous + width
            current = [row[previous:next_prefix] for row in encoded]
            prefix = [row[:next_prefix] for row in encoded]
            h_level = _joint_entropy(current)
            h_prefix = _joint_entropy(prefix)
            marginal_sum = sum(_joint_entropy([row[j:j+1] for row in encoded])
                               for j in range(previous, next_prefix))
            sum_marginal += marginal_sum
            contribution = max(0., h_prefix - old_h)
            explained = max(0., h_level - contribution)
            # Multi-information between the current level and ALL previous mapped levels.
            row_out = {
                'level': depth, 'mapped_node_ids': [m['node_id'] for m in members],
                'mapped_variable_ids': [m['variable_id'] for m in members],
                'structural_node_count': len(index.level_ids(depth)),
                'mapped_node_count': width,
                'unmapped_node_count': len(index.level_ids(depth)) - width,
                'level_joint_entropy_bits': h_level,
                'within_level_total_correlation_bits': max(0., marginal_sum-h_level),
                'cumulative_joint_entropy_bits': h_prefix,
                'incremental_joint_entropy_given_prior_bits': contribution,
                'information_shared_with_prior_bits': explained,
                'target_cumulative_mutual_information_bits': None,
                'target_incremental_information_bits': None,
            }
            if outside:
                # Common X,Y cohort: I(Y;current|previous)=I(Y;prefix)-I(Y;previous).
                h_prefix_y = _joint_entropy([row[:next_prefix]+row[n:] for row in encoded])
                cumulative_mi = max(0., h_prefix + h_y - h_prefix_y)
                row_out['target_cumulative_mutual_information_bits'] = cumulative_mi
                row_out['target_incremental_information_bits'] = max(0., cumulative_mi-old_target_mi)
                old_target_mi = cumulative_mi
            information['levels'].append(row_out)
            group_samples[depth] = current
            group_entropy[depth] = h_level
            previous = next_prefix
            old_h = h_prefix
            prior_names.extend(m['variable_id'] for m in members)
        depths = sorted(group_samples)
        for idx, depth in enumerate(depths):
            for other in depths[idx+1:]:
                first, second = group_samples[depth], group_samples[other]
                h_joint = _joint_entropy([a+b for a,b in zip(first,second)])
                information['cross_level_pairs'].append({
                    'left_level': depth, 'right_level': other,
                    'mutual_information_bits': max(0., group_entropy[depth]+group_entropy[other]-h_joint),
                    'support': len(encoded),
                    'interpretation': 'pairwise_interlevel_overlap_not_additive',
                })
        information['joint_entropy_bits'] = h_all
        information['total_correlation_bits'] = max(0., sum_marginal-h_all)
        information['target_mutual_information_bits'] = old_target_mi if outside else None
        information['sparse_joint_states_relative_to_support'] = len(set(encoded))*5 >= len(encoded)

    mapped_ids = {m['node_id'] for m in ordered}
    inventory = [{'level': depth, 'node_count': len(ids),
                  'mapped_node_count': len(set(ids) & mapped_ids),
                  'unmapped_node_count': len(ids)-len(set(ids)&mapped_ids)}
                 for depth,ids in enumerate(index.levels)]
    return {
        'schema': NETWORK_ACCOUNTING_SCHEMA,
        'source_digest': source_sha256(source), 'binding_digest': binding_sha256(bound),
        'binding_plan': plan.to_dict(), 'observation_unit': source.observation_unit,
        'aligned_rows': len(bound.rows), 'source_topology': index.topology,
        'level_policy': index.level_policy,
        'structural_level_inventory': inventory,
        'structural_node_count': sum(x['node_count'] for x in inventory),
        'mapped_node_count': len(ordered),
        'unmapped_node_count': sum(x['unmapped_node_count'] for x in inventory),
        'complete_observation_coverage': all(x['unmapped_node_count']==0 for x in inventory),
        'statistical_scope': 'complete_network' if all(x['unmapped_node_count']==0 for x in inventory)
                             else 'explicit_mapped_network_subset',
        'mappings': ordered, 'target': outside, 'occurrence_policy': occurrence_policy,
        'discretization': fitted, 'min_samples': min_samples,
        'network_information': information,
        'scientific_cautions': [
            'All estimates use one matched complete-case cohort; target presence may reduce support.',
            'Information increments use the chain rule, not sums of pairwise mutual information.',
            'Cross-level pairwise MI values overlap and must never be summed as total information.',
            'Only explicitly mapped observations have scientific information estimates.',
            'No source edge or structural level is assumed to be a probabilistic dependency.',
            'Estimates are empirical plug-in descriptions; sparse joint states create finite-sample bias.',
            'Information measures are not causal effects, learned influence or statistical significance.',
        ],
    }
