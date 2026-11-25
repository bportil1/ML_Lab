"""HIA-08: empirical conditional information for explicitly mapped levels.

All node identities and depth membership are from HIA-02; observed columns are
assigned explicitly through SU-05. Conditioning uses a *joint* preceding-level
vector (not source parents, causal arrows or a factorized independence model).
Every estimand is computed from one complete-case cohort. An optional target
has its own matched cohort and never silently replaces the level cohort.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from .bindings import ObservationBindingPlan, VariableBinding, bind_observations, source_sha256
from .branch_uncertainty import binding_sha256
from .contracts import StructuredContractError
from .hierarchical_branch_statistics import _entry, _joint_entropy, _rows
from .hierarchical_index import build_hierarchy_index
from .information import _binning_definition, _numeric, _parameters
from .object import StructuredObject

CONDITIONAL_LEVEL_SCHEMA = 'ml-lab.hia-conditional-level-information@1'


def _estimate(rows: list[dict], names: tuple[str, ...], fitted: dict[str, dict],
              *, n_level: int, n_prior: int, minimum: int,
              minimum_stratum: int, target: bool) -> dict[str, Any]:
    cohort = _rows(rows, names, fitted, minimum)
    encoded = cohort.pop('_encoded')
    result: dict[str, Any] = {
        'status': cohort['status'], 'support': cohort['support'],
        'excluded_rows': cohort['excluded_rows'],
        'excluded_by_reason': cohort['excluded_by_reason'],
        'conditioning_strata': None, 'under_supported_strata': None,
        'distinct_joint_states': None, 'sparse_joint_states_relative_to_support': None,
        'finite_sample_bias_warning': True,
    }
    fields = ('level_entropy_bits', 'prior_entropy_bits', 'level_and_prior_entropy_bits',
              'level_conditional_entropy_bits', 'level_prior_mutual_information_bits',
              'conditional_total_correlation_bits')
    result.update({key: None for key in fields})
    result['per_mapped_level_node'] = []
    if target:
        result.update({key: None for key in (
            'target_entropy_given_prior_bits', 'target_entropy_given_level_and_prior_bits',
            'target_incremental_information_bits', 'target_entropy_bits',
            'target_and_prior_entropy_bits', 'target_level_prior_joint_entropy_bits',
        )})
    if result['status'] != 'ok':
        return result

    # Every stratum is the *joint* vector of all mapped preceding-level values.
    conditioning = [row[n_level:n_level + n_prior] for row in encoded]
    strata = Counter(conditioning)
    short = sum(count < minimum_stratum for count in strata.values())
    result['conditioning_strata'] = len(strata)
    result['under_supported_strata'] = short
    if short:
        result['status'] = 'insufficient_conditioning_support'
        return result

    levels = [row[:n_level] for row in encoded]
    level_prior = [row[:n_level + n_prior] for row in encoded]
    hx = _joint_entropy(levels)
    hz = _joint_entropy(conditioning)
    hxz = _joint_entropy(level_prior)
    hx_given_z = max(0.0, hxz - hz)
    result.update(
        level_entropy_bits=hx, prior_entropy_bits=hz,
        level_and_prior_entropy_bits=hxz,
        level_conditional_entropy_bits=hx_given_z,
        level_prior_mutual_information_bits=max(0.0, hx + hz - hxz),
        distinct_joint_states=len(set(encoded)),
        sparse_joint_states_relative_to_support=len(set(encoded)) * 5 >= len(encoded),
    )
    # Conditional total correlation is measured, not a factorization. HIA-09
    # may later contrast its null hypothesis against this empirical quantity.
    per: list[dict[str, Any]] = []
    sum_h_xi_z = 0.0
    for i in range(n_level):
        marginal_context = [row[i:i + 1] + row[n_level:n_level + n_prior]
                            for row in encoded]
        h_xi_z = max(0.0, _joint_entropy(marginal_context) - hz)
        h_xi_given_other_z = max(0.0, hxz - _joint_entropy([
            row[:i] + row[i + 1:n_level + n_prior] for row in encoded]))
        sum_h_xi_z += h_xi_z
        per.append({
            'variable_id': names[i],
            'entropy_given_prior_bits': h_xi_z,
            'entropy_given_other_level_members_and_prior_bits': h_xi_given_other_z,
            'mi_with_other_level_members_given_prior_bits':
                max(0.0, h_xi_z - h_xi_given_other_z),
        })
    result['per_mapped_level_node'] = per
    result['conditional_total_correlation_bits'] = max(0.0, sum_h_xi_z - hx_given_z)

    if target:
        # Last variable is the explicitly mapped Y, not a structural edge label.
        target_states = [row[-1:] for row in encoded]
        target_prior = [row[n_level:] for row in encoded]
        hy = _joint_entropy(target_states)
        hyz = _joint_entropy(target_prior)
        hxyz = _joint_entropy(encoded)
        result.update(
            target_entropy_bits=hy,
            target_and_prior_entropy_bits=hyz,
            target_level_prior_joint_entropy_bits=hxyz,
            target_entropy_given_prior_bits=max(0.0, hyz - hz),
            target_entropy_given_level_and_prior_bits=max(0.0, hxyz - hxz),
            target_incremental_information_bits=max(0.0, hxz + hyz - hz - hxyz),
        )
    return result


def analyze_hierarchical_conditional_level(
    source: StructuredObject, *, level: int,
    mappings: list[dict[str, str]], preceding_mappings: list[dict[str, str]],
    target: dict[str, str] | None = None,
    occurrence_policy: str = 'preserve', bins: dict[str, dict[str, Any]] | None = None,
    min_samples: int = 2, min_stratum_samples: int = 2,
) -> dict[str, Any]:
    """Estimate H(L|prior), I(L;prior), optionally I(Y;L|prior).

    `preceding_mappings` explicitly declares only mapped nodes at depths below
    `level`, not an unverified *complete* prior-level collection. With an
    incomplete mapping the result refers to that subset, not all L_<k.
    """
    if not isinstance(source, StructuredObject):
        raise StructuredContractError('HIA-08 requires a validated StructuredObject')
    index = build_hierarchy_index(source)
    selected = index.level_ids(level)
    previous = index.prior_level_ids(level)
    if not isinstance(mappings, list) or not mappings:
        raise StructuredContractError('declare at least one level node-to-observation mapping')
    if not isinstance(preceding_mappings, list):
        raise StructuredContractError('preceding_mappings must be an explicit list')
    if level > 0 and not preceding_mappings:
        raise StructuredContractError('declare preceding-level mappings to condition level information')
    if level == 0 and preceding_mappings:
        raise StructuredContractError('root level has no preceding source nodes')
    declared = [_entry(item, 'level mapping') for item in mappings]
    prior = [_entry(item, 'preceding-level mapping') for item in preceding_mappings]
    external = _entry(target, 'target mapping') if target is not None else None
    selected_ids, prior_ids = set(selected), set(previous)
    if any(item['node_id'] not in selected_ids for item in declared):
        raise StructuredContractError('mapped node must belong to the selected structural level')
    if any(item['node_id'] not in prior_ids for item in prior):
        raise StructuredContractError('preceding mapping must reference a node at a strictly earlier level')
    if external is not None:
        index.level_of(external['node_id'])
        if external['node_id'] in selected_ids:
            raise StructuredContractError('target node must be outside the selected structural level')
    assigned = [*declared, *prior, *([external] if external else [])]
    if (len({item['node_id'] for item in assigned}) != len(assigned) or
            len({item['variable_id'] for item in assigned}) != len(assigned)):
        raise StructuredContractError('level, preceding and target mappings require distinct source IDs and columns')
    names = tuple(item['variable_id'] for item in assigned)
    definitions = _parameters(bins=bins, variables=names, min_samples=min_samples,
                              min_stratum_samples=min_stratum_samples, log_base=2.0,
                              missing_policy='complete_case', estimator='empirical_discrete')
    plan = ObservationBindingPlan(
        source_sha256(source), source.observation_unit,
        tuple(VariableBinding(item['variable_id'], (item['node_id'],), occurrence_policy)
              for item in assigned),
    )
    bound = bind_observations(source, plan)
    rows = bound.rows
    fitted: dict[str, dict] = {}
    for name, definition in definitions.items():
        present = [row['values'][name]['value'] for row in rows
                   if row['values'][name]['status'] == 'present']
        if any(type(value) not in (int, float) for value in present):
            raise StructuredContractError(f'{name}: explicit numeric bins require numeric observations')
        fitted[name] = _binning_definition([_numeric(value, name) for value in present], name, definition)

    n_level = len(declared)
    n_prior = len(prior)
    base_names = tuple(item['variable_id'] for item in [*declared, *prior])
    ordinary = _estimate(rows, base_names, fitted, n_level=n_level, n_prior=n_prior,
                         minimum=min_samples, minimum_stratum=min_stratum_samples, target=False)
    for mapping, entry in zip(declared, ordinary['per_mapped_level_node']):
        entry['node_id'] = mapping['node_id']
    target_info = None
    if external:
        target_info = _estimate(rows, names, fitted, n_level=n_level, n_prior=n_prior,
                                minimum=min_samples, minimum_stratum=min_stratum_samples,
                                target=True)
        for mapping, entry in zip(declared, target_info['per_mapped_level_node']):
            entry['node_id'] = mapping['node_id']
    return {
        'schema': CONDITIONAL_LEVEL_SCHEMA,
        'source_digest': source_sha256(source),
        'binding_digest': binding_sha256(bound),
        'binding_plan': plan.to_dict(),
        'observation_unit': source.observation_unit,
        'aligned_rows': len(rows),
        'source_topology': index.topology,
        'level_policy': index.level_policy,
        'level': level,
        'level_node_ids': list(selected),
        'level_node_count': len(selected),
        'preceding_level_node_ids': list(previous),
        'preceding_level_node_count': len(previous),
        'mappings': declared,
        'preceding_mappings': prior,
        'target': external,
        'mapped_level_node_count': len(declared),
        'unmapped_level_node_count': len(selected) - len(declared),
        'mapped_preceding_node_count': len(prior),
        'unmapped_preceding_node_count': len(previous) - len(prior),
        'level_complete_observation_coverage': len(declared) == len(selected),
        'preceding_complete_observation_coverage': len(prior) == len(previous),
        'statistical_scope': 'complete_level_and_preceding_levels' if (
            len(declared) == len(selected) and len(prior) == len(previous))
            else 'explicit_mapped_level_and_preceding_subsets',
        'occurrence_policy': occurrence_policy,
        'discretization': fitted,
        'min_samples': min_samples,
        'min_stratum_samples': min_stratum_samples,
        'level_conditioned_information': ordinary,
        'target_increment_information': target_info,
        'independence_model': 'empirical_joint_no_factorization',
        'scientific_cautions': [
            'Source levels and parent links describe topology only, not probabilistic dependencies.',
            'Estimates describe explicitly mapped observation subsets; unmapped nodes are never implicitly zero.',
            'Conditioning uses the entire JOINT preceding-mapped vector, not a product of independent parents.',
            'The level-only and optional target estimates use different complete-case cohorts and must not be subtracted.',
            'Every entropy within a cohort uses exactly the same aligned complete-case observations.',
            'All discrete plug-in information estimates can be biased under sparse joint-state support.',
            'No conditional-independence model, causal information flow or learned direction is inferred.',
        ],
    }
