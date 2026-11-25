"""HIA-09: explicit, in-sample conditional-independence model for a mapped level.

This is a *model comparison*, not an independence test or a causal model.
HIA-08's full empirical joint estimate remains unmodified and authoritative.
All factor tables are fit using the exact same complete-case cohort. Model
cross-entropy and empirical joint entropy are therefore in-sample estimates.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any

from .bindings import ObservationBindingPlan, bind_observations
from .contracts import StructuredContractError
from .hierarchical_branch_statistics import _joint_entropy, _rows
from .hierarchical_conditional_levels import analyze_hierarchical_conditional_level
from .information import _entropy
from .object import StructuredObject

FACTORIZATION_SCHEMA = 'ml-lab.hia-conditional-independence-model@1'
FACTORIZATION_MODES = ('shared_joint_prior', 'explicit_prior_parent_sets')


def _parent_contract(level_mappings: list[dict[str, str]],
                     preceding_mappings: list[dict[str, str]], *, mode: str,
                     parent_sets: dict[str, list[str]] | None) -> dict[str, list[str]]:
    level_ids = [item['node_id'] for item in level_mappings]
    prior_ids = [item['node_id'] for item in preceding_mappings]
    if mode not in FACTORIZATION_MODES:
        raise StructuredContractError('factorization_mode must be shared_joint_prior or explicit_prior_parent_sets')
    if mode == 'shared_joint_prior':
        if parent_sets not in (None, {}):
            raise StructuredContractError('shared_joint_prior does not accept independent parent overrides')
        return {node: list(prior_ids) for node in level_ids}
    if not isinstance(parent_sets, dict) or set(parent_sets) != set(level_ids):
        raise StructuredContractError('explicit_prior_parent_sets requires an entry for every mapped level node')
    allowed = set(prior_ids)
    result: dict[str, list[str]] = {}
    for node in level_ids:
        parents = parent_sets[node]
        if (not isinstance(parents, list) or
                any(not isinstance(parent, str) or parent not in allowed for parent in parents) or
                len(set(parents)) != len(parents)):
            raise StructuredContractError('parent sets must contain distinct exact mapped preceding-level node IDs')
        result[node] = list(parents)
    return result


def _factor_estimate(encoded: list[tuple], *, level_mappings: list[dict[str, str]],
                     preceding_mappings: list[dict[str, str]],
                     parent_sets: dict[str, list[str]], empirical: dict) -> dict[str, Any]:
    """Compute E_P[-log2 Q(X|Z)] and D_KL(P(X|Z)||Q(X|Z)).

    Q(X|Z)=product_i P_emp(X_i | Z_{parents(i)}); no factorization of
    the earlier-level joint P_emp(Z) is made. Every empirical conditional
    probability is fit on the *same* complete-case cohort as HIA-08.
    """
    n = len(level_mappings)
    prior_ids = [item['node_id'] for item in preceding_mappings]
    indices = {node: n + index for index, node in enumerate(prior_ids)}
    count = len(encoded)
    strata_values: dict[tuple, Counter] = defaultdict(Counter)
    for row in encoded:
        strata_values[row[n:]][row[:n]] += 1
    z_counts = Counter({z: sum(xstates.values()) for z, xstates in strata_values.items()})
    factors = []
    cross_entropy = 0.0
    node_results: list[dict[str, Any]] = []
    for i, mapping in enumerate(level_mappings):
        parent_indices = tuple(indices[parent] for parent in parent_sets[mapping['node_id']])
        joint = Counter((tuple(row[k] for k in parent_indices), row[i]) for row in encoded)
        given = Counter({})
        for (parents, _value), freq in joint.items():
            given[parents] += freq
        local_cross_entropy = -sum(freq / count * math.log2(freq / given[parents])
                                   for (parents, _value), freq in joint.items())
        cross_entropy += local_cross_entropy
        observed_full = empirical['per_mapped_level_node'][i]['entropy_given_prior_bits']
        node_results.append({
            'node_id': mapping['node_id'], 'variable_id': mapping['variable_id'],
            'declared_parent_node_ids': list(parent_sets[mapping['node_id']]),
            'parent_strata': len(given), 'distinct_parent_value_states': len(joint),
            'model_conditional_entropy_bits': local_cross_entropy,
            'empirical_entropy_given_full_prior_bits': observed_full,
            'omitted_prior_information_bits': max(0.0, local_cross_entropy - observed_full),
        })
        factors.append((parent_indices, joint, given))

    # This is the weighted conditional KL on the empirical joint P(X,Z).
    empirical_h = empirical['level_conditional_entropy_bits']
    divergence = max(0.0, cross_entropy - empirical_h)
    strata_reports = []
    for state, size in z_counts.items():
        x_counts = strata_values[state]
        empirical_stratum_h = _entropy(x_counts, size, 2.0)
        modeled_cross = 0.0
        for xs, freq in x_counts.items():
            neg_log_q = 0.0
            for i, (parent_indices, joint, given) in enumerate(factors):
                parents = tuple(state[k-n] for k in parent_indices)
                conditional = joint[(parents, xs[i])] / given[parents]
                neg_log_q -= math.log2(conditional)
            modeled_cross += (freq / size) * neg_log_q
        strata_reports.append({
            'preceding_states': [list(value) for value in state], 'support': size,
            'empirical_joint_entropy_given_stratum_bits': empirical_stratum_h,
            'factorized_cross_entropy_given_stratum_bits': modeled_cross,
            'empirical_model_kl_divergence_given_stratum_bits':
                max(0.0, modeled_cross - empirical_stratum_h),
        })

    # Check the empirical weighted-strata decomposition. This is an internal
    # assertion about numerical correctness, never a model assumption.
    weighted = sum(item['support'] / count * item['empirical_model_kl_divergence_given_stratum_bits']
                   for item in strata_reports)
    if not math.isclose(weighted, divergence, abs_tol=1e-8, rel_tol=1e-8):
        raise StructuredContractError('conditional independence comparison failed cohort consistency check')
    residual_dependence = empirical['conditional_total_correlation_bits']
    omitted = sum(item['omitted_prior_information_bits'] for item in node_results)
    return {
        'status': 'ok', 'support': count, 'conditioning_strata': len(z_counts),
        'empirical_joint_conditional_entropy_bits': empirical_h,
        'factorized_model_conditional_entropy_bits': cross_entropy,
        'factorized_cross_entropy_bits': cross_entropy,
        'empirical_model_kl_divergence_bits': divergence,
        'empirical_conditional_total_correlation_bits': residual_dependence,
        'omitted_prior_information_bits': omitted,
        'decomposition_difference_bits': divergence - residual_dependence - omitted,
        'per_mapped_level_node': node_results,
        'per_joint_prior_stratum': strata_reports,
        'in_sample_model_fit': True, 'independence_is_assumed_not_established': True,
    }


def analyze_hierarchical_factorized_level(
    source: StructuredObject, *, level: int,
    mappings: list[dict[str, str]], preceding_mappings: list[dict[str, str]],
    assumption_confirmed: bool, factorization_mode: str,
    parent_sets: dict[str, list[str]] | None = None,
    target: dict[str, str] | None = None,
    occurrence_policy: str = 'preserve', bins: dict[str, dict[str, Any]] | None = None,
    min_samples: int = 2, min_stratum_samples: int = 2,
) -> dict[str, Any]:
    """Compare an *explicit* conditional-independence model to HIA-08.

    `assumption_confirmed=True` is mandatory. The original graph does not
    supply probabilistic parents. Parent sets must reference exact mapped
    preceding-level IDs, never inferred source links.
    """
    if assumption_confirmed is not True:
        raise StructuredContractError('explicit conditional-independence modeling requires assumption_confirmed=true')
    # HIA-08 does exact structural, observation, support and bin validation.
    empirical = analyze_hierarchical_conditional_level(
        source, level=level, mappings=mappings, preceding_mappings=preceding_mappings,
        target=target, occurrence_policy=occurrence_policy, bins=bins,
        min_samples=min_samples, min_stratum_samples=min_stratum_samples,
    )
    parents = _parent_contract(empirical['mappings'], empirical['preceding_mappings'],
                               mode=factorization_mode, parent_sets=parent_sets)
    base = empirical['level_conditioned_information']
    report: dict[str, Any] = {
        'schema': FACTORIZATION_SCHEMA,
        'source_digest': empirical['source_digest'],
        'binding_digest': empirical['binding_digest'],
        'binding_plan': empirical['binding_plan'],
        'level': empirical['level'],
        'level_policy': empirical['level_policy'],
        'source_topology': empirical['source_topology'],
        'level_node_ids': empirical['level_node_ids'],
        'preceding_level_node_ids': empirical['preceding_level_node_ids'],
        'level_node_count': empirical['level_node_count'],
        'preceding_level_node_count': empirical['preceding_level_node_count'],
        'mapped_level_node_count': empirical['mapped_level_node_count'],
        'mapped_preceding_node_count': empirical['mapped_preceding_node_count'],
        'statistical_scope': empirical['statistical_scope'],
        'mappings': empirical['mappings'],
        'preceding_mappings': empirical['preceding_mappings'],
        'target': empirical['target'],
        'factorization_mode': factorization_mode,
        'declared_parent_sets': parents,
        'assumption_confirmed': True,
        'assumption': 'Q(X|Z)=product_i P_emp(X_i|declared_prior_parents_i)',
        'prior_joint_distribution_factorized': False,
        'base_empirical_schema': empirical['schema'],
        'empirical_level_conditioned_information': base,
        'empirical_target_information': empirical['target_increment_information'],
        'occurrence_policy': empirical['occurrence_policy'],
        'discretization': empirical['discretization'],
        'min_samples': empirical['min_samples'],
        'min_stratum_samples': empirical['min_stratum_samples'],
        'model': None,
        'scientific_cautions': [
            'Conditional independence is an EXPLICIT model assumption, not a property inferred from source topology.',
            'All factor tables and diagnostics use the same IN-SAMPLE complete-case cohort; divergence is descriptive, not validation or a hypothesis-test p-value.',
            'Earlier-level variables are modeled as a JOINT conditioning context, not automatically independent.',
            'An explicit parent subset can omit predictive earlier-level information; KL then includes both omitted-context information and within-level dependence.',
            'Model and empirical estimates represent explicitly mapped nodes, never unmapped source nodes.',
            'Sparse joint states bias plug-in estimates; zero in-sample divergence does not establish population independence.',
            'The optional target has its own separate cohort and is not factorized or compared to level model entropy.',
        ],
    }
    if base['status'] != 'ok':
        report['model'] = {
            'status': base['status'], 'support': base['support'],
            'empirical_joint_conditional_entropy_bits': None,
            'factorized_model_conditional_entropy_bits': None,
            'factorized_cross_entropy_bits': None,
            'empirical_model_kl_divergence_bits': None,
            'empirical_conditional_total_correlation_bits': None,
            'omitted_prior_information_bits': None,
            'per_mapped_level_node': [], 'per_joint_prior_stratum': [],
        }
        return report
    plan = ObservationBindingPlan.from_dict(empirical['binding_plan'])
    bound = bind_observations(source, plan)
    names = tuple(item['variable_id'] for item in [*empirical['mappings'], *empirical['preceding_mappings']])
    cohort = _rows(bound.rows, names, empirical['discretization'], min_samples)
    if cohort['status'] != 'ok' or cohort['support'] != base['support']:
        raise StructuredContractError('factorization cohort differs from HIA-08 complete-case cohort')
    report['model'] = _factor_estimate(
        cohort['_encoded'], level_mappings=empirical['mappings'],
        preceding_mappings=empirical['preceding_mappings'], parent_sets=parents,
        empirical=base,
    )
    return report
