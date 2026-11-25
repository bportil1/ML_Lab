"""HIA-07: empirical information at an exact structural depth.

All original level nodes remain in the inventory, including unmapped ones.
Numerical estimates describe only the explicitly mapped observed variables
on one common complete-case cohort. Topology is not a statistical network.

Conditioning on preceding levels and factorization are intentionally out of
scope (HIA-08 and HIA-09); no conditional-independence model is assumed here.
"""
from __future__ import annotations

from typing import Any

from .bindings import ObservationBindingPlan, VariableBinding, bind_observations, source_sha256
from .branch_uncertainty import binding_sha256
from .contracts import StructuredContractError
from .hierarchical_branch_statistics import _entry, _rows, _joint_entropy, _per_variable
from .hierarchical_index import build_hierarchy_index
from .information import _binning_definition, _numeric, _parameters
from .object import StructuredObject

LEVEL_SCHEMA = 'ml-lab.hia-level-statistics@1'


def analyze_hierarchical_level(
    source: StructuredObject, *, level: int, mappings: list[dict[str, str]],
    occurrence_policy: str = 'preserve', bins: dict[str, dict[str, Any]] | None = None,
    min_samples: int = 2,
) -> dict[str, Any]:
    """Compute H(L), TC(L), and node-level information for a *mapped* subset.

    Level membership derives from HIA-02's deterministic rooted-depth (trees,
    forests) or longest-path (DAG) policy. The original node inventory is
    complete. No observation mapping is silently inferred from source edges,
    labels, or node position, and no estimates are generated for unmapped IDs.
    """
    if not isinstance(source, StructuredObject):
        raise StructuredContractError('HIA-07 requires a validated StructuredObject')
    index = build_hierarchy_index(source)
    node_ids = index.level_ids(level)
    membership = set(node_ids)
    if not isinstance(mappings, list) or not mappings:
        raise StructuredContractError('declare at least one level node-to-observation mapping')
    declared = [_entry(item, 'level mapping') for item in mappings]
    if any(item['node_id'] not in membership for item in declared):
        raise StructuredContractError('mapped node must belong to the selected exact structural level')
    if (len({item['node_id'] for item in declared}) != len(declared) or
            len({item['variable_id'] for item in declared}) != len(declared)):
        raise StructuredContractError('level mappings require distinct original node IDs and observed columns')
    names = tuple(item['variable_id'] for item in declared)
    definitions = _parameters(bins=bins, variables=names, min_samples=min_samples,
                              min_stratum_samples=1, log_base=2.0,
                              missing_policy='complete_case', estimator='empirical_discrete')
    plan = ObservationBindingPlan(
        source_sha256(source), source.observation_unit,
        tuple(VariableBinding(item['variable_id'], (item['node_id'],), occurrence_policy)
              for item in declared),
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

    cohort = _rows(rows, names, fitted, min_samples)
    samples = cohort.pop('_encoded')
    information: dict[str, Any] = {
        'status': cohort['status'], 'support': cohort['support'],
        'excluded_rows': cohort['excluded_rows'],
        'excluded_by_reason': cohort['excluded_by_reason'],
        'joint_entropy_bits': None, 'sum_atomic_entropy_bits': None,
        'total_correlation_bits': None,
        'distinct_joint_states': None,
        'sparse_joint_states_relative_to_support': None,
        'per_mapped_node': [], 'finite_sample_bias_warning': True,
    }
    if cohort['status'] == 'ok':
        per = _per_variable(samples, names)
        joint = _joint_entropy(samples)
        marginal_sum = sum(item['entropy_bits'] for item in per)
        information.update(
            joint_entropy_bits=joint,
            sum_atomic_entropy_bits=marginal_sum,
            total_correlation_bits=max(0.0, marginal_sum - joint),
            distinct_joint_states=len(set(samples)),
            sparse_joint_states_relative_to_support=len(set(samples)) * 5 >= len(samples),
            per_mapped_node=[{'node_id': entry['node_id'], **stats}
                             for entry, stats in zip(declared, per)],
        )

    unmapped = len(node_ids) - len(declared)
    return {
        'schema': LEVEL_SCHEMA,
        'source_digest': source_sha256(source),
        'binding_digest': binding_sha256(bound),
        'binding_plan': plan.to_dict(),
        'observation_unit': source.observation_unit,
        'aligned_rows': len(rows),
        'source_topology': index.topology,
        'level_policy': index.level_policy,
        'level': level,
        'level_node_ids': list(node_ids),
        'level_node_count': len(node_ids),
        'preceding_level_count': level,
        'preceding_level_node_count': sum(len(ids) for ids in index.levels[:level]),
        'mapped_node_count': len(declared),
        'unmapped_node_count': unmapped,
        'complete_level_statistical_coverage': unmapped == 0,
        'statistical_scope': 'complete_level' if unmapped == 0 else 'explicit_mapped_level_subset',
        'mappings': declared,
        'occurrence_policy': occurrence_policy,
        'discretization': fitted,
        'level_information': information,
        'conditional_information_status': 'not_computed_hia08',
        'scientific_cautions': [
            'Source depth is structural, not a learned statistical layer or probabilistic parent set.',
            'All original level nodes are retained; estimates cover only explicitly mapped columns.',
            'Each entropy and total correlation uses the same complete-case observation cohort.',
            'MI with the other mapped members is not an additive allocation of joint information.',
            'Earlier levels are not conditioned on here; that analysis belongs to HIA-08.',
            'No conditional-independence approximation or probabilistic factorization is assumed.',
            'Empirical discrete plug-in estimates are sensitive to sparse/high-dimensional joint states.',
        ],
    }
