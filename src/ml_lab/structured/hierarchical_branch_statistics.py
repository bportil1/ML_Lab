"""HIA-06: branch-wide empirical information from explicitly bound observations.

The complete source branch is retained even if only some of its nodes are
mapped to observed columns. Every numerical estimate is labelled as a statistic
of the *mapped subset*, not the unobserved full source branch. Branch structure
never asserts statistical or conditional-independence relations.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from .bindings import ObservationBindingPlan, VariableBinding, bind_observations, source_sha256
from .branch_uncertainty import binding_sha256
from .contracts import StructuredContractError
from .hierarchical_index import build_hierarchy_index
from .information import (_bin_index, _binning_definition, _categorical, _entropy,
                          _numeric, _parameters)
from .object import StructuredObject

BRANCH_SCHEMA = 'ml-lab.hia-branch-statistics@1'


def _entry(raw: Any, label: str) -> dict[str, str]:
    if not isinstance(raw, dict) or set(raw) != {'node_id', 'variable_id'}:
        raise StructuredContractError(f'{label} requires exact node_id and variable_id')
    if any(not isinstance(value, str) or not value.strip() for value in raw.values()):
        raise StructuredContractError(f'{label} requires nonempty source node and observation column IDs')
    return raw


def _encode(cell: dict[str, Any], name: str, fitted: dict[str, dict]) -> tuple:
    if cell['status'] == 'empty':
        return 'state', 'empty'
    if name in fitted:
        return 'bin', _bin_index(_numeric(cell['value'], name), fitted[name])
    return _categorical(cell['value'], name)


def _rows(rows: list[dict], names: tuple[str, ...], fitted: dict[str, dict],
          minimum: int, *, min_stratum: int = 1, context_name: str | None = None) -> dict[str, Any]:
    """One common complete-case cohort per estimand; never mix supports."""
    encoded: list[tuple] = []
    exclusions: Counter[str] = Counter()
    for row in rows:
        cells = row['values']
        states = [cells[name]['status'] for name in names]
        if 'multiple' in states:
            exclusions['unresolved_repeated_values'] += 1
            continue
        if any(state not in ('present', 'empty') for state in states):
            exclusions['incomplete_case'] += 1
            continue
        try:
            encoded.append(tuple(_encode(cells[name], name, fitted) for name in names))
        except StructuredContractError as exc:
            message = str(exc)
            if 'require explicit discretization' in message:
                exclusions['float_requires_explicit_binning'] += 1
            elif 'discretization requires' in message:
                exclusions['binning_requires_numeric_values'] += 1
            else:
                exclusions['unsupported_non_scalar_values'] += 1
    priority = ('unresolved_repeated_values', 'float_requires_explicit_binning',
                'binning_requires_numeric_values', 'unsupported_non_scalar_values')
    invalid = next((code for code in priority if exclusions[code]), None)
    status = invalid or ('insufficient_samples' if len(encoded) < minimum else 'ok')
    strata = None
    short = None
    if context_name and status == 'ok':
        context_index = names.index(context_name)
        counts = Counter(values[context_index] for values in encoded)
        strata = len(counts)
        short = sum(count < min_stratum for count in counts.values())
        if short:
            status = 'insufficient_conditioning_support'
    return {'status': status, 'support': len(encoded),
            'excluded_rows': len(rows) - len(encoded),
            'excluded_by_reason': dict(sorted(exclusions.items())),
            'conditioning_strata': strata, 'under_supported_strata': short,
            '_encoded': encoded}


def _joint_entropy(data: list[tuple]) -> float:
    return _entropy(Counter(data), len(data), 2.0)


def _per_variable(data: list[tuple], names: tuple[str, ...]) -> list[dict[str, Any]]:
    """Each node's entropy and unique remainder on the same joint cohort."""
    joint = _joint_entropy(data)
    results = []
    for index, name in enumerate(names):
        marginal = _entropy(Counter(row[index] for row in data), len(data), 2.0)
        complement = _joint_entropy([row[:index] + row[index + 1:] for row in data])
        h_given_rest = max(0.0, joint - complement)
        shared = max(0.0, marginal - h_given_rest)
        results.append({'variable_id': name, 'entropy_bits': marginal,
                        'conditional_entropy_given_other_mapped_members_bits': h_given_rest,
                        'mi_with_other_mapped_members_bits': shared})
    return results


def analyze_hierarchical_branch(
    source: StructuredObject, *, branch_root_id: str,
    mappings: list[dict[str, str]], context: dict[str, str] | None = None,
    occurrence_policy: str = 'preserve', bins: dict[str, dict[str, Any]] | None = None,
    min_samples: int = 2, min_stratum_samples: int = 2,
) -> dict[str, Any]:
    """Empirical H(branch), total correlation and optional parent/context MI.

    The branch contains all source descendants of its root. The mapped subset
    is declared one-to-one by the caller. Conditioning on an explicitly chosen
    external observed variable is descriptive, not a probabilistic factorization
    and not proof of information flowing along source edges.
    """
    if not isinstance(source, StructuredObject):
        raise StructuredContractError('HIA-06 requires a validated StructuredObject')
    index = build_hierarchy_index(source)  # rejects cyclic/undirected graphs
    if index.topology not in ('tree', 'forest'):
        raise StructuredContractError('branch statistics require unique-parent tree/forest topology; DAGs use level selection')
    all_ids = index.branch_ids(branch_root_id)
    membership = set(all_ids)
    if not isinstance(mappings, list) or not mappings:
        raise StructuredContractError('declare at least one branch node-to-observation mapping')
    declared = [_entry(item, 'branch mapping') for item in mappings]
    if any(item['node_id'] not in membership for item in declared):
        raise StructuredContractError('mapped branch node must belong to the selected exact branch')
    external = _entry(context, 'external context') if context is not None else None
    if external is not None:
        if external['node_id'] in membership:
            raise StructuredContractError('context node must be outside the selected source branch')
        index.level_of(external['node_id'])  # exact source-node validation
    assignments = [*declared, *([external] if external else [])]
    if (len({item['node_id'] for item in assignments}) != len(assignments) or
            len({item['variable_id'] for item in assignments}) != len(assignments)):
        raise StructuredContractError('branch and context mappings must use distinct original nodes and columns')
    names = tuple(item['variable_id'] for item in assignments)
    definitions = _parameters(bins=bins, variables=names, min_samples=min_samples,
                              min_stratum_samples=min_stratum_samples, log_base=2.0,
                              missing_policy='complete_case', estimator='empirical_discrete')
    plan = ObservationBindingPlan(source_sha256(source), source.observation_unit,
              tuple(VariableBinding(item['variable_id'], (item['node_id'],), occurrence_policy)
                    for item in assignments))
    bound = bind_observations(source, plan)
    rows = bound.rows
    fitted: dict[str, dict] = {}
    for name, definition in definitions.items():
        present = [row['values'][name]['value'] for row in rows
                   if row['values'][name]['status'] == 'present']
        if any(type(value) not in (int, float) for value in present):
            raise StructuredContractError(f'{name}: explicit numeric bins require numeric observations')
        fitted[name] = _binning_definition([_numeric(value, name) for value in present], name, definition)

    branch_names = tuple(item['variable_id'] for item in declared)
    ordinary = _rows(rows, branch_names, fitted, min_samples)
    ordinary_samples = ordinary.pop('_encoded')
    branch_statistics = {'status': ordinary['status'], 'support': ordinary['support'],
        'excluded_rows': ordinary['excluded_rows'],
        'excluded_by_reason': ordinary['excluded_by_reason'],
        'joint_entropy_bits': None, 'sum_atomic_entropy_bits': None,
        'total_correlation_bits': None, 'distinct_joint_states': None,
        'sparse_joint_states_relative_to_support': None, 'per_mapped_node': [],
        'finite_sample_bias_warning': True}
    if ordinary['status'] == 'ok':
        per = _per_variable(ordinary_samples, branch_names)
        h_joint = _joint_entropy(ordinary_samples)
        h_sum = sum(item['entropy_bits'] for item in per)
        branch_statistics.update(joint_entropy_bits=h_joint, sum_atomic_entropy_bits=h_sum,
                total_correlation_bits=max(0.0, h_sum - h_joint),
                distinct_joint_states=len(set(ordinary_samples)),
                sparse_joint_states_relative_to_support=len(set(ordinary_samples)) * 5 >= len(ordinary_samples),
                per_mapped_node=[
                    {'node_id': mapping['node_id'], **item}
                    for mapping, item in zip(declared, per)])

    conditioned = None
    if external:
        context_name = external['variable_id']
        joint = _rows(rows, names, fitted, min_samples,
                      min_stratum=min_stratum_samples, context_name=context_name)
        samples = joint.pop('_encoded')
        conditioned = {
            'status': joint['status'], 'support': joint['support'],
            'excluded_rows': joint['excluded_rows'],
            'excluded_by_reason': joint['excluded_by_reason'],
            'conditioning_strata': joint['conditioning_strata'],
            'under_supported_strata': joint['under_supported_strata'],
            'context_node_id': external['node_id'], 'context_variable_id': context_name,
            'context_is_direct_source_parent': external['node_id'] in index.parents_of(branch_root_id),
            'joint_branch_entropy_on_context_cohort_bits': None,
            'context_entropy_bits': None,
            'joint_branch_and_context_entropy_bits': None,
            'distinct_context_joint_states': None,
            'sparse_context_joint_states_relative_to_support': None,
            'branch_conditional_entropy_bits': None,
            'branch_context_mutual_information_bits': None,
            'conditional_total_correlation_bits': None,
        }
        if joint['status'] == 'ok':
            branch_vectors = [record[:-1] for record in samples]
            context_states = [record[-1] for record in samples]
            h_branch = _joint_entropy(branch_vectors)
            h_context = _joint_entropy(context_states)
            h_joint_context = _joint_entropy(samples)
            h_given = max(0.0, h_joint_context - h_context)
            mi = max(0.0, h_branch + h_context - h_joint_context)
            per_given = []
            for i in range(len(branch_names)):
                h_xz = _joint_entropy([(record[i], record[-1]) for record in samples])
                per_given.append(max(0.0, h_xz - h_context))
            conditioned.update(
                joint_branch_entropy_on_context_cohort_bits=h_branch,
                context_entropy_bits=h_context,
                joint_branch_and_context_entropy_bits=h_joint_context,
                distinct_context_joint_states=len(set(samples)),
                sparse_context_joint_states_relative_to_support=len(set(samples)) * 5 >= len(samples),
                branch_conditional_entropy_bits=h_given,
                branch_context_mutual_information_bits=mi,
                conditional_total_correlation_bits=max(0.0, sum(per_given) - h_given),
            )

    unmapped = len(all_ids) - len(declared)
    return {
        'schema': BRANCH_SCHEMA, 'source_digest': source_sha256(source),
        'binding_digest': binding_sha256(bound), 'binding_plan': plan.to_dict(),
        'observation_unit': source.observation_unit, 'aligned_rows': len(rows),
        'branch_root_id': branch_root_id, 'source_topology': index.topology,
        'branch_node_ids': list(all_ids), 'branch_node_count': len(all_ids),
        'branch_structural_level_range': [index.level_of(all_ids[0]), max(index.level_of(node) for node in all_ids)],
        'mapped_node_count': len(declared), 'unmapped_node_count': unmapped,
        'complete_branch_statistical_coverage': unmapped == 0,
        'mappings': declared, 'external_context': external,
        'occurrence_policy': occurrence_policy, 'discretization': fitted,
        'branch_information': branch_statistics, 'context_information': conditioned,
        'statistical_scope': 'complete_branch' if unmapped == 0 else 'explicit_mapped_branch_subset',
        'scientific_cautions': [
            'All node IDs describe source structure, not independent observations.',
            'Estimates cover only explicitly mapped observed variables, not unmapped branch nodes.',
            'Total correlation is sum of marginal entropies minus joint entropy on identical aligned rows.',
            'Context estimates use their own complete-case cohort; they are not compared across unmatched support.',
            'An external source parent is not an inferred probabilistic parent or causal condition.',
            'No conditional-independence model or factorization is assumed.',
            'Empirical discrete plug-in estimates may be biased or sparse in high dimensions.',
        ],
    }
