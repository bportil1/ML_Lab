"""HIA-04: explicit, source-pinned atomic observation summaries.

An original structural node is *not* a sample. A user supplies the single
observed variable bound to that exact node; SU-05 validates and resolves the
mapping. Only aligned observation rows contribute empirical summaries.

The discrete plug-in entropy uses the same encoding and binning primitives as
SU-07's information estimator. This module never infers associations from
source topology, silently aggregates repeated cells, or modifies the source.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from typing import Any

from .bindings import (ObservationBindingPlan, VariableBinding, bind_observations,
                       source_sha256)
from .branch_uncertainty import binding_sha256
from .contracts import StructuredContractError
from .descriptive import _stats
from .hierarchical_index import build_hierarchy_index
from .information import (_bin_index, _binning_definition, _categorical, _entropy,
                          _numeric, _parameters)
from .object import StructuredObject

ATOMIC_SCHEMA = 'ml-lab.hia-atomic-statistics@1'


def atomic_node_statistics(
    source: StructuredObject, *, node_id: str, variable_id: str,
    occurrence_policy: str = 'preserve', bins: dict[str, Any] | None = None,
    min_samples: int = 2,
) -> dict[str, Any]:
    """Describe one exact source node under one explicit SU-05 column mapping.

    ``bins`` is None for unbinned categorical values or an explicit bin rule,
    e.g. {"mode":"quantile","bins":4}. Numeric sample statistics do NOT
    depend on discretization; entropy of floating-point values DOES.
    """
    if not isinstance(source, StructuredObject):
        raise StructuredContractError('HIA-04 requires a validated StructuredObject')
    if not isinstance(node_id, str) or not node_id.strip():
        raise StructuredContractError('node_id must be an original nonempty node ID')
    if not isinstance(variable_id, str) or not variable_id.strip():
        raise StructuredContractError('variable_id must be an explicit nonempty observation column ID')
    # Validate against the actual source, not a graph summary or client node ID.
    index = build_hierarchy_index(source)
    context = index.node_context(node_id)
    source_digest = source_sha256(source)
    if bins is not None:
        if not isinstance(bins, dict):
            raise StructuredContractError('bins must be null or a binning definition')
        _parameters(bins={variable_id: bins}, variables=(variable_id,),
                    min_samples=min_samples, min_stratum_samples=1,
                    log_base=2.0, missing_policy='complete_case',
                    estimator='empirical_discrete')
    else:
        _parameters(bins=None, variables=(variable_id,),
                    min_samples=min_samples, min_stratum_samples=1,
                    log_base=2.0, missing_policy='complete_case',
                    estimator='empirical_discrete')
    # Enforce source-to-observation mapping using the existing SU-05 contract.
    plan = ObservationBindingPlan(
        source_digest=source_digest, observation_unit=source.observation_unit,
        variables=(VariableBinding(variable_id, (node_id,), occurrence_policy),),
    )
    bound = bind_observations(source, plan)
    rows = bound.rows
    original_variable_evidence = bound.report['variables'][0]
    cells = [row['values'][variable_id] for row in rows]
    statuses = Counter(cell['status'] for cell in cells)
    reasons = Counter(cell.get('source_reason', 'unspecified') for cell in cells
                      if cell['status'] in ('missing', 'explicit_null', 'unknown'))
    # Keep every categorical state; do not conflate 1, True, and '1'.
    states: Counter[str] = Counter()
    seen_states: dict[str, dict[str, Any]] = {}
    numeric_values: list[float] = []
    numeric_incompatible = 0
    floats_seen = False
    multiple_rows = statuses['multiple']
    for cell in cells:
        state = cell['status']
        if state not in ('present', 'empty'):
            continue
        val = cell['value']
        if state == 'empty':
            encoded = ('state', 'empty')
        else:
            # The schema permits JSON values other than scalar numbers/string;
            # such values are not supported by the discrete estimator.
            if type(val) is float:
                floats_seen = True
            if type(val) in (int, float):
                numeric_values.append(_numeric(val, variable_id))
            else:
                numeric_incompatible += 1
            if type(val) in (bool, int, float, str):
                encoded = (type(val).__name__, val)
            else:
                # Distinguish unsupported JSON-valued states in the descriptive
                # inventory without pretending the entropy estimator supports them.
                encoded = ('unsupported_json', json.dumps(val, sort_keys=True,
                                                          ensure_ascii=False, allow_nan=False))
        key = json.dumps(encoded, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        states[key] += 1
        seen_states[key] = {'type': encoded[0], 'value': encoded[1]}

    # Numeric stats are a separate descriptive result from binned entropy.
    numeric = (_stats(numeric_values) if numeric_values and not numeric_incompatible
               else None)
    numeric_status = ('available' if numeric is not None else
                      ('mixed_or_non_numeric' if numeric_incompatible else 'no_numeric_observations'))
    included = [cell for cell in cells if cell['status'] in ('present', 'empty')]
    entropy_status = 'ok'
    entropy_value: float | None = None
    fitted = None
    entropy_states: Counter[tuple] = Counter()
    if not rows:
        entropy_status = 'no_observation_rows'
    elif multiple_rows:
        # Prevent an artificially lowered entropy from ignoring repeated rows.
        entropy_status = 'unresolved_repeated_values'
    elif len(included) < min_samples:
        entropy_status = 'insufficient_samples'
    elif any(cell['status'] == 'present' and
             type(cell['value']) not in (bool, int, float, str) for cell in included):
        entropy_status = 'unsupported_non_scalar_values'
    elif floats_seen and bins is None:
        entropy_status = 'float_requires_explicit_binning'
    else:
        if bins is not None:
            if any(cell['status'] == 'present' and
                   type(cell['value']) not in (int, float) for cell in included):
                entropy_status = 'binning_requires_numeric_values'
            else:
                numbers = [_numeric(cell['value'], variable_id) for cell in included
                           if cell['status'] == 'present']
                fitted = _binning_definition(numbers, variable_id, bins)
        if entropy_status == 'ok':
            for cell in included:
                if cell['status'] == 'empty':
                    encoded = ('state', 'empty')
                elif fitted is not None:
                    encoded = ('bin', _bin_index(_numeric(cell['value'], variable_id), fitted))
                else:
                    encoded = _categorical(cell['value'], variable_id)
                entropy_states[encoded] += 1
            entropy_value = _entropy(entropy_states, len(included), 2.0)
            if not math.isfinite(entropy_value):
                raise StructuredContractError('atomic entropy exceeded finite numeric range')
    distribution = [{**seen_states[k], 'count': count} for k, count in sorted(states.items())]
    return {
        'schema': ATOMIC_SCHEMA,
        'source_digest': source_digest, 'binding_digest': binding_sha256(bound),
        'node_id': node_id, 'variable_id': variable_id,
        'occurrence_policy': occurrence_policy, 'observation_unit': source.observation_unit,
        'node_context': context,
        'support': {
            'aligned_rows': len(rows), 'observed_scalar_rows': len(included),
            'unavailable_rows': len(rows) - len(included), 'repeated_rows': multiple_rows,
            'independence_verified': False, 'node_count_is_not_sample_count': True,
        },
        'status_counts': dict(sorted(statuses.items())),
        'status_rates': {key: count / len(rows) for key, count in sorted(statuses.items())} if rows else {},
        'original_binding_evidence': {
            'original_status_counts': original_variable_evidence['original_status_counts'],
            'null_source_reason_counts': original_variable_evidence['null_source_reason_counts'],
            'repeated_rows': original_variable_evidence['repeated_rows'],
            'recorded_occurrences_in_repeated_rows': original_variable_evidence['recorded_occurrences_in_repeated_rows'],
        },
        'missing_reason_counts': dict(sorted(reasons.items())),
        'scalar_distribution': distribution,
        'scalar_distribution_excludes_repeated_rows': bool(multiple_rows),
        'numeric': {'status': numeric_status, 'statistics': numeric,
                    'excludes_unresolved_repeated_rows': bool(multiple_rows),
                    'excluded_nonnumeric_scalar_rows': numeric_incompatible},
        'entropy': {
            'status': entropy_status, 'bits': entropy_value,
            'support': len(included) if entropy_status == 'ok' else None,
            'distinct_states': len(entropy_states) if entropy_status == 'ok' else None,
            'min_samples': min_samples, 'estimator': 'empirical_discrete_plugin',
            'missing_policy': 'complete_case', 'discretization': fitted,
            'finite_sample_bias_warning': True,
        },
        'provenance': {'binding_plan': plan.to_dict(), 'source_object_id': source.object_id,
                       'statistical_edges_inferred_from_structure': False,
                       'repeated_values_never_silently_aggregated': True,
                       'node_mapping_is_user_declared': True},
        'limitations': [
            'One original structural node is not an independent observation.',
            'Finite-sample plug-in entropy is descriptive and generally biased.',
            'Missing and unresolved multiple rows are not assigned numeric zero.',
            'MI/CMI and correlation with neighboring nodes belong to HIA-05.',
        ],
    }
