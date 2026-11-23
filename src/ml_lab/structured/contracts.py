"""ML-UW-27: scientific and eligibility contracts (no analysis engine).

Capabilities are *evidence asserted and validated by a future StructuredObject*.
Neither this module nor callers should infer probability, causality, variable
alignment, or hierarchical semantics from a source graph's edges.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

CONTRACT_VERSION = 1
CONTRACT_SCHEMA = 'ml-lab.structured-object@1'
TOPOLOGIES = frozenset({'tree', 'forest', 'dag', 'directed_graph', 'undirected_graph'})
OBSERVATION_UNITS = frozenset({'source', 'record', 'node', 'graph'})
# Domain-neutral normalized cell states. Adapters retain source-specific
# reason codes (e.g. XML xsi:nil or missing attributes) as provenance metadata.
# No source-specific XML vocabulary is required by scientific core functions.
CELL_STATES = frozenset({'present', 'empty', 'missing', 'explicit_null', 'unknown', 'multiple'})
NULL_REASONS = frozenset({'missing', 'explicit_null', 'unknown'})
CAPABILITIES = frozenset({
    'graph_edges', 'directed_edges', 'acyclic', 'rooted_hierarchy',
    'unique_parent', 'ordered_children', 'aligned_observations',
    'explicit_probability_model',
})
# Prerequisites are minimum conditions; individual estimators may impose more.
# The source topology never implies aligned observations or a probabilistic model.
OPERATION_REQUIREMENTS: Mapping[str, frozenset[str]] = {
    'graph_summary': frozenset({'graph_edges'}),
    'hierarchy_summary': frozenset({'rooted_hierarchy', 'unique_parent', 'acyclic'}),
    'depth_reduction': frozenset({'rooted_hierarchy', 'unique_parent', 'acyclic'}),
    'conditional_information': frozenset({'aligned_observations'}),
    'hierarchical_probability_inference': frozenset({
        'directed_edges', 'acyclic', 'aligned_observations', 'explicit_probability_model',
    }),
}


class StructuredContractError(ValueError):
    """An object or analysis request fails a declared scientific prerequisite."""


def _set_of_names(names: Iterable[str], *, allowed: frozenset[str], field: str) -> frozenset[str]:
    if isinstance(names, (str, bytes)):
        raise StructuredContractError(f'{field} must be a collection of names')
    try:
        collection = tuple(names)
    except TypeError as exc:
        raise StructuredContractError(f'{field} must be iterable') from exc
    if any(not isinstance(name, str) for name in collection):
        raise StructuredContractError(f'{field} contains a non-string name')
    invalid = sorted(set(collection) - allowed)
    if invalid:
        raise StructuredContractError(f'unsupported {field}: {", ".join(invalid)}')
    return frozenset(collection)


def require_capabilities(declared: Iterable[str], operation: str) -> None:
    """Reject unverified operations; never silently simplify the input graph.

    This is the baseline gate only. ML-UW-28 must validate declarations against
    concrete topology; ML-UW-31--34 must check statistical support separately.
    """
    if operation not in OPERATION_REQUIREMENTS:
        raise StructuredContractError(f'unknown operation: {operation!r}')
    given = _set_of_names(declared, allowed=CAPABILITIES, field='capabilities')
    missing = OPERATION_REQUIREMENTS[operation] - given
    if missing:
        raise StructuredContractError(
            f'{operation} requires capabilities: {", ".join(sorted(missing))}'
        )


def require_observation_alignment(
    observation_unit: str,
    observations: Sequence[Mapping[str, object]],
    required_variables: Iterable[str],
) -> None:
    """Require explicitly paired rows, not merely two nodes in one graph.

    Each row identifies its own observation ID and contains a value cell for
    every requested variable. Missing values may be represented by a null-state
    cell, but absent variable identities or duplicate sample IDs are invalid.
    This does not certify estimator support or sample independence.
    """
    if observation_unit not in OBSERVATION_UNITS:
        raise StructuredContractError('unsupported or missing observation unit')
    if isinstance(required_variables, (str, bytes)):
        raise StructuredContractError('variables must be a collection of names')
    try:
        requested = tuple(required_variables)
    except TypeError as exc:
        raise StructuredContractError('variables must be iterable') from exc
    if any(not isinstance(name, str) or not name for name in requested):
        raise StructuredContractError('variable names must be nonempty strings')
    variables = frozenset(requested)
    if not variables:
        raise StructuredContractError('at least one variable must be specified')
    if (not isinstance(observations, Sequence) or isinstance(observations, (str, bytes))
            or not observations):
        raise StructuredContractError('at least one aligned observation is required')
    seen: set[str] = set()
    for index, row in enumerate(observations):
        if not isinstance(row, Mapping):
            raise StructuredContractError(f'row {index} must be a mapping')
        obs_id = row.get('observation_id')
        if not isinstance(obs_id, str) or not obs_id or obs_id in seen:
            raise StructuredContractError('observation IDs must be nonempty and unique')
        seen.add(obs_id)
        values = row.get('values')
        if not isinstance(values, Mapping) or not variables.issubset(values):
            raise StructuredContractError(f'row {obs_id} lacks aligned variables')
        for key in variables:
            cell = values[key]
            if not isinstance(cell, Mapping) or cell.get('status') not in CELL_STATES:
                raise StructuredContractError(f'row {obs_id}: invalid state for {key}')
            status, value = cell['status'], cell.get('value')
            if status in NULL_REASONS and value is not None:
                raise StructuredContractError(f'row {obs_id}: null reason must have null value')
            if status == 'empty' and value != '':
                raise StructuredContractError(f'row {obs_id}: empty must have empty string value')
            if status == 'present' and value is None:
                raise StructuredContractError(f'row {obs_id}: present value cannot be null')
            if status == 'multiple' and not isinstance(cell.get('occurrences'), list):
                raise StructuredContractError(f'row {obs_id}: multiple requires occurrences')
