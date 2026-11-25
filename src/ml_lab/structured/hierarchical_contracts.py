"""HIA-01: versioned requests and *eligibility*, not information estimates.

This module has no IO, renderer, estimator, project state or background work.
An original source edge does not establish statistical dependence; a node is
not a sample. HIA-02 supplies actual level membership, HIA-04--10 estimators.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from typing import Any

from .bindings import ObservationBindingResult, RESULT_SCHEMA as BINDING_SCHEMA, source_sha256
from .branch_uncertainty import binding_sha256
from .contracts import StructuredContractError
from .object import StructuredObject

REQUEST_SCHEMA = 'ml-lab.hierarchical-information-request@1'
ELIGIBILITY_SCHEMA = 'ml-lab.hierarchical-information-eligibility@1'
SCOPE_KINDS = frozenset({'node', 'branch', 'level', 'network'})
LEVEL_POLICIES = frozenset({'rooted_depth', 'dag_longest_path', 'none'})
NEIGHBOR_BASES = frozenset({'structural', 'statistical'})
CONDITIONING_SCOPES = frozenset({'none', 'prior_levels', 'explicit_variables'})
INDEPENDENCE_MODELS = frozenset({'empirical_joint', 'conditioned_factorization'})
# Metric identifiers specify separate estimands; availability is NOT a value.
METRIC_SCOPES = {
    'node_structure': frozenset({'node'}),
    'node_descriptive': frozenset({'node'}),
    'node_neighbor_associations': frozenset({'node'}),
    'branch_structure': frozenset({'branch'}),
    'branch_joint_entropy': frozenset({'branch'}),
    'level_joint_entropy': frozenset({'level'}),
    'level_conditional_entropy': frozenset({'level'}),
    'level_target_increment': frozenset({'level'}),
    'level_factorized_conditional_entropy': frozenset({'level'}),
    'network_structure': frozenset({'network'}),
    'network_information': frozenset({'network'}),
}
STRUCTURAL_METRICS = frozenset({'node_structure', 'branch_structure', 'network_structure'})
_CONDITIONAL_METRICS = frozenset({
    'level_conditional_entropy', 'level_target_increment',
    'level_factorized_conditional_entropy',
})
_SHA_LEN = 71


def _error(message: str) -> None:
    raise StructuredContractError(message)


def _digest(value: object, label: str) -> str:
    if (not isinstance(value, str) or len(value) != _SHA_LEN or
            not value.startswith('sha256:') or
            any(c not in '0123456789abcdef' for c in value[7:])):
        _error(f'{label} must be a lowercase sha256 fingerprint')
    return value


def _names(value: object, label: str) -> tuple[str, ...]:
    if (not isinstance(value, tuple) or
            any(not isinstance(item, str) or not item.strip() for item in value) or
            len(value) != len(set(value))):
        _error(f'{label} must be a tuple of unique nonempty identities')
    return value


def _canonical(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False)


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            _error(f'duplicate HIA-01 JSON key: {key}')
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class HierarchicalScope:
    """Exactly one original node, a rooted branch, a depth, or the network."""

    kind: str
    node_ids: tuple[str, ...] = ()
    level: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in SCOPE_KINDS:
            _error('unknown HIA scope kind')
        _names(self.node_ids, 'scope.node_ids')
        if self.kind in ('node', 'branch'):
            if len(self.node_ids) != 1 or self.level is not None:
                _error(f'{self.kind} scope requires exactly one node ID and no level')
        elif self.kind == 'level':
            if self.node_ids or type(self.level) is not int or self.level < 0:
                _error('level scope requires a nonnegative integer depth and no node IDs')
        elif self.node_ids or self.level is not None:
            _error('network scope cannot specify node IDs or depth')

    def to_dict(self) -> dict[str, Any]:
        return {'kind': self.kind, 'node_ids': list(self.node_ids), 'level': self.level}

    @classmethod
    def from_dict(cls, raw: object) -> 'HierarchicalScope':
        if not isinstance(raw, Mapping) or set(raw) != {'kind', 'node_ids', 'level'}:
            _error('invalid HIA scope fields')
        if not isinstance(raw['node_ids'], list):
            _error('scope.node_ids must be an array')
        return cls(raw['kind'], tuple(raw['node_ids']), raw['level'])


@dataclass(frozen=True, slots=True)
class HierarchicalAnalysisRequest:
    """Read-only scientific intention pinned to an immutable source and binding.

    No source relationship is converted to a statistical or probabilistic edge.
    A factorization is valid only when the user opts into the explicit model.
    """

    source_digest: str
    scope: HierarchicalScope
    metrics: tuple[str, ...]
    binding_digest: str | None = None
    level_policy: str = 'none'
    neighbor_basis: str = 'structural'
    conditioning_scope: str = 'none'
    conditioning_variables: tuple[str, ...] = ()
    target_variable: str | None = None
    independence_model: str = 'empirical_joint'
    acknowledge_assumption: bool = False

    def __post_init__(self) -> None:
        _digest(self.source_digest, 'source_digest')
        if not isinstance(self.scope, HierarchicalScope):
            _error('scope must be a validated HierarchicalScope')
        if self.binding_digest is not None:
            _digest(self.binding_digest, 'binding_digest')
        _names(self.metrics, 'metrics')
        if not self.metrics or set(self.metrics) - set(METRIC_SCOPES):
            _error('metrics must contain only supported nonempty identifiers')
        if any(self.scope.kind not in METRIC_SCOPES[name] for name in self.metrics):
            _error('metric incompatible with selected structural scope')
        if self.level_policy not in LEVEL_POLICIES:
            _error('unknown hierarchy level policy')
        if self.neighbor_basis not in NEIGHBOR_BASES:
            _error('unknown neighbor basis')
        if self.conditioning_scope not in CONDITIONING_SCOPES:
            _error('unknown conditioning scope')
        _names(self.conditioning_variables, 'conditioning_variables')
        if self.conditioning_scope != 'explicit_variables' and self.conditioning_variables:
            _error('explicit conditioning variables require explicit_variables scope')
        if self.conditioning_scope == 'explicit_variables' and not self.conditioning_variables:
            _error('explicit_variables scope requires conditioning variables')
        if self.conditioning_scope == 'prior_levels' and self.scope.kind != 'level':
            _error('prior_levels conditioning requires a level scope')
        if self.target_variable is not None and (
                not isinstance(self.target_variable, str) or not self.target_variable.strip()):
            _error('target_variable must be a nonempty ID or null')
        if self.target_variable in self.conditioning_variables:
            _error('target cannot also be an explicit conditioning variable')
        if 'level_target_increment' in self.metrics and self.target_variable is None:
            _error('incremental target relevance requires an explicit target variable')
        if self.target_variable is not None and 'level_target_increment' not in self.metrics:
            _error('target_variable is only valid with level_target_increment')
        if set(self.metrics) & _CONDITIONAL_METRICS and self.conditioning_scope == 'none':
            _error('conditional metrics require an explicit conditioning scope')
        if self.independence_model not in INDEPENDENCE_MODELS:
            _error('unknown independence model')
        if type(self.acknowledge_assumption) is not bool:
            _error('acknowledge_assumption must be boolean')
        if self.independence_model == 'conditioned_factorization':
            if (not self.acknowledge_assumption or self.scope.kind != 'level' or
                    self.conditioning_scope != 'prior_levels' or
                    'level_factorized_conditional_entropy' not in self.metrics):
                _error('conditioned factorization needs level/prior_levels, explicit consent and its metric')
        elif self.acknowledge_assumption or 'level_factorized_conditional_entropy' in self.metrics:
            _error('factorized entropy requires the explicit conditioned_factorization model')
        if self.scope.kind == 'level' and self.level_policy == 'none':
            _error('level scope requires an explicit deterministic hierarchy policy')

    def to_dict(self) -> dict[str, Any]:
        return {'schema': REQUEST_SCHEMA, 'source_digest': self.source_digest,
                'binding_digest': self.binding_digest, 'scope': self.scope.to_dict(),
                'metrics': list(self.metrics), 'level_policy': self.level_policy,
                'neighbor_basis': self.neighbor_basis,
                'conditioning_scope': self.conditioning_scope,
                'conditioning_variables': list(self.conditioning_variables),
                'target_variable': self.target_variable,
                'independence_model': self.independence_model,
                'acknowledge_assumption': self.acknowledge_assumption}

    def to_json(self) -> str:
        return _canonical(self.to_dict())

    @classmethod
    def from_dict(cls, raw: object) -> 'HierarchicalAnalysisRequest':
        fields = {'schema', 'source_digest', 'binding_digest', 'scope', 'metrics',
                  'level_policy', 'neighbor_basis', 'conditioning_scope',
                  'conditioning_variables', 'target_variable', 'independence_model',
                  'acknowledge_assumption'}
        if not isinstance(raw, Mapping) or set(raw) != fields or raw.get('schema') != REQUEST_SCHEMA:
            _error('invalid HIA-01 request schema or fields')
        if not isinstance(raw['metrics'], list) or not isinstance(raw['conditioning_variables'], list):
            _error('metrics and conditioning_variables must be arrays')
        return cls(raw['source_digest'], HierarchicalScope.from_dict(raw['scope']),
                   tuple(raw['metrics']), raw['binding_digest'], raw['level_policy'],
                   raw['neighbor_basis'], raw['conditioning_scope'],
                   tuple(raw['conditioning_variables']), raw['target_variable'],
                   raw['independence_model'], raw['acknowledge_assumption'])

    @classmethod
    def from_json(cls, text: str) -> 'HierarchicalAnalysisRequest':
        if not isinstance(text, str):
            _error('HIA-01 request must be JSON text')
        try:
            return cls.from_dict(json.loads(text, object_pairs_hook=_pairs,
                parse_constant=lambda value: _error(f'invalid JSON constant: {value}')))
        except (TypeError, ValueError, RecursionError, UnicodeError) as exc:
            raise StructuredContractError(f'invalid HIA-01 request JSON: {exc}') from exc


def inspect_hierarchical_eligibility(
    source: StructuredObject, request: HierarchicalAnalysisRequest,
    binding: ObservationBindingResult | None = None,
) -> dict[str, Any]:
    """Verify scope, revision and evidence availability; do not compute statistics.

    Each metric is explicitly deferred or unavailable. No invented numerical
    value, empirical relationship or graph level is returned in HIA-01.
    """
    if not isinstance(source, StructuredObject) or not isinstance(request, HierarchicalAnalysisRequest):
        _error('eligibility requires a StructuredObject and HIA request')
    if source_sha256(source) != request.source_digest:
        _error('HIA source fingerprint mismatch')
    node_ids = {node['id'] for node in source.nodes}
    if any(node_id not in node_ids for node_id in request.scope.node_ids):
        _error('HIA scope refers to unknown original source node')
    if request.scope.kind == 'branch' and 'rooted_hierarchy' not in source.capabilities:
        _error('branch scope requires verified rooted tree/forest hierarchy')
    if request.scope.kind == 'level':
        if source.topology['kind'] in ('tree', 'forest'):
            expected_policy = 'rooted_depth'
        elif source.topology['kind'] == 'dag':
            expected_policy = 'dag_longest_path'
        else:
            _error('level scope unavailable for cyclic/undirected/general graphs')
        if request.level_policy != expected_policy:
            _error(f'{source.topology["kind"]} level scope requires {expected_policy}')
    elif request.level_policy != 'none':
        _error('level policy belongs only to a level scope')
    if request.conditioning_scope == 'prior_levels' and request.scope.level == 0:
        _error('level zero has no previous levels to condition on')

    if binding is not None:
        if not isinstance(binding, ObservationBindingResult):
            _error('HIA binding must be an SU-05 ObservationBindingResult')
        doc = binding.to_dict()
        if doc.get('schema') != BINDING_SCHEMA or doc.get('source_digest') != request.source_digest:
            _error('HIA binding source mismatch')
        if request.binding_digest is None or binding_sha256(binding) != request.binding_digest:
            _error('HIA binding fingerprint mismatch')
    elif request.binding_digest is not None:
        _error('source-pinned binding missing for HIA request')
    else:
        doc = None
    all_variables = {entry['variable_id'] for entry in doc['variables']} if doc else set()
    # Never substitute labels or source attributes for SU-05 mapped variables.
    for name in (*request.conditioning_variables, *((request.target_variable,)
                  if request.target_variable else ())):
        if doc is not None and name not in all_variables:
            _error(f'unknown SU-05 bound observation variable: {name}')
    if request.conditioning_scope == 'explicit_variables' and doc is None:
        _error('explicit conditioning needs a verified SU-05 binding')
    if request.target_variable and doc is None:
        _error('target information needs a verified SU-05 binding')
    if request.scope.kind in ('node', 'branch') and doc:
        if request.scope.kind == 'branch':
            branch = next((item for item in (doc.get('branches') or ())
                           if item['node_id'] == request.scope.node_ids[0]), None)
            mapped = set(branch['subtree_variable_ids']) if branch else set()
        else:
            mapped = {entry['variable_id'] for entry in doc['variables']
                      if request.scope.node_ids[0] in entry['node_ids']}
        ambiguous = {entry['variable_id'] for entry in doc['variables']
                     if entry['variable_id'] in mapped and entry['merged_node_identity_ambiguous']}
    else:
        mapped = all_variables
        ambiguous = set()

    evidence: list[dict[str, Any]] = []
    for metric in request.metrics:
        if metric in STRUCTURAL_METRICS:
            status, reason = 'deferred', 'HIA-02_hierarchy_or_HIA-03_inspector_not_implemented'
        elif doc is None:
            status, reason = 'unavailable', 'verified_observation_binding_required'
        elif not doc['rows']:
            status, reason = 'unavailable', 'no_aligned_observations'
        elif request.scope.kind in ('node', 'branch') and not mapped:
            status, reason = 'unavailable', 'no_directly_bound_observation_variables'
        elif ambiguous:
            status, reason = 'unavailable', 'merged_source_node_attribution_ambiguous'
        elif request.scope.kind == 'level':
            status, reason = 'deferred', 'HIA-02_level_assignment_required'
        else:
            status, reason = 'deferred', 'HIA_scientific_estimator_not_implemented'
        evidence.append({'metric': metric, 'status': status, 'reason': reason,
                         'estimate': None, 'estimator_support': None})
    return {
        'schema': ELIGIBILITY_SCHEMA, 'source_digest': request.source_digest,
        'binding_digest': request.binding_digest, 'scope': request.scope.to_dict(),
        'level_policy': request.level_policy,
        'source_topology': source.topology['kind'],
        'observation_unit': source.observation_unit,
        'observed_rows_available': len(doc['rows']) if doc else 0,
        'mapped_variable_ids': sorted(mapped) if doc else [],
        'ambiguous_variable_ids': sorted(ambiguous),
        'assumption': {'model': request.independence_model,
                       'acknowledged': request.acknowledge_assumption,
                       'causal_interpretation': False,
                       'independence_established_by_topology': False},
        'metrics': evidence,
        'source_graph_is_statistical_graph': False,
        'statistical_estimates_computed': False,
        'source_modified': False,
    }
