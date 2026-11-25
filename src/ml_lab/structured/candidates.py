"""SU-08: deterministic, source-pinned structural transformation candidates.

This is a headless *preview and replay* layer over ML-UW-30 and SU-05.  The
original StructuredObject never changes.  Structural edges are not statistical
edges, and a removed node does not imply that an observed column disappeared.
No sampling, estimation, automatic optimization, IO or UI is performed here.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from typing import Any

from .bindings import ObservationBindingPlan, bind_observations, source_sha256
from .branch_uncertainty import binding_sha256
from .contracts import StructuredContractError
from .object import StructuredObject
from .transformations import TransformationPlan, transform

PLAN_SCHEMA = 'ml-lab.structural-candidate-plan@1'
MANIFEST_SCHEMA = 'ml-lab.structural-candidate-manifest@1'
OPERATIONS = frozenset({'preserve', 'prune', 'aggregate', 'depth_reduce',
                        'summarize', 'flatten'})
OBSERVATION_POLICIES = frozenset({'preserve', 'drop_exclusive'})
_MAX_CANDIDATES = 128
_MAX_NODES = 10_000
_MAX_EDGES = 30_000


def _fail(message: str) -> None:
    raise StructuredContractError(message)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def _hash(data: str) -> str:
    return 'sha256:' + hashlib.sha256(data.encode('utf-8')).hexdigest()


def _fingerprint(value: object, name: str) -> None:
    if (not isinstance(value, str) or not value.startswith('sha256:') or
            len(value) != 71 or any(c not in '0123456789abcdef' for c in value[7:])):
        _fail(f'{name} must be a lowercase SHA-256 fingerprint')


def _ids(items: object, name: str) -> None:
    if (not isinstance(items, tuple) or
            len(set(value for value in items if isinstance(value, str))) != len(items) or
            any(not isinstance(value, str) or not value.strip() for value in items)):
        _fail(f'{name} must be a tuple of unique nonempty IDs')


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in items:
        if key in result:
            _fail(f'duplicate candidate JSON field: {key}')
        result[key] = value
    return result


def _json_loads(value: str) -> Any:
    if not isinstance(value, str):
        _fail('candidate JSON must be text')
    try:
        return json.loads(value, object_pairs_hook=_pairs,
                          parse_constant=lambda x: _fail(f'nonfinite JSON constant: {x}'))
    except (TypeError, ValueError, RecursionError, UnicodeError) as exc:
        raise StructuredContractError(f'invalid candidate JSON: {exc}') from exc


@dataclass(frozen=True, slots=True)
class StructuralCandidatePlan:
    """An explicit operation pinned to the entire validated original source.

    Node IDs select *whole* rooted branches for prune, or retained boundary
    nodes for summarize.  max_depth is absolute from source hierarchy roots.
    A binding digest is mandatory before removing any observed variable.
    """

    source_digest: str
    operation: str
    node_ids: tuple[str, ...] = ()
    max_depth: int | None = None
    observation_policy: str = 'preserve'
    binding_digest: str | None = None

    def __post_init__(self) -> None:
        _fingerprint(self.source_digest, 'source_digest')
        if not isinstance(self.operation, str) or self.operation not in OPERATIONS:
            _fail(f'unsupported candidate operation: {self.operation!r}')
        _ids(self.node_ids, 'node_ids')
        if self.operation in {'prune', 'summarize'}:
            if not self.node_ids:
                _fail(f'{self.operation} requires explicitly selected node_ids')
        elif self.node_ids:
            _fail('node_ids are only valid for prune and summarize')
        if self.operation in {'aggregate', 'depth_reduce'}:
            if type(self.max_depth) is not int or self.max_depth < 0:
                _fail('aggregate and depth_reduce require a nonnegative max_depth')
        elif self.max_depth is not None:
            _fail('max_depth is only valid for aggregate and depth_reduce')
        if not isinstance(self.observation_policy, str) or self.observation_policy not in OBSERVATION_POLICIES:
            _fail('unsupported observation_policy')
        if self.binding_digest is not None:
            _fingerprint(self.binding_digest, 'binding_digest')
        if self.observation_policy == 'drop_exclusive' and self.binding_digest is None:
            _fail('drop_exclusive requires an explicit SU-05 binding_digest')

    def to_dict(self) -> dict[str, Any]:
        return {'schema': PLAN_SCHEMA, 'source_digest': self.source_digest,
                'operation': self.operation, 'node_ids': list(self.node_ids),
                'max_depth': self.max_depth,
                'observation_policy': self.observation_policy,
                'binding_digest': self.binding_digest}

    def to_json(self) -> str:
        return _canonical(self.to_dict())

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> 'StructuralCandidatePlan':
        required = {'schema', 'source_digest', 'operation', 'node_ids', 'max_depth',
                    'observation_policy', 'binding_digest'}
        if not isinstance(raw, Mapping) or set(raw) != required or raw.get('schema') != PLAN_SCHEMA:
            _fail('invalid candidate plan schema or fields')
        if not isinstance(raw['node_ids'], list):
            _fail('candidate node_ids must be an array')
        return cls(raw['source_digest'], raw['operation'], tuple(raw['node_ids']),
                   raw['max_depth'], raw['observation_policy'], raw['binding_digest'])

    @classmethod
    def from_json(cls, text: str) -> 'StructuralCandidatePlan':
        return cls.from_dict(_json_loads(text))


@dataclass(frozen=True, slots=True)
class StructuralCandidateResult:
    """Transformed preview plus detached, canonical, replay-verifiable audit."""

    object: StructuredObject
    plan: StructuralCandidatePlan
    _manifest_json: str

    @property
    def manifest(self) -> dict[str, Any]:
        return json.loads(self._manifest_json)

    def to_dict(self) -> dict[str, Any]:
        return {'manifest': self.manifest, 'object': self.object.to_dict()}

    def to_json(self) -> str:
        return _canonical(self.to_dict())


def _validated_binding(source: StructuredObject,
                       binding: ObservationBindingPlan | None) -> tuple[str | None, dict | None]:
    if binding is None:
        return None, None
    if not isinstance(binding, ObservationBindingPlan):
        _fail('binding must be an SU-05 ObservationBindingPlan')
    # This checks the complete source fingerprint, actual source-node identities,
    # observation columns, grouping/weight columns, and occurrence policies.
    bound = bind_observations(source, binding)
    return binding_sha256(bound), binding.to_dict()


def _hierarchy(source: StructuredObject) -> tuple[dict[str, list[str]], dict[str, str]]:
    source.require('depth_reduction')
    children: dict[str, list[str]] = defaultdict(list)
    parents: dict[str, str] = {}
    for edge in source.edges:
        children[edge['source_id']].append(edge['target_id'])
        parents[edge['target_id']] = edge['source_id']
    return children, parents


def _branch_nodes(selected: tuple[str, ...], children: Mapping[str, list[str]],
                  parents: Mapping[str, str], all_nodes: set[str], *, include_roots: bool
                  ) -> set[str]:
    if set(selected) - all_nodes:
        _fail(f'unknown source node IDs: {", ".join(sorted(set(selected) - all_nodes))}')
    # Reject overlap rather than silently interpreting repeated/nested actions.
    chosen = set(selected)
    for item in selected:
        cursor = item
        while cursor in parents:
            cursor = parents[cursor]
            if cursor in chosen:
                _fail('selected branches overlap; select disjoint branch roots')
    removed: set[str] = set()
    queue = deque(selected)
    while queue:
        node = queue.popleft()
        if node in removed:
            continue
        removed.add(node)
        queue.extend(children[node])
    if not include_roots:
        removed.difference_update(chosen)
    return removed


def _build_branch_candidate(source: StructuredObject, plan: StructuralCandidatePlan,
                            ) -> tuple[StructuredObject, dict[str, Any]]:
    children, parents = _hierarchy(source)
    original = source.to_dict()
    ids = {node['id'] for node in original['nodes']}
    removed = _branch_nodes(plan.node_ids, children, parents, ids,
                            include_roots=plan.operation == 'prune')
    retained = ids - removed
    if not retained:
        _fail('candidate would remove every structural node; no empty StructuredObject')
    doc = source.to_dict()
    doc.pop('object_id', None)
    doc['nodes'] = [node for node in doc['nodes'] if node['id'] in retained]
    doc['edges'] = [edge for edge in doc['edges']
                    if edge['source_id'] in retained and edge['target_id'] in retained]
    collapsed_into: dict[str, str] = {}
    if plan.operation == 'summarize':
        kinds = {node['id']: node['kind'] for node in original['nodes']}
        for node in doc['nodes']:
            if node['id'] not in plan.node_ids:
                continue
            descendants = _branch_nodes((node['id'],), children, parents, ids,
                                        include_roots=False)
            collapsed_into.update({item: node['id'] for item in descendants})
            removed_edges = [edge for edge in original['edges'] if
                             edge['source_id'] in descendants or edge['target_id'] in descendants]
            metadata = dict(node.get('metadata', {}))
            key = 'ml_lab_su08_structural_summary_v1'
            if key in metadata:
                _fail('structural summary metadata collision; refusing overwrite')
            metadata[key] = {
                'removed_nodes': len(descendants),
                'removed_edges': len(removed_edges),
                'node_kinds': dict(sorted(Counter(kinds[item] for item in descendants).items())),
                'value_aggregation': 'none',
                'source_nodes_are_not_observations': True,
            }
            node['metadata'] = metadata
    if plan.operation == 'prune':
        roots = [node for node in retained if node not in parents]
        doc['topology'] = {'kind': 'tree' if len(roots) == 1 else 'forest'}
    else:
        doc['topology'] = {'kind': original['topology']['kind']}
    provenance = doc['provenance']
    history = provenance.get('structural_candidates', [])
    if not isinstance(history, list):
        _fail('invalid structural_candidates provenance history')
    provenance['structural_candidates'] = history + [
        {'schema': PLAN_SCHEMA, 'source_digest': plan.source_digest, 'plan': plan.to_dict()}]
    output = StructuredObject.from_dict(doc)
    return output, {'removed_node_ids': sorted(removed),
                    'collapsed_into': dict(sorted(collapsed_into.items())),
                    'removed_edge_ids': sorted(set(e['id'] for e in original['edges']) -
                                               set(e['id'] for e in doc['edges']))}


def _project_variables(object_: StructuredObject, retained: list[str]
                       ) -> StructuredObject:
    if not object_.observations:
        _fail('cannot project variables without aligned observations')
    if not retained:
        _fail('candidate would remove all aligned variables')
    doc = object_.to_dict()
    doc.pop('object_id', None)
    for row in doc['observations']:
        row['values'] = {name: row['values'][name] for name in retained}
    return StructuredObject.from_dict(doc)


def apply_structural_candidate(source: StructuredObject, plan: StructuralCandidatePlan,
                               *, binding: ObservationBindingPlan | None = None
                               ) -> StructuralCandidateResult:
    """Preview one candidate; fail closed on stale identity or unsupported topology.

    ``drop_exclusive`` is an explicit opt-in. Only variables whose *every*
    declared source node was removed can be dropped. Variables with mixed
    source membership or no binding remain intact, without claiming attribution.
    Every observed record retains its original observation ID and sampling unit.
    """
    if not isinstance(source, StructuredObject) or not isinstance(plan, StructuralCandidatePlan):
        _fail('candidate preview requires a StructuredObject and StructuralCandidatePlan')
    if len(source.nodes) > _MAX_NODES or len(source.edges) > _MAX_EDGES:
        _fail('candidate exceeds explicit node/edge budget; no silent truncation')
    if source_sha256(source) != plan.source_digest:
        _fail('source fingerprint mismatch: stale candidate plan')
    binding_digest, bound = _validated_binding(source, binding)
    if plan.binding_digest != binding_digest:
        _fail('binding fingerprint mismatch: stale or unprovided binding')
    if plan.operation in {'prune', 'summarize'}:
        output, audit = _build_branch_candidate(source, plan)
    else:
        modes = {'preserve': 'full', 'aggregate': 'aggregate',
                 'depth_reduce': 'depth', 'flatten': 'flat'}
        legacy = TransformationPlan(modes[plan.operation], max_depth=plan.max_depth)
        transformed = transform(source, legacy)
        output = transformed.object
        audit = transformed.audit
    source_doc = source.to_dict()
    source_vars = list(source_doc['observations'][0]['values']) if source_doc['observations'] else []
    before_nodes = {node['id'] for node in source_doc['nodes']}
    after_nodes = {node['id'] for node in output.nodes}
    removed_nodes = sorted(before_nodes - after_nodes)
    removed_edges = sorted({edge['id'] for edge in source_doc['edges']} -
                           {edge['id'] for edge in output.edges})
    touched = []
    fully_removed = []
    mixed = []
    unbound = source_vars[:]
    if bound is not None:
        bound_ids = set()
        for rule in bound['variables']:
            name, linked = rule['variable_id'], set(rule['node_ids'])
            bound_ids.add(name)
            if not linked <= after_nodes:
                touched.append(name)
                if linked.isdisjoint(after_nodes):
                    fully_removed.append(name)
                else:
                    mixed.append(name)
        unbound = sorted(set(source_vars) - bound_ids)
    protected = set(bound['group_variables']) if bound is not None else set()
    if bound is not None and bound['weight_variable'] is not None:
        protected.add(bound['weight_variable'])
    dropped: list[str] = []
    if plan.observation_policy == 'drop_exclusive':
        if bound is None or not source_vars:
            _fail('drop_exclusive requires a verified binding and aligned observations')
        if set(fully_removed) & protected:
            _fail('cannot drop a bound group or weight variable')
        dropped = sorted(set(fully_removed) & set(source_vars))
        if dropped:
            output = _project_variables(output, [name for name in source_vars if name not in dropped])
    output_vars = list(output.observations[0]['values']) if output.observations else []
    original_ids = [row['observation_id'] for row in source_doc['observations']]
    observed_ids = [row['observation_id'] for row in output.observations]
    if original_ids != observed_ids:
        _fail('candidate must retain exactly the source observation identities and row order')
    # Verify that every retained cell is *exactly* the source cell; only an
    # explicit feature projection is allowed, never synthetic aggregation.
    for original, derived in zip(source_doc['observations'], output.to_dict()['observations']):
        if any(derived['values'][name] != original['values'][name] for name in derived['values']):
            _fail('candidate changed the source observation cell values')
    structurally_changed = bool(removed_nodes or removed_edges or
                                output.topology['kind'] != source.topology['kind'])
    manifest = {
        'schema': MANIFEST_SCHEMA,
        'candidate_id': _hash(plan.to_json()),
        'source_digest': plan.source_digest,
        'source_object_id': source.object_id,
        'binding_digest': binding_digest,
        'output_digest': source_sha256(output),
        'output_object_id': output.object_id,
        'plan': plan.to_dict(),
        'topology': {'before': source.topology['kind'], 'after': output.topology['kind']},
        'structural_loss': {
            'removed_node_ids': removed_nodes, 'removed_edge_ids': removed_edges,
            'collapsed_into': dict(sorted(audit.get('collapsed_into', {}).items())),
            'removed_node_count': len(removed_nodes),
            'removed_edge_count': len(removed_edges),
            'retained_node_count': len(output.nodes),
            'retained_edge_count': len(output.edges),
            'topology_changed': source.topology['kind'] != output.topology['kind'],
            'structurally_changed': structurally_changed,
        },
        'feature_impact': {
            'input_variable_ids': sorted(source_vars),
            'output_variable_ids': sorted(output_vars),
            'source_nodes_removed_for_bound_variables': sorted(touched),
            'fully_removed_source_mapping_variables': sorted(fully_removed),
            'mixed_retained_removed_source_mapping_variables': sorted(mixed),
            'unbound_variable_ids': sorted(unbound),
            'protected_group_or_weight_variables': sorted(protected),
            'dropped_variable_ids': dropped,
            'dropped_variable_count': len(dropped),
            'observed_cell_values_changed': False,
            'sampling_unit_unchanged': True,
            'observation_count': len(original_ids),
            'observation_join_performed': False,
            'empirical_information_loss_estimated': False,
        },
        'structural_summary_only': plan.operation in {'summarize', 'aggregate'},
        'lossless_representation': not structurally_changed and not dropped,
        'output_alone_can_reconstruct_source': plan.operation == 'preserve' and not dropped,
        'notes': [
            'source_graph_relationships_are_not_statistical_or_causal_edges',
            'feature_impact_uses_explicit_SU_05_node_bindings_not_inferred_joins',
            'branch_aggregation_never_aggregates_observation_values',
            'variance_information_and_task_utility_are_not_estimated_by_SU_08',
            'a_replayable_manifest_is_not_a_lossless_inverse_of_dropped_data',
        ],
    }
    return StructuralCandidateResult(output, plan, _canonical(manifest))


def replay_structural_candidate(source: StructuredObject, manifest: Mapping[str, Any] | str,
                                *, binding: ObservationBindingPlan | None = None
                                ) -> StructuralCandidateResult:
    """Require exact manifest agreement after recomputation from original source."""
    document = _json_loads(manifest) if isinstance(manifest, str) else manifest
    required = {'schema', 'candidate_id', 'source_digest', 'source_object_id',
                'binding_digest', 'output_digest', 'output_object_id', 'plan',
                'topology', 'structural_loss', 'feature_impact',
                'structural_summary_only', 'lossless_representation',
                'output_alone_can_reconstruct_source', 'notes'}
    if not isinstance(document, Mapping) or set(document) != required or document.get('schema') != MANIFEST_SCHEMA:
        _fail('unsupported or incomplete structural-candidate manifest')
    plan = StructuralCandidatePlan.from_dict(document['plan'])
    result = apply_structural_candidate(source, plan, binding=binding)
    if _canonical(document) != _canonical(result.manifest):
        _fail('candidate replay manifest mismatch: source, plan, output or audit was changed')
    return result


def enumerate_structural_candidates(
    source: StructuredObject, *, binding: ObservationBindingPlan | None = None,
    node_ids: tuple[str, ...] = (), depths: tuple[int, ...] = (),
    operations: tuple[str, ...] = ('preserve', 'prune', 'aggregate',
                                 'depth_reduce', 'summarize', 'flatten'),
    observation_policy: str = 'preserve', max_candidates: int = 64,
) -> tuple[StructuralCandidatePlan, ...]:
    """Deterministic *plans* only; never materialize datasets or scan all branches.

    Only specifically selected node IDs/depths are proposed. Unsupported
    hierarchy-only operations are omitted from enumeration (but rejected if
    explicitly applied). The budget is enforced, never silently truncated.
    """
    if not isinstance(source, StructuredObject):
        _fail('source must be a validated StructuredObject')
    _ids(node_ids, 'node_ids')
    if not isinstance(depths, tuple) or any(type(d) is not int or d < 0 for d in depths):
        _fail('depths must be a tuple of nonnegative integer depths')
    if (not isinstance(operations, tuple) or any(not isinstance(op, str) or op not in OPERATIONS for op in operations) or
            len(set(op for op in operations if isinstance(op, str))) != len(operations)):
        _fail('operations must be a tuple of unique recognized operations')
    if type(max_candidates) is not int or not 1 <= max_candidates <= _MAX_CANDIDATES:
        _fail(f'max_candidates must be between 1 and {_MAX_CANDIDATES}')
    if not isinstance(observation_policy, str) or observation_policy not in OBSERVATION_POLICIES:
        _fail('unsupported observation_policy')
    if len(source.nodes) > _MAX_NODES or len(source.edges) > _MAX_EDGES:
        _fail('candidate exceeds explicit node/edge budget; no silent truncation')
    valid = {node['id'] for node in source.nodes}
    if set(node_ids) - valid:
        _fail('selected node IDs must exist in source')
    binding_digest, _ = _validated_binding(source, binding)
    if observation_policy == 'drop_exclusive' and binding_digest is None:
        _fail('drop_exclusive requires a verified SU-05 binding')
    hierarchical = 'rooted_hierarchy' in source.capabilities
    fingerprint = source_sha256(source)
    plans: list[StructuralCandidatePlan] = []
    # A tree's only root cannot be pruned; do not enumerate an impossible
    # candidate. A forest root can be pruned if another root remains.
    root_of_whole_tree = None
    if hierarchical and source.topology['kind'] == 'tree':
        parents = {edge['target_id'] for edge in source.edges}
        root_of_whole_tree = next(node['id'] for node in source.nodes if node['id'] not in parents)
    # Fixed operation ordering regardless of caller input, so enum is stable.
    for operation in ('preserve', 'prune', 'aggregate', 'depth_reduce', 'summarize', 'flatten'):
        if operation not in operations:
            continue
        if operation in {'prune', 'aggregate', 'depth_reduce', 'summarize'} and not hierarchical:
            continue
        if operation in {'prune', 'summarize'}:
            for node in sorted(node_ids):
                if operation == 'prune' and node == root_of_whole_tree:
                    continue
                plans.append(StructuralCandidatePlan(fingerprint, operation, (node,),
                             observation_policy=observation_policy, binding_digest=binding_digest))
        elif operation in {'aggregate', 'depth_reduce'}:
            for depth in sorted(set(depths)):
                plans.append(StructuralCandidatePlan(fingerprint, operation,
                             max_depth=depth, observation_policy=observation_policy,
                             binding_digest=binding_digest))
        else:
            plans.append(StructuralCandidatePlan(fingerprint, operation,
                         observation_policy=observation_policy, binding_digest=binding_digest))
        if len(plans) > max_candidates:
            _fail('candidate enumeration exceeds explicit maximum; narrow node/depth selection')
    return tuple(plans)
