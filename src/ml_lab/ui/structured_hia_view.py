"""HIA-03 read-only web projection for an exact, source-pinned hierarchy.

Presentation metadata and paged member controls only.  No analysis, storage,
implicit observation bindings, or derivation of statistical edges from topology.
"""
from __future__ import annotations

from ml_lab.structured import (
    HierarchicalAnalysisRequest, HierarchicalScope, StructuredContractError,
    StructuredObject, build_hierarchy_index, inspect_hierarchical_selection,
    source_sha256,
)

OVERVIEW_SCHEMA = 'ml-lab.hia-workspace-overview@1'
INSPECTION_SCHEMA = 'ml-lab.hia-workspace-inspection@1'
PAGE_SIZE = 80  # Only a view-page size, never a cap on total membership.


def hierarchy_overview(source: StructuredObject, *, source_revision: str) -> dict:
    """Return level counts without duplicating the complete source graph."""
    if not isinstance(source_revision, str) or not source_revision.startswith('sha256:'):
        raise StructuredContractError('HIA workspace needs a source revision')
    digest = source_sha256(source)
    basic = {
        'schema': OVERVIEW_SCHEMA,
        'source_revision': source_revision,
        'source_digest': digest,
        'topology': source.topology['kind'],
        'node_count': len(source.nodes),
        'edge_count': len(source.edges),
        'source_edges_are_statistical_dependencies': False,
    }
    if source.topology['kind'] not in ('tree', 'forest', 'dag'):
        return {**basic, 'available': False, 'reason': 'levels_require_verified_tree_forest_or_dag',
                'level_policy': None, 'roots': [], 'levels': [], 'branch_selection_allowed': False}
    index = build_hierarchy_index(source)
    return {
        **basic, 'available': True, 'reason': None,
        'level_policy': index.level_policy, 'roots': list(index.roots),
        'levels': [{'depth': i, 'node_count': len(nodes)} for i, nodes in enumerate(index.levels)],
        'branch_selection_allowed': index.topology in ('tree', 'forest'),
    }


def inspect_workspace_scope(source: StructuredObject, *, source_revision: str,
                            scope_input: object, offset: object = 0) -> dict:
    """Inspect an explicit scope and return paged buttons plus full identity metadata.

    `node_ids` remain exhaustive in the verified HIA-02 selection report; the
    UI chooses at most PAGE_SIZE rendered buttons at a time.  This does not
    sample, substitute summary node identities, or estimate numerical metrics.
    """
    if type(offset) is not int or offset < 0:
        raise StructuredContractError('HIA page offset must be a nonnegative integer')
    scope = HierarchicalScope.from_dict(scope_input)
    index = build_hierarchy_index(source)
    kind = scope.kind
    metrics = {
        'node': ('node_structure', 'node_descriptive', 'node_neighbor_associations'),
        'branch': ('branch_structure', 'branch_joint_entropy'),
        'level': ('level_joint_entropy',),
        'network': ('network_structure', 'network_information'),
    }[kind]
    request = HierarchicalAnalysisRequest(
        source_digest=source_sha256(source), scope=scope, metrics=metrics,
        level_policy=index.level_policy if kind == 'level' else 'none',
    )
    selection = inspect_hierarchical_selection(source, request, index=index)
    members = selection['node_ids']
    # UI offsets are position-based and only page the buttons, never the source.
    first = min(offset, len(members))
    page_ids = members[first:first + PAGE_SIZE]
    extra = {'node_context': index.node_context(scope.node_ids[0]) if kind == 'node' else None}
    return {
        'schema': INSPECTION_SCHEMA,
        'source_revision': source_revision,
        'source_digest': index.source_digest,
        'level_policy': index.level_policy,
        'scope': scope.to_dict(),
        'selection': selection,
        'member_page': {'offset': first, 'page_size': PAGE_SIZE,
                        'total': len(members), 'node_ids': page_ids,
                        'has_previous': first > 0,
                        'has_next': first + len(page_ids) < len(members)},
        'node_context': extra['node_context'],
        'metrics_are_estimates': False,
        'source_edges_are_statistical_dependencies': False,
    }
