"""SU-10: explicit, read-only orchestration of SU-05--09 for a selected node.

No model fitting, implicit joins, artifact mutation, database access, or graph
renderer logic. An absent estimate is represented as unavailable, never zero.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ml_lab.structured import (
    StructuredContractError, StructuredObject, ObservationBindingPlan,
    bind_observations, binding_sha256, source_sha256,
    BranchUncertaintyPlan, analyze_branch_uncertainty,
    BranchInformationPlan, analyze_branch_information,
    StructuralCandidatePlan, apply_structural_candidate,
    StructuralRetentionStudy, compare_structural_retention,
)
from .structured_graph_view import make_shared_graph_view

SCHEMA = 'ml-lab.structural-sandbox@1'
_ALLOWED = frozenset({'source_revision', 'mode', 'node_id', 'binding_plan',
    'uncertainty_targets', 'group_levels', 'information_target', 'retained_variables',
    'information_bins', 'missing_policy', 'include_information', 'operation', 'max_depth',
    'observation_policy', 'retention'})
_MAX_ROWS = 10_000
_MAX_EXAMPLES = 12


def _fail(message: str) -> None:
    raise StructuredContractError(message)


def _names(value: Any, field: str, *, required: bool = False) -> tuple[str, ...]:
    if (not isinstance(value, list) or len(value) > 100 or
            any(not isinstance(x, str) or not x.strip() or len(x) > 256 for x in value) or
            len(set(value)) != len(value) or (required and not value)):
        _fail(f'{field}: expected unique nonempty variable-ID array')
    return tuple(value)


def _optional_name(value: Any, field: str) -> str | None:
    if value is None or value == '':
        return None
    if not isinstance(value, str) or len(value) > 256 or not value.strip():
        _fail(f'{field}: expected a nonempty variable ID or null')
    return value


def _unavailable(reason: str) -> dict[str, Any]:
    return {'status': 'unavailable', 'reason': reason}


def _binding(source: StructuredObject, raw: object):
    if raw is None:
        return None, None
    if not isinstance(raw, dict):
        _fail('binding_plan: supply the complete SU-05 JSON object')
    plan = ObservationBindingPlan.from_dict(raw)
    bound = bind_observations(source, plan)
    return plan, bound


def _binding_evidence(bound, node_id: str) -> dict[str, Any]:
    data = bound.to_dict()
    branches = data['branches']
    if branches is not None:
        matching = next((item for item in branches if item['node_id'] == node_id), None)
    else:
        matching = None
    definitions = data['variables']
    direct = sorted(item['variable_id'] for item in definitions if node_id in item['node_ids'])
    scoped = (matching['subtree_variable_ids'] if matching is not None else direct)
    displayed = list(scoped)
    ambiguous = sorted(item['variable_id'] for item in definitions
                       if item['merged_node_identity_ambiguous'] and
                       item['variable_id'] in displayed)
    observed_rows = []
    for row in data['rows'][:_MAX_EXAMPLES]:
        observed_rows.append({'observation_id': row['observation_id'],
                              'values': {key: row['values'][key] for key in displayed}})
    return {
        'status': 'available', 'scope': 'subtree' if matching is not None else 'direct_only',
        'direct_variable_ids': direct, 'scope_variable_ids': displayed,
        'ambiguous_merged_variable_ids': ambiguous,
        'sample_unit': data['observation_unit'],
        'aligned_rows': len(data['rows']), 'total_rows': len(data['rows']),
        'rows_displayed': len(observed_rows), 'rows_truncated': len(data['rows']) > _MAX_EXAMPLES,
        'example_observations': observed_rows,
        'counts': {item['variable_id']: {
            'cell_status': item['original_status_counts'],
            'null_reasons': item['null_source_reason_counts'],
        } for item in definitions if item['variable_id'] in displayed},
        'limitations': ['source_node_is_not_a_sampling_unit',
                        'ambiguous_merges_do_not_allocate_observations_to_individual_nodes',
                        'repeated_values_are_not_new_independent_rows'],
    }



def _graph_overlay(report: dict[str, Any], node_id: str) -> dict[str, Any]:
    """PAH_Shared graph-view@1 descriptive node overlay, never edge attribution.

    A variable-population variance is displayed only for a single declared
    target within this node's explicit mapped scope; it is *not* a share of
    total variance explained by the source graph.
    """
    metrics: list[dict[str, Any]] = []
    info = report['information']
    if info.get('status') == 'available':
        block = info['node']['information']
        support = block.get('support') or {}
        count = support.get('included_rows')
        for name, part in (
            ('candidate_joint_entropy', (block.get('entropy') or {}).get('candidate_joint')),
            ('target_candidate_mi', (block.get('relevance') or {}).get('target_candidate_mi')),
            ('target_candidate_cmi', (block.get('relevance') or {}).get('target_candidate_cmi_given_retained')),
            ('candidate_retained_redundancy_mi', block.get('redundancy_candidate_retained_mi')),
        ):
            if isinstance(part, dict):
                value = part.get('value') if part.get('status') == 'ok' else None
                metrics.append({'name': name, 'unit': 'bits',
                    'estimator': 'empirical-discrete-plugin', 'support': count,
                    'status': 'measured' if value is not None else 'unavailable',
                    'value': value})
    uncertainty = report['uncertainty']
    if uncertainty.get('status') == 'available':
        scoped = uncertainty['selected_node']
        targets = uncertainty['targets']
        if (isinstance(scoped, dict) and 'scope_variable_ids' in scoped and
                len(targets) == 1):
            target = next(iter(targets))
            if target in scoped['scope_variable_ids']:
                summary = targets[target]['raw']
                value = summary.get('population_variance')
                metrics.append({'name': 'observed_variable_population_variance',
                    'unit': 'observed_variable_squared',
                    'estimator': 'weighted-descriptive-population',
                    'support': summary.get('rows'),
                    'status': 'measured' if value is not None else 'unavailable',
                    'value': value})
    return {'node_id': node_id, 'overlay': {
        'status': 'measured' if any(item['status'] == 'measured' for item in metrics)
                  else 'unavailable',
        'metrics': metrics,
    }, 'interpretation': 'explicit_observed_variable_scope_only_no_edge_attribution'}


def analyze_sandbox(source: StructuredObject, *, source_revision: str,
                    request_data: Mapping[str, Any]) -> dict[str, Any]:
    """Analyze only after an explicit POST with exact SHA-verified source.

    The caller must re-read project-local bytes and compare the raw source SHA;
    this function validates the additional complete StructuredObject digest.
    """
    if not isinstance(source, StructuredObject):
        _fail('verified StructuredObject required')
    if not isinstance(request_data, Mapping) or set(request_data) - _ALLOWED:
        _fail('unsupported sandbox fields')
    if (not isinstance(source_revision, str) or len(source_revision) != 71 or
            not source_revision.startswith('sha256:') or
            any(c not in '0123456789abcdef' for c in source_revision[7:]) or
            request_data.get('source_revision') != source_revision):
        _fail('source revision mismatch: reopen the original graph')
    mode = request_data.get('mode')
    if mode not in ('inspect', 'preview'):
        _fail('mode must be inspect or preview')
    node_id = _optional_name(request_data.get('node_id'), 'node_id')
    node_ids = {item['id'] for item in source.nodes}
    if node_id not in node_ids:
        _fail('select an original source node before requesting branch evidence')
    if len(source.observations) > _MAX_ROWS:
        _fail('interactive sandbox exceeds 10000 aligned rows; use headless analysis')
    digest = source_sha256(source)
    binding_plan, binding = _binding(source, request_data.get('binding_plan'))
    bound_digest = binding_sha256(binding) if binding is not None else None

    out = {'schema': SCHEMA, 'source_revision': source_revision,
           'source_digest': digest, 'node_id': node_id, 'mode': mode,
           'source_topology': source.topology['kind'],
           'binding_digest': bound_digest,
           'observation_evidence': (_binding_evidence(binding, node_id) if binding else
                                    _unavailable('explicit_SU_05_binding_required')),
           'uncertainty': _unavailable('not_requested'),
           'information': _unavailable('not_requested'),
           'candidate': _unavailable('not_requested'),
           'retention': _unavailable('not_requested'),
           'limitations': ['no_implicit_sampling_or_independence_assumption',
                           'structural_edges_not_statistical_or_causal_edges',
                           'source_and_observation_values_unchanged']}

    targets = _names(request_data.get('uncertainty_targets', []), 'uncertainty_targets')
    groups = _names(request_data.get('group_levels', []), 'group_levels')
    if targets:
        if binding is None:
            out['uncertainty'] = _unavailable('explicit_SU_05_binding_required')
        else:
            plan = BranchUncertaintyPlan(bound_digest, targets, group_levels=groups)
            report = analyze_branch_uncertainty(binding, plan).to_dict()
            chosen = next((item for item in report['nodes'] if item['node_id'] == node_id), None)
            out['uncertainty'] = {'status': 'available', 'support': report['support'],
                                   'selected_node': chosen or _unavailable('node_not_in_binding_scope'),
                                   'variables': report['variables'],
                                   'targets': report['targets'],
                                   'group_support': report['group_support'],
                                   'limitations': report['limitations']}
    target = _optional_name(request_data.get('information_target'), 'information_target')
    retained = _names(request_data.get('retained_variables', []), 'retained_variables')
    bins = request_data.get('information_bins', [])
    if (not isinstance(bins, list) or len(bins) > 100 or
            any(not isinstance(entry, list) or len(entry) != 3 for entry in bins)):
        _fail('information_bins must be an array of [variable, mode, n_bins]')
    missing = request_data.get('missing_policy', 'complete_case')
    include_information = request_data.get('include_information', False)
    if type(include_information) is not bool:
        _fail('include_information must be boolean')
    if include_information or target or retained or bins:
        if binding is None:
            out['information'] = _unavailable('explicit_SU_05_binding_required')
        else:
            inventory = binding.to_dict()
            eligible = (any(item['node_id'] == node_id for item in inventory['branches'])
                        if inventory['branches'] is not None else
                        any(node_id in item['node_ids'] for item in inventory['variables']))
            if not eligible:
                out['information'] = _unavailable('node_not_in_binding_scope')
            else:
                plan = BranchInformationPlan(bound_digest, nodes=(node_id,), target=target,
                         retained_variables=retained, bins=tuple(tuple(x) for x in bins),
                         missing_policy=missing)
                report = analyze_branch_information(binding, plan).to_dict()
                out['information'] = {'status': 'available', 'estimator': report['estimator'],
                                      'support': report['support'],
                                      'node': report['nodes'][0],
                                      'overlay': report['association_layer']['node_overlays'][0],
                                      'limitations': report['limitations']}
    if mode == 'preview':
        operation = request_data.get('operation')
        if not isinstance(operation, str):
            _fail('select an SU-08 candidate operation')
        raw_depth = request_data.get('max_depth')
        if raw_depth is not None and (type(raw_depth) is not int or not 0 <= raw_depth <= 10000):
            _fail('max_depth must be a nonnegative integer')
        policy = request_data.get('observation_policy', 'preserve')
        candidate_plan = StructuralCandidatePlan(digest, operation,
            node_ids=(node_id,) if operation in ('prune', 'summarize') else (),
            max_depth=raw_depth if operation in ('aggregate', 'depth_reduce') else None,
            observation_policy=policy, binding_digest=bound_digest)
        result = apply_structural_candidate(source, candidate_plan, binding=binding_plan)
        manifest = result.manifest
        after = make_shared_graph_view(result.object,
                           fingerprint=manifest['output_digest'].removeprefix('sha256:'))
        removed = manifest['structural_loss']['removed_node_ids']
        retained_ids = {n['id'] for n in result.object.nodes}
        out['candidate'] = {
            'status': 'available', 'manifest': manifest, 'view': after,
            'selection_survives': node_id in retained_ids,
            'highlight': {'removed_node_ids': removed,
                          'retained_selected_node_id': node_id if node_id in retained_ids else None,
                          'collapsed_into': manifest['structural_loss']['collapsed_into']},
            'source_unchanged': True,
        }
        study_raw = request_data.get('retention')
        if study_raw is not None:
            if binding is None:
                out['retention'] = _unavailable('explicit_SU_05_binding_required')
            elif operation == 'preserve':
                out['retention'] = _unavailable('candidate_matches_reference')
            elif not isinstance(study_raw, dict):
                _fail('retention must be a complete SU-09 options object')
            else:
                keys = {'predictors', 'kind', 'target', 'variance_variables',
                        'reconstruction_variables', 'group_variable',
                        'acknowledge_independent_rows', 'seed', 'holdout_fraction'}
                if set(study_raw) - keys:
                    _fail('unrecognized retention study options')
                preserve = StructuralCandidatePlan(digest, 'preserve', binding_digest=bound_digest)
                study = StructuralRetentionStudy(digest, bound_digest, (preserve, candidate_plan),
                    predictors=_names(study_raw.get('predictors', []), 'predictors', required=True),
                    kind=study_raw.get('kind', 'target_free'),
                    target=_optional_name(study_raw.get('target'), 'retention target'),
                    variance_variables=_names(study_raw.get('variance_variables', []), 'variance_variables'),
                    reconstruction_variables=_names(study_raw.get('reconstruction_variables', []), 'reconstruction_variables'),
                    group_variable=_optional_name(study_raw.get('group_variable'), 'group_variable'),
                    acknowledge_independent_rows=study_raw.get('acknowledge_independent_rows', False),
                    seed=study_raw.get('seed', 17),
                    holdout_fraction=study_raw.get('holdout_fraction', 0.25))
                report = compare_structural_retention(source, binding_plan, study).to_dict()
                out['retention'] = {'status': 'available', 'report': report,
                                    'note': 'Matched holdout; candidate comparison is exploratory, not external validation'}
    out['graph_overlay'] = _graph_overlay(out, node_id)
    return out
