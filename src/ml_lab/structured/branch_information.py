"""SU-07: empirical branch information over explicitly bound observation rows.

Structure determines *which variables belong to a branch*, never what constitutes
an observation or a stochastic/causal dependency. All calculations use the
SU-05 pinned projection and the ML-UW-32 empirical-discrete estimator primitives.
No graph renderer, source artifact, persistent store, or previous estimator is
modified by this module.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from typing import Any

from .bindings import ObservationBindingResult, RESULT_SCHEMA as BINDING_SCHEMA
from .branch_uncertainty import binding_sha256
from .contracts import StructuredContractError
from .information import (
    BINNING_MODES, _bin_index, _binning_definition, _categorical,
    _conditional_entropy, _conditional_mi, _entropy, _numeric,
)

PLAN_SCHEMA = 'ml-lab.branch-information-plan@1'
RESULT_SCHEMA = 'ml-lab.branch-information@1'
ASSOCIATION_SCHEMA = 'ml-lab.statistical-node-overlay@1'
MISSING_POLICIES = frozenset({'complete_case', 'as_category', 'error'})
_MAX_NODES = 256
_MAX_ROWS = 50_000
_MAX_WORK = 5_000_000  # selected-node * aligned-row comparisons; no hidden subsampling
_MAX_COLUMNS_PER_ESTIMATE = 12
_MAX_JOINT_STATES = 5_000


def _fail(message: str) -> None:
    raise StructuredContractError(message)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False)


def _names(items: Any, label: str) -> None:
    if (not isinstance(items, tuple) or
            len(set(x for x in items if isinstance(x, str))) != len(items) or
            any(not isinstance(x, str) or not x.strip() for x in items)):
        _fail(f'{label} must be a tuple of unique nonempty names')


@dataclass(frozen=True, slots=True)
class BranchInformationPlan:
    """Declarative SU-07 analysis; no implicit feature/target/retained inference.

    ``nodes=()`` selects every available mapped source node (up to the declared
    cost cap). ``bins`` gives (variable, equal_width|quantile, n_bins) triples;
    floats must be explicitly binned. All categorical integers are literal
    categories unless discretization was requested for that variable.
    """

    binding_digest: str
    nodes: tuple[str, ...] = ()
    target: str | None = None
    retained_variables: tuple[str, ...] = ()
    bins: tuple[tuple[str, str, int], ...] = ()
    missing_policy: str = 'complete_case'
    min_samples: int = 2
    min_stratum_samples: int = 2
    log_base: float = 2.0

    def __post_init__(self) -> None:
        # Reuse the SU-06/SU-05 source fingerprint contract.
        if (not isinstance(self.binding_digest, str) or len(self.binding_digest) != 71 or
                not self.binding_digest.startswith('sha256:') or
                any(c not in '0123456789abcdef' for c in self.binding_digest[7:])):
            _fail('binding_digest must be a lowercase SHA-256 fingerprint')
        _names(self.nodes, 'nodes')
        _names(self.retained_variables, 'retained_variables')
        if self.target is not None and (not isinstance(self.target, str) or not self.target.strip()):
            _fail('target must be a nonempty variable name or null')
        if self.target in self.retained_variables:
            _fail('target cannot be a retained predictor')
        if not isinstance(self.missing_policy, str) or self.missing_policy not in MISSING_POLICIES:
            _fail('unsupported missing_policy')
        if type(self.min_samples) is not int or not 2 <= self.min_samples <= _MAX_ROWS:
            _fail('min_samples must be an integer between 2 and 50000')
        if type(self.min_stratum_samples) is not int or not 2 <= self.min_stratum_samples <= _MAX_ROWS:
            _fail('min_stratum_samples must be an integer between 2 and 50000')
        if type(self.log_base) not in (int, float):
            _fail('log_base must be finite and > 1')
        try:
            base = float(self.log_base)
        except (ValueError, OverflowError) as exc:
            raise StructuredContractError('log_base exceeds the finite numeric range') from exc
        if not math.isfinite(base) or base <= 1:
            _fail('log_base must be finite and > 1')
        if not isinstance(self.bins, tuple):
            _fail('bins must be a tuple of (variable, mode, n_bins) triples')
        found: set[str] = set()
        for spec in self.bins:
            if (not isinstance(spec, tuple) or len(spec) != 3 or
                    not isinstance(spec[0], str) or not spec[0].strip() or
                    spec[0] in found or not isinstance(spec[1], str) or spec[1] not in BINNING_MODES or
                    type(spec[2]) is not int or not 2 <= spec[2] <= 128):
                _fail('invalid or duplicate discretization specification')
            found.add(spec[0])

    def to_dict(self) -> dict[str, Any]:
        return {'schema': PLAN_SCHEMA, 'binding_digest': self.binding_digest,
                'nodes': list(self.nodes), 'target': self.target,
                'retained_variables': list(self.retained_variables),
                'bins': [list(spec) for spec in self.bins],
                'missing_policy': self.missing_policy,
                'min_samples': self.min_samples,
                'min_stratum_samples': self.min_stratum_samples,
                'log_base': self.log_base}

    def to_json(self) -> str:
        return _json(self.to_dict())

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> 'BranchInformationPlan':
        required = {'schema', 'binding_digest', 'nodes', 'target', 'retained_variables',
                    'bins', 'missing_policy', 'min_samples', 'min_stratum_samples', 'log_base'}
        if not isinstance(raw, Mapping) or set(raw) != required or raw.get('schema') != PLAN_SCHEMA:
            _fail('invalid SU-07 plan schema or fields')
        if any(not isinstance(raw[key], list) for key in ('nodes', 'retained_variables', 'bins')):
            _fail('nodes, retained_variables and bins must be arrays')
        if any(not isinstance(item, list) or len(item) != 3 for item in raw['bins']):
            _fail('bins must contain three-element arrays')
        return cls(raw['binding_digest'], tuple(raw['nodes']), raw['target'],
                   tuple(raw['retained_variables']), tuple(tuple(item) for item in raw['bins']),
                   raw['missing_policy'], raw['min_samples'], raw['min_stratum_samples'],
                   raw['log_base'])

    @classmethod
    def from_json(cls, value: str) -> 'BranchInformationPlan':
        if not isinstance(value, str):
            _fail('SU-07 plan must be JSON text')
        def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
            out = {}
            for key, entry in items:
                if key in out:
                    _fail(f'duplicate SU-07 JSON field: {key}')
                out[key] = entry
            return out
        try:
            return cls.from_dict(json.loads(value, object_pairs_hook=pairs,
                 parse_constant=lambda name: _fail(f'invalid JSON constant {name}')))
        except (ValueError, TypeError, RecursionError) as exc:
            raise StructuredContractError(f'invalid SU-07 plan JSON: {exc}') from exc


@dataclass(frozen=True, slots=True)
class BranchInformationResult:
    _json: str

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._json)

    def to_json(self) -> str:
        return self._json


def _inventory(bound: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Trust only declared SU-05 scope; never traverse a non-tree as a subtree."""
    variables = bound['variables']
    by_node: dict[str, set[str]] = {}
    for item in variables:
        for node in item['node_ids']:
            by_node.setdefault(node, set()).add(item['variable_id'])
    branches = bound['branches']
    if branches is None:
        return {node: {'node_id': node, 'scope': 'direct_only',
                        'direct_variable_ids': sorted(values),
                        'scope_variable_ids': sorted(values)}
                for node, values in sorted(by_node.items())}
    return {item['node_id']: {'node_id': item['node_id'], 'scope': 'subtree',
            'direct_variable_ids': item['direct_variable_ids'],
            'scope_variable_ids': item['subtree_variable_ids']}
            for item in branches}


def _discretizations(rows: list[dict[str, Any]], definitions: tuple[tuple[str, str, int], ...]
                     ) -> dict[str, dict[str, Any]]:
    """Fit fixed descriptive bin edges once, over all available aligned rows.

    A per-node complete-case subset never silently refits quantile cutpoints.
    Bins use all available data, so they MUST be fitted only on training data
    if an eventual predictive comparison wants to prevent leakage.
    """
    output = {}
    for name, mode, n_bins in definitions:
        values = [_numeric(row['values'][name]['value'], name) for row in rows
                  if row['values'][name]['status'] == 'present']
        output[name] = _binning_definition(values, name, {'mode': mode, 'bins': n_bins})
    return output


def _encode(cell: Mapping[str, Any], variable: str,
            spec: dict[str, Any] | None) -> tuple[str, Any]:
    state = cell['status']
    if state != 'present':
        return ('state', state)
    if spec is not None:
        if spec['actual_bins'] < 1:
            _fail(f'{variable}: cannot discretize present observations with no fitted bins')
        return ('bin', _bin_index(_numeric(cell['value'], variable), spec))
    return _categorical(cell['value'], variable)


def _metric(value: float | None, status: str, **details: Any) -> dict[str, Any]:
    if value is not None and not math.isfinite(value):
        _fail('nonfinite information estimate')
    return {'status': status, 'value': value, **details}


def _unavailable(reason: str) -> dict[str, Any]:
    return _metric(None, reason)


def _node_information(rows: list[dict[str, Any]], scope: tuple[str, ...],
                      candidates: tuple[str, ...], plan: BranchInformationPlan,
                      specifications: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    retained = plan.retained_variables
    target = plan.target
    required = tuple(dict.fromkeys((*scope, *retained, *((target,) if target else ()))))
    exclusions: Counter[str] = Counter()
    states = {name: Counter() for name in required}
    source_reasons = {name: Counter() for name in required}
    included: list[dict[str, tuple[str, Any]]] = []
    for row in rows:
        cells = row['values']
        missing = []
        for name in required:
            state = cells[name]['status']
            states[name][state] += 1
            if cells[name].get('source_reason'):
                source_reasons[name][cells[name]['source_reason']] += 1
            if state == 'multiple':
                # A repeat is not an extra independent sample or a category.
                _fail(f'{name}: repeated observations need explicit SU-05 scalar/mean aggregation before SU-07')
            if state in {'missing', 'explicit_null', 'unknown'}:
                missing.append((name, state))
        if missing and plan.missing_policy == 'error':
            _fail('missing observation under error policy')
        if missing and plan.missing_policy == 'complete_case':
            for name, state in missing:
                exclusions[f'{name}:{state}'] += 1
            continue
        included.append({name: _encode(cells[name], name, specifications.get(name))
                         for name in required})

    n = len(included)
    support = {'aligned_rows': len(rows), 'included_rows': n,
               'excluded_rows': len(rows) - n,
               'exclusion_reasons': dict(sorted(exclusions.items())),
               'min_samples': plan.min_samples,
               'min_stratum_samples': plan.min_stratum_samples,
               'sampling_independence_verified': False,
               'cell_status_counts': {k: dict(sorted(v.items())) for k, v in states.items()},
               'null_source_reasons': {k: dict(sorted(v.items())) for k, v in source_reasons.items()}}
    keys = {
        'scope': [tuple(record[name] for name in scope) for record in included],
        'candidate': [tuple(record[name] for name in candidates) for record in included],
        'retained': [tuple(record[name] for name in retained) for record in included],
    }
    if target:
        keys['target'] = [tuple([record[target]]) for record in included]
    # Shared node-level complete-case denominator for every returned score.
    all_keys = [tuple(record[name] for name in required) for record in included]
    joint_counts = Counter(all_keys)
    support['distinct_joint_states'] = len(joint_counts)
    support['singleton_joint_states'] = sum(v == 1 for v in joint_counts.values())
    retained_count = Counter(keys['retained']) if retained else Counter()
    support['conditioning_strata'] = len(retained_count) if retained else None
    support['under_supported_strata'] = (sum(c < plan.min_stratum_samples for c in retained_count.values())
                                          if retained else None)
    support['distinct_states'] = {name: len({row[name] for row in included}) for name in required}
    warnings = ['finite_sample_plugin_bias_uncorrected',
                'no_independence_guarantee_for_aligned_observations']
    if support['singleton_joint_states']:
        warnings.append('sparse_joint_states')
    if n and len(joint_counts) * 5 >= n:
        warnings.append('high_joint_cardinality_relative_to_sample_count')
    if n < plan.min_samples:
        status = 'insufficient_samples'
    elif len(joint_counts) > _MAX_JOINT_STATES:
        status = 'joint_state_limit'
    else:
        status = 'ok'
    def available(fn: Any, *, conditional: bool = False) -> dict[str, Any]:
        if status != 'ok':
            return _unavailable(status)
        if conditional and support['under_supported_strata']:
            return _unavailable('insufficient_conditioning_support')
        value = fn()
        return _metric(value, 'ok')

    entropy = {
        'scope_joint': available(lambda: _entropy(Counter(keys['scope']), n, plan.log_base)),
        'candidate_joint': available(lambda: _entropy(Counter(keys['candidate']), n, plan.log_base)),
        'candidate_given_retained': available(
            lambda: _conditional_entropy(list(zip(keys['candidate'], keys['retained'])), plan.log_base),
            conditional=bool(retained)),
    }
    if target:
        relevance = {
            'target_scope_mi': available(lambda: _conditional_mi(
                list(zip(keys['target'], keys['scope'], [()] * n)), plan.log_base)),
            'target_candidate_mi': available(lambda: _conditional_mi(
                list(zip(keys['target'], keys['candidate'], [()] * n)), plan.log_base)),
            'target_candidate_cmi_given_retained': available(lambda: _conditional_mi(
                list(zip(keys['target'], keys['candidate'], keys['retained'])), plan.log_base),
                conditional=bool(retained)),
            'target_entropy_given_retained': available(lambda: _conditional_entropy(
                list(zip(keys['target'], keys['retained'])), plan.log_base),
                conditional=bool(retained)),
        }
    else:
        relevance = {name: _unavailable('no_target_declared') for name in (
            'target_scope_mi', 'target_candidate_mi', 'target_candidate_cmi_given_retained',
            'target_entropy_given_retained')}
    redundancy = (available(lambda: _conditional_mi(
        list(zip(keys['candidate'], keys['retained'], [()] * n)), plan.log_base))
        if retained else _unavailable('no_retained_variables'))
    if retained and support['under_supported_strata']:
        warnings.append('conditioning_strata_below_minimum')
    if not retained:
        warnings.append('incremental_target_mi_equals_unconditioned_target_candidate_mi')
    return {'status': status, 'support': support, 'entropy': entropy,
            'relevance': relevance, 'redundancy_candidate_retained_mi': redundancy,
            'warnings': warnings}


def analyze_branch_information(binding: ObservationBindingResult,
                               plan: BranchInformationPlan) -> BranchInformationResult:
    """Return source-linked, descriptive branch association *node overlays*.

    No edges are created between source nodes; zero MI never authorizes prune.
    Conditional MI is I(Y; candidate | explicitly retained variables), where
    the candidate excludes retained variables and target itself.
    """
    if not isinstance(binding, ObservationBindingResult) or not isinstance(plan, BranchInformationPlan):
        _fail('SU-07 requires an ObservationBindingResult and BranchInformationPlan')
    if binding_sha256(binding) != plan.binding_digest:
        _fail('binding fingerprint mismatch: stale SU-05 observations')
    data = binding.to_dict()
    if data['schema'] != BINDING_SCHEMA:
        _fail('unsupported SU-05 observation binding schema')
    definitions = {item['variable_id']: item for item in data['variables']}
    names = set(definitions)
    if plan.target is not None and plan.target not in names:
        _fail('target must be an explicitly bound observed variable')
    if not set(plan.retained_variables) <= names:
        _fail('retained_variables must be explicitly bound observed variables')
    if not {name for name, _, _ in plan.bins} <= names:
        _fail('discretization may refer only to explicitly bound variables')
    rows = data['rows']
    if len(rows) > _MAX_ROWS:
        _fail('SU-07 exceeds 50000 aligned observations; no implicit sampling')
    if any(row['weight'] != 1.0 for row in rows):
        _fail('SU-07 empirical discrete estimator requires unit record weights; nonunit sampling weights are not ignored or replicated')
    branch_inventory = _inventory(data)
    if plan.nodes and not set(plan.nodes) <= branch_inventory.keys():
        _fail('nodes must be exact source node IDs present in SU-05 branch inventory')
    selected = sorted(plan.nodes) if plan.nodes else sorted(branch_inventory)
    if len(selected) > _MAX_NODES or len(selected) * len(rows) > _MAX_WORK:
        _fail('SU-07 node/row analysis exceeds explicit processing limit; select fewer nodes, no implicit sampling')
    ambiguous = {item['variable_id'] for item in data['variables']
                 if item['merged_node_identity_ambiguous']}
    accepted_scopes: dict[str, tuple[str, ...]] = {}
    for node in selected:
        entry = branch_inventory[node]
        scope = tuple(sorted((set(entry['scope_variable_ids']) - ambiguous) -
                             ({plan.target} if plan.target else set())))
        accepted_scopes[node] = scope
    used_variables = {name for scope in accepted_scopes.values() for name in scope}
    used_variables.update(plan.retained_variables)
    if plan.target:
        used_variables.add(plan.target)
    specs = _discretizations(rows, tuple(spec for spec in plan.bins if spec[0] in used_variables))
    output = []
    overlays = []
    for node in selected:
        entry = branch_inventory[node]
        scope = accepted_scopes[node]
        candidate = tuple(sorted(set(scope) - set(plan.retained_variables)))
        excluded_ambiguous = sorted(set(entry['scope_variable_ids']) & ambiguous)
        output_node = {
            'node_id': node, 'scope': entry['scope'],
            'direct_variable_ids': entry['direct_variable_ids'],
            'scope_variable_ids': list(scope),
            'candidate_variable_ids': list(candidate),
            'retained_overlap_variable_ids': sorted(set(scope) & set(plan.retained_variables)),
            'ambiguous_merged_variable_ids': excluded_ambiguous,
            'source_graph_edges_are_not_statistical_associations': True,
        }
        if not candidate:
            metrics = {'status': 'no_new_observed_variables', 'support': None,
                       'entropy': None, 'relevance': None,
                       'redundancy_candidate_retained_mi': None,
                       'warnings': ['empty_candidate_after_target_retained_and_ambiguous_exclusions']}
        elif len(set((*scope, *plan.retained_variables, *((plan.target,) if plan.target else ())))) > \
                 _MAX_COLUMNS_PER_ESTIMATE:
            metrics = {'status': 'joint_dimensionality_limit', 'support': None,
                       'entropy': None, 'relevance': None,
                       'redundancy_candidate_retained_mi': None,
                       'warnings': ['too_many_joint_variables_for_empirical_discrete_estimation']}
        else:
            metrics = _node_information(rows, scope, candidate, plan, specs)
        output_node['information'] = metrics
        output.append(output_node)
        relevance = metrics.get('relevance') or {}
        entropy = metrics.get('entropy') or {}
        overlays.append({
            'node_id': node,
            'scope': entry['scope'],
            'status': metrics['status'],
            'candidate_joint_entropy': entropy.get('candidate_joint'),
            'target_candidate_mi': relevance.get('target_candidate_mi'),
            'target_candidate_cmi_given_retained': relevance.get('target_candidate_cmi_given_retained'),
            'redundancy_candidate_retained_mi': metrics.get('redundancy_candidate_retained_mi'),
            'evidence_is_node_scoped_not_edge_attributed': True,
        })
    result = {
        'schema': RESULT_SCHEMA,
        'source_object_id': data['source_object_id'],
        'source_digest': data['source_digest'],
        'binding_digest': plan.binding_digest,
        'observation_unit': data['observation_unit'],
        'plan': plan.to_dict(),
        'estimator': {'name': 'empirical_discrete', 'log_base': plan.log_base,
                      'units': 'bits' if plan.log_base == 2 else 'log_base_units',
                      'bias_correction': None, 'confidence_intervals': False,
                      'bin_fit_scope': 'all_available_aligned_rows_not_fit_per_branch',
                      'discretization': specs,
                      'missing_policy': plan.missing_policy,
                      'weighted_observations': False},
        'support': {'aligned_rows': len(rows), 'analyzed_nodes': len(output),
                    'independence_verified': False},
        'nodes': output,
        'association_layer': {'schema': ASSOCIATION_SCHEMA,
                              'kind': 'empirical_statistical_node_overlay',
                              'node_overlays': overlays,
                              'source_edges': [], 'probabilistic_edges': [],
                              'causal_interpretation': False},
        'limitations': [
            'empirical_discrete_plugin_estimates_have_finite_sample_bias',
            'conditional_support_and_joint_sparsity_limit_interpretation',
            'descriptive_bin_cutpoints_not_train_test_fitted_for_predictive_evaluation',
            'no_sampling_independence_verified',
            'source_edges_are_not_statistical_associations_or_causal_links',
            'redundancy_and_low_marginal_mi_do_not_authorize_pruning',
            'zero_marginal_mi_can_hide_conditional_xor_information',
            'scope_variable_membership_not_variance_or_information_allocation_to_graph_edges',
            'candidate_excludes_retained_and_target_variables_to_avoid_self_information',
            'nonhierarchical_graphs_have_direct_mappings_only',
            'no_inference_from_structural_node_counts_or_repeated_cell_occurrences',
        ],
    }
    return BranchInformationResult(_json(result))
