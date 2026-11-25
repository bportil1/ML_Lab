"""Domain-independent structured-data contracts, transformations and analysis."""
from .contracts import (
    CONTRACT_SCHEMA, CONTRACT_VERSION, TOPOLOGIES, OBSERVATION_UNITS,
    CELL_STATES, NULL_REASONS, CAPABILITIES, OPERATION_REQUIREMENTS,
    StructuredContractError, require_capabilities, require_observation_alignment,
)
from .object import StructuredObject
from .transformations import (
    PLAN_SCHEMA, REPORT_SCHEMA, TransformationPlan, TransformationResult, transform,
)
from .information import (
    INFORMATION_SCHEMA, MEASURES, MISSING_POLICIES, BINNING_MODES,
    estimate_information, estimate_entropy, estimate_conditional_entropy,
    estimate_mutual_information, estimate_conditional_mutual_information,
)
from .probabilistic import (
    MODEL_SCHEMA, MODEL_TYPE, MISSING_POLICIES as MODEL_MISSING_POLICIES,
    ProbabilisticModelSpec, source_reference,
)
from .probabilistic_inference import (
    FIT_SCHEMA, QUERY_SCHEMA, FittingResult, fit_categorical_model, query_exact,
)
from .comparison import (
    STUDY_SCHEMA, ComparisonCandidate, ComparisonStudy, compare_representations,
)
from .descriptive import (
    STRUCTURE_SCHEMA, COLLECTION_SCHEMA, OBSERVATION_SCHEMA, VARIANCE_SCHEMA,
    describe_structure, describe_collection, describe_observations, describe_nested_variance,
)

from .bindings import (
    ObservationBindingPlan, ObservationBindingResult, VariableBinding,
    bind_observations, source_sha256,
)

from .branch_uncertainty import (
    BranchUncertaintyPlan, BranchUncertaintyResult, analyze_branch_uncertainty,
    binding_sha256,
)

from .branch_information import (
    BranchInformationPlan, BranchInformationResult, analyze_branch_information,
    ASSOCIATION_SCHEMA as BRANCH_ASSOCIATION_SCHEMA,
    PLAN_SCHEMA as BRANCH_INFORMATION_PLAN_SCHEMA,
    RESULT_SCHEMA as BRANCH_INFORMATION_SCHEMA,
)

__all__ = [
    'CONTRACT_SCHEMA', 'CONTRACT_VERSION', 'TOPOLOGIES', 'OBSERVATION_UNITS',
    'CELL_STATES', 'NULL_REASONS', 'CAPABILITIES', 'OPERATION_REQUIREMENTS',
    'StructuredContractError', 'require_capabilities', 'require_observation_alignment',
    'StructuredObject', 'PLAN_SCHEMA', 'REPORT_SCHEMA',
    'TransformationPlan', 'TransformationResult', 'transform',
    'STRUCTURE_SCHEMA', 'COLLECTION_SCHEMA', 'OBSERVATION_SCHEMA', 'VARIANCE_SCHEMA',
    'describe_structure', 'describe_collection', 'describe_observations', 'describe_nested_variance',
    'INFORMATION_SCHEMA', 'MEASURES', 'MISSING_POLICIES', 'BINNING_MODES',
    'estimate_information', 'estimate_entropy', 'estimate_conditional_entropy',
    'estimate_mutual_information', 'estimate_conditional_mutual_information',
    'MODEL_SCHEMA', 'MODEL_TYPE', 'MODEL_MISSING_POLICIES',
    'ProbabilisticModelSpec', 'source_reference',
    'FIT_SCHEMA', 'QUERY_SCHEMA', 'FittingResult', 'fit_categorical_model', 'query_exact',
    'STUDY_SCHEMA', 'ComparisonCandidate', 'ComparisonStudy', 'compare_representations',
    'ObservationBindingPlan', 'ObservationBindingResult', 'VariableBinding',
    'bind_observations', 'source_sha256',
    'BranchUncertaintyPlan', 'BranchUncertaintyResult', 'analyze_branch_uncertainty',
    'binding_sha256',
    'BranchInformationPlan', 'BranchInformationResult', 'analyze_branch_information',
    'BRANCH_ASSOCIATION_SCHEMA', 'BRANCH_INFORMATION_PLAN_SCHEMA',
    'BRANCH_INFORMATION_SCHEMA',
]

from .candidates import (
    StructuralCandidatePlan, StructuralCandidateResult,
    apply_structural_candidate, replay_structural_candidate,
    enumerate_structural_candidates,
    PLAN_SCHEMA as STRUCTURAL_CANDIDATE_PLAN_SCHEMA,
    MANIFEST_SCHEMA as STRUCTURAL_CANDIDATE_MANIFEST_SCHEMA,
    OBSERVATION_POLICIES as STRUCTURAL_CANDIDATE_OBSERVATION_POLICIES,
    OPERATIONS as STRUCTURAL_CANDIDATE_OPERATIONS,
)

__all__ += [
    'StructuralCandidatePlan', 'StructuralCandidateResult',
    'apply_structural_candidate', 'replay_structural_candidate',
    'enumerate_structural_candidates',
    'STRUCTURAL_CANDIDATE_PLAN_SCHEMA', 'STRUCTURAL_CANDIDATE_MANIFEST_SCHEMA',
    'STRUCTURAL_CANDIDATE_OBSERVATION_POLICIES', 'STRUCTURAL_CANDIDATE_OPERATIONS',
]

from .retention import (
    StructuralRetentionStudy, StructuralRetentionResult, compare_structural_retention,
    PLAN_SCHEMA as STRUCTURAL_RETENTION_STUDY_SCHEMA,
    REPORT_SCHEMA as STRUCTURAL_RETENTION_REPORT_SCHEMA,
)

__all__ += [
    'StructuralRetentionStudy', 'StructuralRetentionResult',
    'compare_structural_retention', 'STRUCTURAL_RETENTION_STUDY_SCHEMA',
    'STRUCTURAL_RETENTION_REPORT_SCHEMA',
]

from .network_information import (
    analyze_observed_network,
    SCHEMA as OBSERVED_INFORMATION_NETWORK_SCHEMA,
)

__all__ += ['analyze_observed_network', 'OBSERVED_INFORMATION_NETWORK_SCHEMA']

# HIA-01: headless, read-only scope and scientific eligibility contracts.
from .hierarchical_contracts import (
    REQUEST_SCHEMA as HIA_REQUEST_SCHEMA,
    ELIGIBILITY_SCHEMA as HIA_ELIGIBILITY_SCHEMA,
    METRIC_SCOPES as HIA_METRIC_SCOPES,
    HierarchicalScope, HierarchicalAnalysisRequest, inspect_hierarchical_eligibility,
)

__all__ += [
    'HIA_REQUEST_SCHEMA', 'HIA_ELIGIBILITY_SCHEMA', 'HIA_METRIC_SCOPES',
    'HierarchicalScope', 'HierarchicalAnalysisRequest',
    'inspect_hierarchical_eligibility',
]

# HIA-02: deterministic tree/forest/DAG levels and exact scope membership.
from .hierarchical_index import (
    HIERARCHY_SCHEMA as HIA_HIERARCHY_SCHEMA,
    SELECTION_SCHEMA as HIA_SELECTION_SCHEMA,
    HierarchyIndex, build_hierarchy_index, inspect_hierarchical_selection,
)

__all__ += [
    'HIA_HIERARCHY_SCHEMA', 'HIA_SELECTION_SCHEMA', 'HierarchyIndex',
    'build_hierarchy_index', 'inspect_hierarchical_selection',
]

# HIA-04: explicit one-to-one node/observation profiles (no inferred mapping).
from .atomic_node_statistics import ATOMIC_SCHEMA as HIA_ATOMIC_SCHEMA, atomic_node_statistics
__all__ += ['HIA_ATOMIC_SCHEMA', 'atomic_node_statistics']

# HIA-05: explicit neighboring-node observed associations.
from .neighbor_statistics import NEIGHBOR_SCHEMA as HIA_NEIGHBOR_SCHEMA, analyze_neighbor_relationships
__all__ += ['HIA_NEIGHBOR_SCHEMA', 'analyze_neighbor_relationships']
