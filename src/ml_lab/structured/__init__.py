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
from .descriptive import (
    STRUCTURE_SCHEMA, COLLECTION_SCHEMA, OBSERVATION_SCHEMA, VARIANCE_SCHEMA,
    describe_structure, describe_collection, describe_observations, describe_nested_variance,
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
]
