"""Domain-neutral contract vocabulary for future structured analyses.

ML-UW-27 establishes names and precondition gates only. StructuredObject
construction, graph algorithms and adapters begin in later sprints.
"""
from .contracts import (
    CONTRACT_SCHEMA, CONTRACT_VERSION, TOPOLOGIES, OBSERVATION_UNITS,
    CELL_STATES, NULL_REASONS, CAPABILITIES, OPERATION_REQUIREMENTS,
    StructuredContractError, require_capabilities, require_observation_alignment,
)

__all__ = [
    'CONTRACT_SCHEMA', 'CONTRACT_VERSION', 'TOPOLOGIES', 'OBSERVATION_UNITS',
    'CELL_STATES', 'NULL_REASONS', 'CAPABILITIES', 'OPERATION_REQUIREMENTS',
    'StructuredContractError', 'require_capabilities', 'require_observation_alignment',
]
