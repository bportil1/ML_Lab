"""Generic unknown-data intake and statistical profiling for ML Lab.

Data Lab currently supports read-only CSV/TSV inventory (A1) and statistical
profiling (A2). Cleaning and transformations are intentionally deferred.
"""

from .intake import inspect_file, inspect_paths
from .profiling import profile_file, profile_paths
from .reporting import save_inventory, save_profile
from .types import (
    ColumnProfile,
    DataFileRecord,
    DataInventory,
    DataProfile,
    DataProfileCollection,
    DataQualityIssue,
    MalformedRow,
    RelationshipProfile,
)

__all__ = [
    "ColumnProfile",
    "DataFileRecord",
    "DataInventory",
    "DataProfile",
    "DataProfileCollection",
    "DataQualityIssue",
    "MalformedRow",
    "RelationshipProfile",
    "inspect_file",
    "inspect_paths",
    "profile_file",
    "profile_paths",
    "save_inventory",
    "save_profile",
]
