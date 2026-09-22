"""Generic unknown-data intake and statistical profiling for ML Lab.

Data Lab supports CSV/TSV inventory, profiling, browsing, and explicit non-destructive
transformation recipes. Raw source files remain immutable by default.
"""

from .artifacts import load_persisted_profile, load_profile_run, persist_profile_run, recent_profile_runs
from .intake import inspect_file, inspect_paths
from .profiling import profile_file, profile_paths
from .reporting import save_inventory, save_profile
from .transform import apply_transformation, load_recipe, preview_transformation, transform_dataframe
from .table import read_table_page
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
    "load_persisted_profile",
    "load_profile_run",
    "inspect_paths",
    "profile_file",
    "profile_paths",
    "persist_profile_run",
    "read_table_page",
    "recent_profile_runs",
    "save_inventory",
    "save_profile",
    "apply_transformation",
    "load_recipe",
    "preview_transformation",
    "transform_dataframe",
]
