"""Generic unknown-data intake and statistical profiling for ML Lab.

Data Lab supports CSV/TSV inventory, profiling, browsing, and explicit non-destructive
transformation recipes. Raw source files remain immutable by default.
"""

from .artifacts import load_persisted_profile, load_profile_run, persist_profile_run, recent_profile_runs
from .comparison import compare_files, compare_paths, save_comparison
from .intake import inspect_file, inspect_paths
from .profiling import profile_file, profile_paths
from .provenance import (
    describe_dataset,
    load_provenance_event,
    logical_table_sha256,
    provenance_sidecar_path,
    save_lineage,
    trace_lineage,
)
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
    "compare_files",
    "compare_paths",
    "DataFileRecord",
    "DataInventory",
    "DataProfile",
    "DataProfileCollection",
    "DataQualityIssue",
    "describe_dataset",
    "MalformedRow",
    "RelationshipProfile",
    "inspect_file",
    "load_persisted_profile",
    "load_provenance_event",
    "logical_table_sha256",
    "load_profile_run",
    "inspect_paths",
    "profile_file",
    "profile_paths",
    "provenance_sidecar_path",
    "persist_profile_run",
    "read_table_page",
    "recent_profile_runs",
    "save_comparison",
    "save_inventory",
    "save_profile",
    "save_lineage",
    "apply_transformation",
    "load_recipe",
    "preview_transformation",
    "trace_lineage",
    "transform_dataframe",
]
