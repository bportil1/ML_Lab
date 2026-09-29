"""Generic unknown-data intake and statistical profiling for ML Lab.

Data Lab supports CSV/TSV inventory, profiling, browsing, and explicit non-destructive
transformation recipes. XML sources can also be safely ingested into a structural artifact
without being treated as tabular until guided extraction is configured. Repeated XML branches can be assigned explicit one-to-many rules before preview/materialization. Raw source files
remain immutable by default.
"""

from .artifacts import load_persisted_profile, load_profile_run, persist_profile_run, recent_profile_runs
from .comparison import compare_files, compare_paths, save_comparison
from .intake import inspect_file, inspect_paths
from .xml_ingest import XmlSafetyError, analyze_xml_structure, inspect_xml_structure, load_xml_structure
from .xml_selection import XmlSelectionError, build_xml_record_selection
from .xml_rules import XmlCollectionRuleError, build_xml_collection_plan, xml_collection_strategy_catalog
from .xml_preview import XmlPreviewError, preview_xml_tabularization
from .xml_materialize import XmlMaterializationError, materialize_xml_tabularization
from .xml_provenance import build_xml_extraction_recipe, persist_xml_extraction_provenance, regenerate_xml_extraction
from .profiling import profile_file, profile_paths
from .provenance import (
    describe_dataset,
    load_provenance_event,
    logical_table_sha256,
    provenance_sidecar_path,
    save_lineage,
    trace_lineage,
)
from .reporting import save_inventory, save_profile, save_xml_structure
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
    XmlAttributeProfile,
    XmlElementProfile,
    XmlNamespace,
    XmlFieldCandidate,
    XmlRecordCandidate,
    XmlRecordSelection,
    XmlCollectionPlan,
    XmlCollectionRule,
    XmlRepeatedBranch,
    XmlPreviewColumn,
    XmlPreviewTable,
    XmlTabularPreview,
    XmlMaterializationResult,
    XmlStructureArtifact,
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
    "XmlAttributeProfile",
    "XmlElementProfile",
    "XmlNamespace",
    "XmlFieldCandidate",
    "XmlRecordCandidate",
    "XmlRecordSelection",
    "XmlCollectionPlan",
    "XmlCollectionRule",
    "XmlRepeatedBranch",
    "XmlPreviewColumn",
    "XmlPreviewTable",
    "XmlTabularPreview",
    "XmlMaterializationResult",
    "XmlStructureArtifact",
    "XmlSafetyError",
    "XmlSelectionError",
    "XmlCollectionRuleError",
    "XmlPreviewError",
    "XmlMaterializationError",
    "RelationshipProfile",
    "inspect_file",
    "analyze_xml_structure",
    "inspect_xml_structure",
    "load_xml_structure",
    "build_xml_record_selection",
    "build_xml_collection_plan",
    "xml_collection_strategy_catalog",
    "preview_xml_tabularization",
    "materialize_xml_tabularization",
    "build_xml_extraction_recipe",
    "persist_xml_extraction_provenance",
    "regenerate_xml_extraction",
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
    "save_xml_structure",
    "save_profile",
    "save_lineage",
    "apply_transformation",
    "load_recipe",
    "preview_transformation",
    "trace_lineage",
    "transform_dataframe",
]
