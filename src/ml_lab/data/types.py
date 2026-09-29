from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MalformedRow:
    line_number: int
    observed_columns: int | None
    expected_columns: int | None
    reason: str

    def to_record(self) -> dict[str, Any]:
        return {
            "line_number": self.line_number,
            "observed_columns": self.observed_columns,
            "expected_columns": self.expected_columns,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class XmlNamespace:
    prefix: str
    uri: str

    def to_record(self) -> dict[str, Any]:
        return {"prefix": self.prefix, "uri": self.uri}


@dataclass(frozen=True)
class XmlAttributeProfile:
    name: str
    local_name: str
    namespace_uri: str | None
    prefix: str | None
    occurrence_count: int
    element_occurrence_count: int
    presence_rate: float
    sampled_value_count: int
    distinct_sample_count: int
    distinct_sample_rate: float
    likely_identifier: bool = False

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "local_name": self.local_name,
            "namespace_uri": self.namespace_uri,
            "prefix": self.prefix,
            "occurrence_count": self.occurrence_count,
            "element_occurrence_count": self.element_occurrence_count,
            "presence_rate": self.presence_rate,
            "sampled_value_count": self.sampled_value_count,
            "distinct_sample_count": self.distinct_sample_count,
            "distinct_sample_rate": self.distinct_sample_rate,
            "likely_identifier": self.likely_identifier,
        }


@dataclass(frozen=True)
class XmlElementProfile:
    path: str
    canonical_path: str
    tag: str
    local_name: str
    namespace_uri: str | None
    prefix: str | None
    depth: int
    occurrence_count: int
    parent_path: str | None
    parent_occurrence_count: int | None
    parents_with_element: int | None
    min_per_parent: int | None
    max_per_parent: int | None
    mean_per_parent: float | None
    repeated: bool
    optional: bool
    text_occurrence_count: int
    text_presence_rate: float
    text_sampled_value_count: int
    text_distinct_sample_count: int
    text_distinct_sample_rate: float
    likely_identifier_text: bool
    attributes: tuple[XmlAttributeProfile, ...] = ()
    child_paths: tuple[str, ...] = ()
    likely_identifier_attributes: tuple[str, ...] = ()
    record_candidate_score: float = 0.0
    record_candidate_reasons: tuple[str, ...] = ()

    def to_record(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "canonical_path": self.canonical_path,
            "tag": self.tag,
            "local_name": self.local_name,
            "namespace_uri": self.namespace_uri,
            "prefix": self.prefix,
            "depth": self.depth,
            "occurrence_count": self.occurrence_count,
            "parent_path": self.parent_path,
            "parent_occurrence_count": self.parent_occurrence_count,
            "parents_with_element": self.parents_with_element,
            "cardinality": {
                "min_per_parent": self.min_per_parent,
                "max_per_parent": self.max_per_parent,
                "mean_per_parent": self.mean_per_parent,
                "repeated": self.repeated,
                "optional": self.optional,
            },
            "text": {
                "occurrence_count": self.text_occurrence_count,
                "presence_rate": self.text_presence_rate,
                "sampled_value_count": self.text_sampled_value_count,
                "distinct_sample_count": self.text_distinct_sample_count,
                "distinct_sample_rate": self.text_distinct_sample_rate,
                "likely_identifier": self.likely_identifier_text,
            },
            "attributes": [item.to_record() for item in self.attributes],
            "child_paths": list(self.child_paths),
            "likely_identifier_attributes": list(self.likely_identifier_attributes),
            "record_candidate_score": self.record_candidate_score,
            "record_candidate_reasons": list(self.record_candidate_reasons),
        }


@dataclass(frozen=True)
class XmlRecordCandidate:
    path: str
    canonical_path: str
    score: float
    occurrence_count: int
    depth: int
    reasons: tuple[str, ...] = ()

    def to_record(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "canonical_path": self.canonical_path,
            "score": self.score,
            "occurrence_count": self.occurrence_count,
            "depth": self.depth,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class XmlFieldCandidate:
    field_id: str
    kind: str
    path: str
    canonical_path: str
    relative_path: str
    source_element_path: str
    source_element_canonical_path: str
    name: str
    local_name: str
    namespace_uri: str | None
    prefix: str | None
    repeated: bool
    optional: bool
    likely_identifier: bool = False

    def to_record(self) -> dict[str, Any]:
        return {
            "field_id": self.field_id,
            "kind": self.kind,
            "path": self.path,
            "canonical_path": self.canonical_path,
            "relative_path": self.relative_path,
            "source_element_path": self.source_element_path,
            "source_element_canonical_path": self.source_element_canonical_path,
            "name": self.name,
            "local_name": self.local_name,
            "namespace_uri": self.namespace_uri,
            "prefix": self.prefix,
            "repeated": self.repeated,
            "optional": self.optional,
            "likely_identifier": self.likely_identifier,
        }


@dataclass(frozen=True)
class XmlRecordSelection:
    source_fingerprint: str
    record_root_path: str
    record_root_canonical_path: str
    record_root_occurrence_count: int
    record_root_candidate_score: float
    record_root_candidate_reasons: tuple[str, ...] = ()
    fields: tuple[XmlFieldCandidate, ...] = ()
    schema: str = "ml-lab.xml-record-selection@1"

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_fingerprint": self.source_fingerprint,
            "record_root": {
                "path": self.record_root_path,
                "canonical_path": self.record_root_canonical_path,
                "occurrence_count": self.record_root_occurrence_count,
                "candidate_score": self.record_root_candidate_score,
                "candidate_reasons": list(self.record_root_candidate_reasons),
            },
            "fields": [field.to_record() for field in self.fields],
        }


@dataclass(frozen=True)
class XmlCollectionRule:
    branch_canonical_path: str
    strategy: str | None = None
    join_delimiter: str | None = None
    aggregate_operation: str | None = None
    pivot_key_field_id: str | None = None
    pivot_value_field_id: str | None = None
    separate_table_name: str | None = None
    valid: bool = False
    errors: tuple[str, ...] = ()

    def to_record(self) -> dict[str, Any]:
        return {
            "branch_canonical_path": self.branch_canonical_path,
            "strategy": self.strategy,
            "options": {
                "join_delimiter": self.join_delimiter,
                "aggregate_operation": self.aggregate_operation,
                "pivot_key_field_id": self.pivot_key_field_id,
                "pivot_value_field_id": self.pivot_value_field_id,
                "separate_table_name": self.separate_table_name,
            },
            "valid": self.valid,
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class XmlRepeatedBranch:
    path: str
    canonical_path: str
    relative_path: str
    depth: int
    occurrence_count: int
    parent_occurrence_count: int | None
    min_per_parent: int | None
    max_per_parent: int | None
    mean_per_parent: float | None
    parent_repeated_branch_canonical_path: str | None = None
    selected_field_ids: tuple[str, ...] = ()
    selected_fields: tuple[XmlFieldCandidate, ...] = ()
    rule: XmlCollectionRule | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "canonical_path": self.canonical_path,
            "relative_path": self.relative_path,
            "depth": self.depth,
            "occurrence_count": self.occurrence_count,
            "parent_occurrence_count": self.parent_occurrence_count,
            "cardinality": {
                "min_per_parent": self.min_per_parent,
                "max_per_parent": self.max_per_parent,
                "mean_per_parent": self.mean_per_parent,
            },
            "parent_repeated_branch_canonical_path": self.parent_repeated_branch_canonical_path,
            "selected_field_ids": list(self.selected_field_ids),
            "selected_fields": [field.to_record() for field in self.selected_fields],
            "rule": self.rule.to_record() if self.rule is not None else None,
        }


@dataclass(frozen=True)
class XmlCollectionPlan:
    source_fingerprint: str
    record_root_path: str
    record_root_canonical_path: str
    selected_field_ids: tuple[str, ...] = ()
    repeated_branches: tuple[XmlRepeatedBranch, ...] = ()
    strategy_catalog: tuple[dict[str, Any], ...] = ()
    unresolved_branch_count: int = 0
    invalid_rule_count: int = 0
    ready_for_preview: bool = False
    warnings: tuple[str, ...] = ()
    schema: str = "ml-lab.xml-collection-plan@1"

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_fingerprint": self.source_fingerprint,
            "record_root": {
                "path": self.record_root_path,
                "canonical_path": self.record_root_canonical_path,
            },
            "selected_field_ids": list(self.selected_field_ids),
            "repeated_branches": [branch.to_record() for branch in self.repeated_branches],
            "strategy_catalog": [dict(item) for item in self.strategy_catalog],
            "summary": {
                "repeated_branch_count": len(self.repeated_branches),
                "unresolved_branch_count": self.unresolved_branch_count,
                "invalid_rule_count": self.invalid_rule_count,
                "ready_for_preview": self.ready_for_preview,
            },
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class XmlPreviewColumn:
    name: str
    source_field_ids: tuple[str, ...] = ()
    source_relative_paths: tuple[str, ...] = ()
    source_element_canonical_paths: tuple[str, ...] = ()
    strategy: str = "direct"
    repeated_branch_canonical_path: str | None = None
    dynamic: bool = False

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_field_ids": list(self.source_field_ids),
            "source_relative_paths": list(self.source_relative_paths),
            "source_element_canonical_paths": list(self.source_element_canonical_paths),
            "strategy": self.strategy,
            "repeated_branch_canonical_path": self.repeated_branch_canonical_path,
            "dynamic": self.dynamic,
        }


@dataclass(frozen=True)
class XmlPreviewTable:
    name: str
    role: str
    columns: tuple[XmlPreviewColumn, ...] = ()
    rows: tuple[dict[str, Any], ...] = ()
    preview_row_count: int = 0
    estimated_total_rows: int | None = None
    truncated: bool = False

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "columns": [column.to_record() for column in self.columns],
            "rows": [dict(row) for row in self.rows],
            "preview_row_count": self.preview_row_count,
            "estimated_total_rows": self.estimated_total_rows,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class XmlTabularPreview:
    source_fingerprint: str
    preview_signature: str
    record_root_path: str
    record_root_canonical_path: str
    selected_field_ids: tuple[str, ...] = ()
    main_table: XmlPreviewTable | None = None
    child_tables: tuple[XmlPreviewTable, ...] = ()
    warnings: tuple[str, ...] = ()
    schema: str = "ml-lab.xml-tabular-preview@1"

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_fingerprint": self.source_fingerprint,
            "preview_signature": self.preview_signature,
            "record_root": {
                "path": self.record_root_path,
                "canonical_path": self.record_root_canonical_path,
            },
            "selected_field_ids": list(self.selected_field_ids),
            "main_table": self.main_table.to_record() if self.main_table is not None else None,
            "child_tables": [table.to_record() for table in self.child_tables],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class XmlMaterializationResult:
    source_path: str
    source_fingerprint: str
    preview_signature: str
    dataset_path: str
    structure_artifact_path: str
    manifest_path: str
    row_count: int
    column_count: int
    columns: tuple[str, ...] = ()
    child_table_paths: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    schema: str = "ml-lab.xml-materialization@1"

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source": {
                "path": self.source_path,
                "fingerprint": self.source_fingerprint,
            },
            "preview_signature": self.preview_signature,
            "dataset": {
                "path": self.dataset_path,
                "rows": self.row_count,
                "columns": self.column_count,
                "column_names": list(self.columns),
            },
            "structure_artifact_path": self.structure_artifact_path,
            "manifest_path": self.manifest_path,
            "child_table_paths": list(self.child_table_paths),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class XmlStructureArtifact:
    source_path: str
    source_sha256: str
    source_fingerprint: str
    encoding: str
    has_xml_declaration: bool
    root_tag: str
    root_local_name: str
    root_namespace_uri: str | None
    root_prefix: str | None
    root_attribute_names: tuple[str, ...] = ()
    namespaces: tuple[XmlNamespace, ...] = ()
    top_level_element_tags: tuple[str, ...] = ()
    top_level_child_count: int = 0
    element_count: int = 0
    observed_max_depth: int = 0
    unique_element_path_count: int = 0
    leaf_path_count: int = 0
    repeated_path_count: int = 0
    optional_path_count: int = 0
    element_profiles: tuple[XmlElementProfile, ...] = ()
    record_candidates: tuple[XmlRecordCandidate, ...] = ()
    schema: str = "ml-lab.xml-structure@2"

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "source_fingerprint": self.source_fingerprint,
            "encoding": self.encoding,
            "has_xml_declaration": self.has_xml_declaration,
            "root": {
                "tag": self.root_tag,
                "local_name": self.root_local_name,
                "namespace_uri": self.root_namespace_uri,
                "prefix": self.root_prefix,
                "attribute_names": list(self.root_attribute_names),
            },
            "namespaces": [item.to_record() for item in self.namespaces],
            "top_level_element_tags": list(self.top_level_element_tags),
            "top_level_child_count": self.top_level_child_count,
            "element_count": self.element_count,
            "observed_max_depth": self.observed_max_depth,
            "analysis": {
                "unique_element_path_count": self.unique_element_path_count,
                "leaf_path_count": self.leaf_path_count,
                "repeated_path_count": self.repeated_path_count,
                "optional_path_count": self.optional_path_count,
            },
            "element_profiles": [item.to_record() for item in self.element_profiles],
            "record_candidates": [item.to_record() for item in self.record_candidates],
        }


@dataclass(frozen=True)
class DataFileRecord:
    path: str
    relative_path: str
    size_bytes: int
    sha256: str
    suffix: str
    format: str
    supported: bool
    parse_status: str
    encoding: str | None = None
    delimiter: str | None = None
    has_header: bool | None = None
    row_count: int | None = None
    column_count: int | None = None
    columns: tuple[str, ...] = ()
    likely_exported_index_columns: tuple[str, ...] = ()
    malformed_row_count: int = 0
    malformed_rows: tuple[MalformedRow, ...] = ()
    warnings: tuple[str, ...] = ()
    error: str | None = None
    tabular_ready: bool = True
    xml_structure: XmlStructureArtifact | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "suffix": self.suffix,
            "format": self.format,
            "supported": self.supported,
            "parse_status": self.parse_status,
            "encoding": self.encoding,
            "delimiter": self.delimiter,
            "has_header": self.has_header,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": list(self.columns),
            "likely_exported_index_columns": list(self.likely_exported_index_columns),
            "malformed_row_count": self.malformed_row_count,
            "malformed_rows": [row.to_record() for row in self.malformed_rows],
            "warnings": list(self.warnings),
            "error": self.error,
            "tabular_ready": self.tabular_ready,
            "xml_structure": self.xml_structure.to_record() if self.xml_structure is not None else None,
        }


@dataclass(frozen=True)
class DataInventory:
    requested_paths: tuple[str, ...]
    recursive: bool
    include_hidden: bool
    discovered_file_count: int
    supported_file_count: int
    parsed_file_count: int
    partial_file_count: int
    failed_file_count: int
    unsupported_file_count: int
    total_size_bytes: int
    files: tuple[DataFileRecord, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    schema: str = "ml-lab.data-inventory@1"

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "requested_paths": list(self.requested_paths),
            "recursive": self.recursive,
            "include_hidden": self.include_hidden,
            "summary": {
                "discovered_file_count": self.discovered_file_count,
                "supported_file_count": self.supported_file_count,
                "parsed_file_count": self.parsed_file_count,
                "partial_file_count": self.partial_file_count,
                "failed_file_count": self.failed_file_count,
                "unsupported_file_count": self.unsupported_file_count,
                "total_size_bytes": self.total_size_bytes,
            },
            "warnings": list(self.warnings),
            "files": [file.to_record() for file in self.files],
        }


@dataclass(frozen=True)
class DataQualityIssue:
    severity: str
    code: str
    column: str | None
    message: str

    def to_record(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "column": self.column,
            "message": self.message,
        }


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    position: int
    inferred_type: str
    non_missing_count: int
    missing_count: int
    missing_rate: float
    unique_count: int
    cardinality_ratio: float
    constant: bool
    likely_identifier: bool
    dominant_value_rate: float
    top_values: tuple[dict[str, Any], ...] = ()
    numeric_summary: dict[str, Any] = field(default_factory=dict)
    detail_summary: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "position": self.position,
            "inferred_type": self.inferred_type,
            "non_missing_count": self.non_missing_count,
            "missing_count": self.missing_count,
            "missing_rate": self.missing_rate,
            "unique_count": self.unique_count,
            "cardinality_ratio": self.cardinality_ratio,
            "constant": self.constant,
            "likely_identifier": self.likely_identifier,
            "dominant_value_rate": self.dominant_value_rate,
            "top_values": list(self.top_values),
            "numeric_summary": dict(self.numeric_summary),
            "detail_summary": dict(self.detail_summary),
        }


@dataclass(frozen=True)
class RelationshipProfile:
    left: str
    right: str
    pair_count: int
    pearson: float | None
    spearman: float | None
    mutual_information: float | None
    normalized_mutual_information: float | None
    mi_method: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "left": self.left,
            "right": self.right,
            "pair_count": self.pair_count,
            "pearson": self.pearson,
            "spearman": self.spearman,
            "mutual_information": self.mutual_information,
            "normalized_mutual_information": self.normalized_mutual_information,
            "mi_method": self.mi_method,
        }


@dataclass(frozen=True)
class DataProfile:
    path: str
    relative_path: str
    source_sha256: str
    source_row_count: int
    profiled_row_count: int
    column_count: int
    sampled: bool
    sample_strategy: str
    relationship_row_count: int
    duplicate_row_count: int
    duplicate_row_rate: float
    columns: tuple[ColumnProfile, ...] = field(default_factory=tuple)
    quality_issues: tuple[DataQualityIssue, ...] = field(default_factory=tuple)
    relationships: tuple[RelationshipProfile, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    schema: str = "ml-lab.data-profile@1"

    def to_record(self) -> dict[str, Any]:
        severity_counts: dict[str, int] = {}
        for issue in self.quality_issues:
            severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1
        return {
            "schema": self.schema,
            "path": self.path,
            "relative_path": self.relative_path,
            "source_sha256": self.source_sha256,
            "summary": {
                "source_row_count": self.source_row_count,
                "profiled_row_count": self.profiled_row_count,
                "column_count": self.column_count,
                "sampled": self.sampled,
                "sample_strategy": self.sample_strategy,
                "relationship_row_count": self.relationship_row_count,
                "duplicate_row_count": self.duplicate_row_count,
                "duplicate_row_rate": self.duplicate_row_rate,
                "quality_issue_count": len(self.quality_issues),
                "quality_issue_severity_counts": severity_counts,
                "relationship_count": len(self.relationships),
            },
            "warnings": list(self.warnings),
            "quality_issues": [issue.to_record() for issue in self.quality_issues],
            "columns": [column.to_record() for column in self.columns],
            "relationships": [relationship.to_record() for relationship in self.relationships],
        }


@dataclass(frozen=True)
class DataProfileCollection:
    requested_paths: tuple[str, ...]
    inventory: DataInventory
    profiles: tuple[DataProfile, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    schema: str = "ml-lab.data-profile-collection@1"

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "requested_paths": list(self.requested_paths),
            "summary": {
                "profile_count": len(self.profiles),
                "inventory_file_count": self.inventory.discovered_file_count,
                "supported_file_count": self.inventory.supported_file_count,
            },
            "warnings": list(self.warnings),
            "inventory": self.inventory.to_record(),
            "profiles": [profile.to_record() for profile in self.profiles],
        }
