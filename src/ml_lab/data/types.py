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
