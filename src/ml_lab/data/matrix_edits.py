"""Non-destructive scientific matrix drafts with Data Lab provenance.

Coordinates are 1-based source data rows (excluding the CSV/TSV header) and
0-based column positions. Editing a sorted/filtered view is deliberately not
supported: callers must submit coordinates from a stable, unfiltered page.
"""
from __future__ import annotations

import csv
import os
import re
import tempfile
import uuid
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence

from .intake import inspect_file
from .provenance import describe_dataset, persist_transformation_provenance

MAX_DRAFT_EDITS = 5000
MAX_CELL_CHARS = 16_384
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class MatrixDraftError(ValueError):
    """A draft contains invalid coordinates, values, or an outdated source."""


def _source_record(path: Path, expected_sha256: str):
    if path.is_symlink() or not path.is_file():
        raise MatrixDraftError("Source must be an existing regular file, not a symlink")
    record = inspect_file(path)
    if (record.format != "delimited_text" or record.parse_status != "parsed"
            or record.malformed_row_count or not record.has_header
            or not record.tabular_ready or record.delimiter not in {",", "\t"}):
        raise MatrixDraftError("Editing requires a fully parsed, headed CSV/TSV dataset without malformed rows")
    if not record.columns or len(record.columns) != len(set(record.columns)) or not all(record.columns):
        raise MatrixDraftError("Source column names must be unique and nonempty")
    if not isinstance(expected_sha256, str) or not _SHA256.fullmatch(expected_sha256):
        raise MatrixDraftError("An exact source SHA-256 is required")
    if record.sha256 != expected_sha256:
        raise MatrixDraftError("Source has changed; reload the matrix before saving")
    return record


def _validate_edits(edits: Sequence[Mapping[str, Any]], *, rows: int,
                    columns: int, dtypes: Sequence[str]) -> dict[tuple[int, int], str]:
    if not isinstance(edits, list) or not 1 <= len(edits) <= MAX_DRAFT_EDITS:
        raise MatrixDraftError(f"Submit between 1 and {MAX_DRAFT_EDITS} edited cells")
    changes: dict[tuple[int, int], str] = {}
    for item in edits:
        if not isinstance(item, dict):
            raise MatrixDraftError("Each edit must be an object")
        row, col, value = item.get("row"), item.get("column"), item.get("value")
        if (type(row) is not int or type(col) is not int or not 1 <= row <= rows
                or not 0 <= col < columns or not isinstance(value, str)
                or len(value) > MAX_CELL_CHARS or "\x00" in value):
            raise MatrixDraftError("Invalid row, column or cell value")
        key = row, col
        if key in changes:
            raise MatrixDraftError("Duplicate cell coordinates are not allowed")
        dtype = dtypes[col]
        if value != "":
            if dtype.startswith(("int", "uint")) and not re.fullmatch(r"[+-]?[0-9]+", value):
                raise MatrixDraftError(f"Row {row}, column {col + 1} requires an integer")
            if dtype.startswith("float"):
                try:
                    Decimal(value)
                except InvalidOperation as exc:
                    raise MatrixDraftError(f"Row {row}, column {col + 1} requires a number") from exc
        changes[key] = value
    return changes


def commit_matrix_draft(
    source_path: str | Path, *, output_root: str | Path,
    expected_sha256: str, edits: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Create a fresh derived CSV/TSV with cell edits, never overwriting source.

    Original bytes and typed cell *strings* stay authoritative. CSV parsing and
    writing preserve values such as long decimals and scientific notation without
    converting them through JavaScript float or Pandas numeric representations.
    """
    source = Path(source_path).expanduser().absolute()
    record = _source_record(source, expected_sha256)
    identity = describe_dataset(source)
    if identity["sha256"] != expected_sha256:
        raise MatrixDraftError("Source changed while preparing the draft")
    changes = _validate_edits(edits, rows=identity["rows"], columns=identity["columns"], dtypes=identity["dtypes"])
    output_directory = Path(output_root).expanduser().absolute()
    if output_directory.is_symlink():
        raise MatrixDraftError("Derived output directory must not be a symlink")
    output_directory.mkdir(parents=True, exist_ok=True)
    if output_directory.is_symlink():
        raise MatrixDraftError("Derived output directory must not be a symlink")
    destination = output_directory / f"{source.stem}-matrix-{uuid.uuid4().hex[:16]}{source.suffix.lower()}"
    fd, temp_name = tempfile.mkstemp(prefix=".matrix-draft-", suffix=".tmp", dir=output_directory)
    temporary = Path(temp_name)
    applied: list[dict[str, Any]] = []
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as output_file, \
                source.open("r", encoding=record.encoding, newline="") as input_file:
            reader = csv.reader(input_file, delimiter=record.delimiter, strict=True)
            writer = csv.writer(output_file, delimiter=record.delimiter, lineterminator="\n")
            writer.writerow(next(reader))
            data_rows = 0
            for data_rows, values in enumerate(reader, start=1):
                if len(values) != identity["columns"]:
                    raise MatrixDraftError("Source row width changed during editing")
                for col in range(len(values)):
                    key = data_rows, col
                    if key in changes and values[col] != changes[key]:
                        applied.append({"row": data_rows, "column": col,
                                        "old": values[col], "new": changes[key]})
                        values[col] = changes[key]
                writer.writerow(values)
            output_file.flush()
            os.fsync(output_file.fileno())
        if data_rows != identity["rows"] or inspect_file(source).sha256 != expected_sha256:
            raise MatrixDraftError("Source changed while writing the derived dataset")
        if not applied:
            raise MatrixDraftError("No cells changed; nothing to save")
        # Hard-link creation is atomic and fails rather than overwriting a file.
        os.link(temporary, destination)
        try:
            recipe = {"schema": "ml-lab.matrix-edit-recipe@1", "source_sha256": expected_sha256,
                      "coordinates": "1-based data row, 0-based column", "edits": applied}
            event = persist_transformation_provenance(
                source_path=source, derived_path=destination, recipe=recipe,
                operations=[{"type": "edit_cells", "changed_cells": len(applied),
                             "rows_before": identity["rows"], "rows_after": identity["rows"],
                             "columns_before": identity["columns"], "columns_after": identity["columns"]}],
            )
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        return {"schema": "ml-lab.matrix-draft-result@1", "path": str(destination),
                "changed_cells": len(applied), "source_sha256": expected_sha256,
                "derived_sha256": event["derived"]["sha256"], "provenance_event_id": event["event_id"]}
    finally:
        temporary.unlink(missing_ok=True)
