from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Mapping

from .intake import inspect_file


def _normalise(value: Any) -> str:
    return "" if value is None else str(value)


def _matches(row: list[str], *, search: str, filters: Mapping[str, str], columns: tuple[str, ...]) -> bool:
    if search:
        needle = search.casefold()
        if not any(needle in _normalise(value).casefold() for value in row):
            return False
    if filters:
        indexes = {name: index for index, name in enumerate(columns)}
        for name, raw_filter in filters.items():
            if not raw_filter:
                continue
            index = indexes.get(name)
            if index is None:
                continue
            value = row[index] if index < len(row) else ""
            if raw_filter.casefold() not in _normalise(value).casefold():
                return False
    return True


def _sort_value(value: str) -> tuple[int, Any]:
    text = value.strip()
    if text == "":
        return (2, "")
    try:
        number = float(text)
    except ValueError:
        return (1, text.casefold())
    if math.isfinite(number):
        return (0, number)
    return (1, text.casefold())


def read_table_page(
    path: str | Path,
    *,
    page: int = 1,
    page_size: int | None = 50,
    search: str = "",
    filters: Mapping[str, str] | None = None,
    sort_column: str | None = None,
    sort_direction: str = "asc",
) -> dict[str, Any]:
    """Read an accessible page from a supported delimited source.

    Pagination limits only presentation. ``page_size=None`` returns every matching
    row, and no source row is permanently hidden by this API. Filtering is a
    case-insensitive substring match. Sorting keeps numeric values numeric where
    possible and otherwise falls back to case-insensitive text ordering.
    """
    if page < 1:
        raise ValueError("page must be at least 1")
    if page_size is not None and page_size < 1:
        raise ValueError("page_size must be positive or None for all rows")
    direction = sort_direction.lower()
    if direction not in {"asc", "desc"}:
        raise ValueError("sort_direction must be 'asc' or 'desc'")

    record = inspect_file(path)
    if not record.supported or record.format != "delimited_text":
        raise ValueError(f"unsupported source table: {record.path}")
    if record.parse_status == "failed" or record.encoding is None or record.delimiter is None:
        raise ValueError(f"source table could not be parsed: {record.error or record.path}")

    columns = tuple(record.columns)
    if sort_column is not None and sort_column not in columns:
        raise ValueError(f"unknown sort column: {sort_column}")
    active_filters = {str(key): str(value) for key, value in (filters or {}).items() if str(value)}
    unknown_filters = sorted(set(active_filters).difference(columns))
    if unknown_filters:
        raise ValueError("unknown filter column(s): " + ", ".join(unknown_filters))

    expected_width = int(record.column_count or 0)
    all_matching: list[list[str]] | None = [] if (sort_column is not None or page_size is None) else None
    page_rows: list[list[str]] = []
    matched_count = 0
    valid_count = 0
    skipped_malformed = 0
    start = (page - 1) * page_size if page_size is not None else 0
    stop = start + page_size if page_size is not None else None

    with Path(record.path).open("r", encoding=record.encoding, newline="", errors="strict") as handle:
        reader = csv.reader(handle, delimiter=record.delimiter, quotechar='"', doublequote=True, strict=True)
        if record.has_header:
            try:
                next(reader)
            except StopIteration:
                pass
        for row in reader:
            if len(row) != expected_width:
                skipped_malformed += 1
                continue
            valid_count += 1
            if not _matches(row, search=search.strip(), filters=active_filters, columns=columns):
                continue
            if all_matching is not None:
                all_matching.append(list(row))
            elif matched_count >= start and (stop is None or matched_count < stop):
                page_rows.append(list(row))
            matched_count += 1

    if all_matching is not None:
        if sort_column is not None:
            index = columns.index(sort_column)
            all_matching.sort(key=lambda row: _sort_value(row[index]), reverse=(direction == "desc"))
        matched_count = len(all_matching)
        if page_size is None:
            page_rows = all_matching
            page = 1
        else:
            page_count = max(1, math.ceil(matched_count / page_size)) if matched_count else 1
            page = min(page, page_count)
            start = (page - 1) * page_size
            page_rows = all_matching[start : start + page_size]

    page_count = 1 if page_size is None else (max(1, math.ceil(matched_count / page_size)) if matched_count else 1)
    first_row = 0 if not page_rows else ((page - 1) * (page_size or matched_count) + 1)
    last_row = 0 if not page_rows else first_row + len(page_rows) - 1
    return {
        "schema": "ml-lab.data-table-page@1",
        "path": record.path,
        "relative_path": record.relative_path,
        "columns": list(columns),
        "rows": page_rows,
        "page": page,
        "page_size": page_size,
        "page_count": page_count,
        "matched_row_count": matched_count,
        "valid_source_row_count": valid_count,
        "skipped_malformed_row_count": skipped_malformed,
        "first_row": first_row,
        "last_row": last_row,
        "search": search,
        "filters": active_filters,
        "sort_column": sort_column,
        "sort_direction": direction,
    }
