from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .intake import inspect_file, inspect_paths

_COMPARISON_SCHEMA = "ml-lab.dataset-comparison@1"
_COLLECTION_SCHEMA = "ml-lab.dataset-comparison-collection@1"


def _read_frame(path: str | Path) -> tuple[pd.DataFrame, Any]:
    source = Path(path).expanduser().resolve()
    record = inspect_file(source)
    if not record.supported or record.format != "delimited_text":
        raise ValueError(f"unsupported comparison source: {source}")
    if record.parse_status == "failed" or record.encoding is None or record.delimiter is None:
        raise ValueError(f"source could not be parsed: {record.error or source}")
    frame = pd.read_csv(
        source,
        sep=record.delimiter,
        encoding=record.encoding,
        header=0 if record.has_header else None,
        on_bad_lines="skip",
        keep_default_na=True,
    )
    if not record.has_header:
        frame.columns = list(record.columns)
    frame.columns = [str(column) for column in frame.columns]
    return frame, record


def _cell(value: Any) -> str:
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, float):
        return format(value, ".17g")
    return str(value)


def _row_counter(frame: pd.DataFrame, columns: list[str]) -> Counter[tuple[str, ...]]:
    if not columns:
        return Counter()
    return Counter(tuple(_cell(value) for value in row) for row in frame.loc[:, columns].itertuples(index=False, name=None))


def _counter_intersection_size(left: Counter[Any], right: Counter[Any]) -> int:
    return int(sum((left & right).values()))


def _schema_signature(frame: pd.DataFrame) -> str:
    payload = json.dumps([str(column) for column in frame.columns], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _identifier_candidates(frame: pd.DataFrame) -> list[str]:
    candidates: list[str] = []
    rows = len(frame)
    if rows == 0:
        return candidates
    for column in frame.columns:
        series = frame[column]
        non_missing = int(series.notna().sum())
        if non_missing == 0:
            continue
        unique = int(series.nunique(dropna=True))
        ratio = unique / non_missing
        name = str(column).casefold()
        name_hint = name == "id" or name.endswith("_id") or name.startswith("id_") or "identifier" in name or "uuid" in name
        if unique == non_missing and (ratio >= 0.98) and (name_hint or non_missing >= 10):
            candidates.append(str(column))
    return candidates


def _filename_role(path: Path) -> str | None:
    name = path.stem.casefold()
    tokens = set(name.replace("-", "_").split("_"))
    if {"train", "training"} & tokens:
        return "train"
    if {"test", "testing", "validation", "valid", "val"} & tokens:
        return "test"
    return None


def compare_files(left_path: str | Path, right_path: str | Path) -> dict[str, Any]:
    left_path = Path(left_path).expanduser().resolve()
    right_path = Path(right_path).expanduser().resolve()
    left, left_record = _read_frame(left_path)
    right, right_record = _read_frame(right_path)

    left_columns = [str(column) for column in left.columns]
    right_columns = [str(column) for column in right.columns]
    shared_columns = [column for column in left_columns if column in right_columns]
    union_columns = list(dict.fromkeys(left_columns + right_columns))
    schema_overlap = len(shared_columns) / len(union_columns) if union_columns else 1.0
    same_column_order = left_columns == right_columns
    same_schema = set(left_columns) == set(right_columns)

    left_rows = _row_counter(left, shared_columns)
    right_rows = _row_counter(right, shared_columns)
    shared_row_instances = _counter_intersection_size(left_rows, right_rows)
    left_row_count = len(left)
    right_row_count = len(right)
    left_overlap = shared_row_instances / left_row_count if left_row_count else (1.0 if right_row_count == 0 else 0.0)
    right_overlap = shared_row_instances / right_row_count if right_row_count else (1.0 if left_row_count == 0 else 0.0)
    union_instances = left_row_count + right_row_count - shared_row_instances
    row_jaccard = shared_row_instances / union_instances if union_instances else 1.0

    left_ids = _identifier_candidates(left)
    right_ids = _identifier_candidates(right)
    common_id_candidates = [column for column in left_ids if column in right_ids]
    id_column = common_id_candidates[0] if common_id_candidates else None
    id_shared = 0
    id_left_overlap = None
    id_right_overlap = None
    if id_column:
        left_values = set(_cell(value) for value in left[id_column].dropna())
        right_values = set(_cell(value) for value in right[id_column].dropna())
        id_shared = len(left_values & right_values)
        id_left_overlap = id_shared / len(left_values) if left_values else 0.0
        id_right_overlap = id_shared / len(right_values) if right_values else 0.0

    exact_duplicate = bool(left_record.sha256 and left_record.sha256 == right_record.sha256)
    content_equivalent = bool(same_column_order and left_row_count == right_row_count and left_rows == right_rows)
    left_subset = bool(same_schema and left_row_count <= right_row_count and left_overlap == 1.0)
    right_subset = bool(same_schema and right_row_count <= left_row_count and right_overlap == 1.0)

    left_role = _filename_role(left_path)
    right_role = _filename_role(right_path)
    complementary_roles = {left_role, right_role} == {"train", "test"}
    likely_split = bool(same_schema and complementary_roles and row_jaccard < 0.25)
    likely_version = bool(
        same_schema
        and not content_equivalent
        and not likely_split
        and (left_overlap >= 0.70 or right_overlap >= 0.70 or (id_left_overlap is not None and max(id_left_overlap, id_right_overlap or 0.0) >= 0.70))
    )

    if exact_duplicate:
        relationship = "exact_duplicate"
    elif content_equivalent:
        relationship = "content_equivalent"
    elif left_subset and not right_subset:
        relationship = "left_subset_of_right"
    elif right_subset and not left_subset:
        relationship = "right_subset_of_left"
    elif likely_split:
        relationship = "likely_train_test_split"
    elif likely_version:
        relationship = "likely_version_or_derivative"
    elif same_schema:
        relationship = "same_schema_distinct_data"
    elif schema_overlap >= 0.5:
        relationship = "related_schema"
    else:
        relationship = "weak_or_unknown_relationship"

    confidence_signals: list[str] = []
    if same_schema:
        confidence_signals.append("same column set")
    if same_column_order:
        confidence_signals.append("same column order")
    if row_jaccard >= 0.5:
        confidence_signals.append("substantial row overlap")
    if id_column:
        confidence_signals.append(f"shared identifier candidate: {id_column}")
    if complementary_roles:
        confidence_signals.append("train/test filename roles")

    return {
        "schema": _COMPARISON_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "left": {
            "path": str(left_path), "name": left_path.name, "sha256": left_record.sha256,
            "rows": left_row_count, "columns": left_columns, "schema_sha256": _schema_signature(left),
        },
        "right": {
            "path": str(right_path), "name": right_path.name, "sha256": right_record.sha256,
            "rows": right_row_count, "columns": right_columns, "schema_sha256": _schema_signature(right),
        },
        "schema_comparison": {
            "same_schema": same_schema,
            "same_column_order": same_column_order,
            "shared_columns": shared_columns,
            "left_only_columns": [column for column in left_columns if column not in right_columns],
            "right_only_columns": [column for column in right_columns if column not in left_columns],
            "column_jaccard": schema_overlap,
        },
        "row_comparison": {
            "shared_row_instances": shared_row_instances,
            "left_overlap_fraction": left_overlap,
            "right_overlap_fraction": right_overlap,
            "row_jaccard": row_jaccard,
            "left_subset_of_right": left_subset,
            "right_subset_of_left": right_subset,
            "content_equivalent": content_equivalent,
        },
        "identifier_comparison": {
            "column": id_column,
            "shared_identifiers": id_shared,
            "left_overlap_fraction": id_left_overlap,
            "right_overlap_fraction": id_right_overlap,
        },
        "relationship": {
            "label": relationship,
            "exact_duplicate": exact_duplicate,
            "likely_train_test_split": likely_split,
            "likely_version_or_derivative": likely_version,
            "signals": confidence_signals,
        },
    }


def compare_paths(
    paths: Iterable[str | Path], *, recursive: bool = True, include_hidden: bool = False, max_pairs: int | None = 200
) -> dict[str, Any]:
    inventory = inspect_paths(paths, recursive=recursive, include_hidden=include_hidden)
    files = [Path(record.path) for record in inventory.files if record.supported and record.parse_status != "failed"]
    comparisons: list[dict[str, Any]] = []
    truncated = False
    for left_index in range(len(files)):
        for right_index in range(left_index + 1, len(files)):
            if max_pairs is not None and max_pairs > 0 and len(comparisons) >= max_pairs:
                truncated = True
                break
            comparisons.append(compare_files(files[left_index], files[right_index]))
        if truncated:
            break
    counts: dict[str, int] = {}
    for comparison in comparisons:
        label = comparison["relationship"]["label"]
        counts[label] = counts.get(label, 0) + 1
    return {
        "schema": _COLLECTION_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "dataset_count": len(files),
            "possible_pair_count": len(files) * (len(files) - 1) // 2,
            "compared_pair_count": len(comparisons),
            "truncated": truncated,
            "relationship_counts": counts,
        },
        "datasets": [str(path) for path in files],
        "comparisons": comparisons,
        "warnings": list(inventory.warnings),
    }


def save_comparison(result: dict[str, Any], output: str | Path) -> Path:
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
