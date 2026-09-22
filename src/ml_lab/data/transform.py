from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .intake import inspect_file
from .provenance import persist_transformation_provenance, provenance_sidecar_path

_RECIPE_SCHEMA = "ml-lab.transformation-recipe@1"
_PREVIEW_SCHEMA = "ml-lab.transformation-preview@1"
_DERIVED_SCHEMA = "ml-lab.derived-dataset@1"
_SUPPORTED_TYPES = {"string", "integer", "float", "boolean", "datetime", "category"}
_SUPPORTED_OPERATIONS = {
    "select_columns",
    "drop_columns",
    "drop_quality_columns",
    "rename_columns",
    "reorder_columns",
    "coerce_types",
    "sentinel_to_missing",
    "fill_missing",
    "drop_missing_rows",
    "drop_duplicates",
    "filter_rows",
    "clean_strings",
    "derive",
    "encode_categorical",
    "scale",
    "outliers",
    "join",
    "melt",
    "pivot",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if pd.isna(value):
        return None
    return value


def _records(frame: pd.DataFrame, limit: int) -> dict[str, Any]:
    clipped = frame.head(max(0, limit))
    rows = [[_jsonable(value) for value in row] for row in clipped.itertuples(index=False, name=None)]
    return {"columns": [str(column) for column in frame.columns], "rows": rows, "row_count": len(frame), "shown_row_count": len(rows)}


def _normalise_recipe(recipe: Mapping[str, Any] | None) -> dict[str, Any]:
    if recipe is None:
        raw: dict[str, Any] = {}
    elif isinstance(recipe, Mapping):
        raw = deepcopy(dict(recipe))
    else:
        raise TypeError("recipe must be an object")

    operations = raw.get("operations", [])
    if not isinstance(operations, list):
        raise ValueError("recipe.operations must be a list")
    normalized: list[dict[str, Any]] = []
    for index, operation in enumerate(operations):
        if not isinstance(operation, Mapping):
            raise ValueError(f"recipe operation {index + 1} must be an object")
        item = deepcopy(dict(operation))
        kind = str(item.get("type", "")).strip()
        if kind not in _SUPPORTED_OPERATIONS:
            raise ValueError(f"unsupported transformation operation: {kind or '<missing>'}")
        item["type"] = kind
        normalized.append(item)

    return {
        "schema": _RECIPE_SCHEMA,
        "name": str(raw.get("name", "Transformation recipe")),
        "description": str(raw.get("description", "")),
        "allow_malformed_rows": bool(raw.get("allow_malformed_rows", False)),
        "operations": normalized,
    }


def _read_frame(path: str | Path, *, allow_malformed_rows: bool = False) -> tuple[pd.DataFrame, Any]:
    source = Path(path).expanduser().resolve()
    record = inspect_file(source)
    if not record.supported or record.format != "delimited_text":
        raise ValueError(f"unsupported transformation source: {source}")
    if record.parse_status == "failed" or record.encoding is None or record.delimiter is None:
        raise ValueError(f"source could not be parsed: {record.error or source}")
    if record.malformed_row_count and not allow_malformed_rows:
        raise ValueError(
            f"source contains {record.malformed_row_count} malformed row(s); "
            "repair them first or set allow_malformed_rows=true explicitly"
        )
    frame = pd.read_csv(
        source,
        sep=record.delimiter,
        encoding=record.encoding,
        header=0 if record.has_header else None,
        on_bad_lines="skip" if allow_malformed_rows else "error",
        keep_default_na=True,
    )
    if not record.has_header:
        frame.columns = list(record.columns)
    return frame, record


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, operation: str) -> list[str]:
    names = [str(name) for name in columns]
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ValueError(f"{operation}: unknown column(s): {', '.join(missing)}")
    return names


def _changed_cells(before: pd.DataFrame, after: pd.DataFrame) -> int | None:
    common = [column for column in before.columns if column in after.columns]
    if len(before) != len(after) or not common:
        return None
    left = before.loc[:, common].reset_index(drop=True)
    right = after.loc[:, common].reset_index(drop=True)
    try:
        same = left.eq(right) | (left.isna() & right.isna())
        return int((~same).to_numpy().sum())
    except Exception:
        return None


def _boolean_series(series: pd.Series, *, errors: str) -> tuple[pd.Series, int]:
    mapping = {
        "true": True, "t": True, "yes": True, "y": True, "1": True,
        "false": False, "f": False, "no": False, "n": False, "0": False,
    }
    result: list[Any] = []
    failures = 0
    for value in series:
        if pd.isna(value):
            result.append(pd.NA)
            continue
        key = str(value).strip().casefold()
        if key in mapping:
            result.append(mapping[key])
        else:
            failures += 1
            if errors == "raise":
                raise ValueError(f"cannot coerce {value!r} to boolean")
            result.append(value if errors == "ignore" else pd.NA)
    return pd.Series(result, index=series.index, dtype="object" if errors == "ignore" else "boolean"), failures


def _coerce_series(series: pd.Series, target: str, *, errors: str) -> tuple[pd.Series, int]:
    target = target.lower()
    if target not in _SUPPORTED_TYPES:
        raise ValueError(f"unsupported target type: {target}")
    if errors not in {"coerce", "raise", "ignore"}:
        raise ValueError("coerce_types errors must be coerce, raise, or ignore")

    before_non_missing = int(series.notna().sum())
    if target == "string":
        result = series.astype("string")
        return result, 0
    if target == "category":
        return series.astype("category"), 0
    if target == "boolean":
        return _boolean_series(series, errors=errors)
    if target == "datetime":
        converted = pd.to_datetime(series, errors="coerce" if errors != "raise" else "raise")
        failures = max(0, before_non_missing - int(converted.notna().sum()))
        if errors == "ignore" and failures:
            return series, failures
        return converted, failures

    converted = pd.to_numeric(series, errors="coerce" if errors != "raise" else "raise")
    failures = max(0, before_non_missing - int(converted.notna().sum()))
    if errors == "ignore" and failures:
        return series, failures
    if target == "integer":
        non_missing = converted.dropna()
        fractional = int((np.abs(non_missing - np.round(non_missing)) > 1e-12).sum())
        failures += fractional
        if fractional and errors == "raise":
            raise ValueError("non-integral values cannot be coerced to integer")
        if fractional and errors == "ignore":
            return series, failures
        if fractional:
            converted.loc[converted.notna() & (np.abs(converted - np.round(converted)) > 1e-12)] = np.nan
        return converted.round().astype("Int64"), failures
    return converted.astype(float), failures


def _apply_filter(frame: pd.DataFrame, op: Mapping[str, Any]) -> pd.DataFrame:
    column = str(op.get("column", ""))
    _require_columns(frame, [column], operation="filter_rows")
    operator = str(op.get("operator", "eq"))
    value = op.get("value")
    series = frame[column]
    if operator == "is_missing":
        mask = series.isna()
    elif operator == "not_missing":
        mask = series.notna()
    elif operator in {"contains", "not_contains"}:
        mask = series.astype("string").str.contains(str(value), case=bool(op.get("case_sensitive", False)), na=False, regex=bool(op.get("regex", False)))
        if operator == "not_contains":
            mask = ~mask
    elif operator in {"in", "not_in"}:
        values = value if isinstance(value, list) else [value]
        mask = series.isin(values)
        if operator == "not_in":
            mask = ~mask
    elif operator in {"gt", "ge", "lt", "le"}:
        numeric = pd.to_numeric(series, errors="coerce")
        threshold = float(value)
        mask = {"gt": numeric.gt, "ge": numeric.ge, "lt": numeric.lt, "le": numeric.le}[operator](threshold)
    elif operator in {"eq", "ne"}:
        mask = series.eq(value)
        if operator == "ne":
            mask = ~mask
    else:
        raise ValueError(f"filter_rows: unsupported operator {operator}")
    return frame.loc[mask.fillna(False)].copy()


def _apply_operation(frame: pd.DataFrame, op: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    kind = str(op["type"])
    before = frame.copy(deep=True)
    failures = 0
    warnings: list[str] = []

    if kind == "select_columns":
        columns = _require_columns(frame, op.get("columns", []), operation=kind)
        frame = frame.loc[:, columns].copy()

    elif kind == "drop_columns":
        columns = _require_columns(frame, op.get("columns", []), operation=kind)
        frame = frame.drop(columns=columns)

    elif kind == "drop_quality_columns":
        drop_all_missing = bool(op.get("all_missing", True))
        drop_constant = bool(op.get("constant", False))
        protected = {str(value) for value in op.get("protected_columns", [])}
        candidates: list[str] = []
        for column in frame.columns:
            name = str(column)
            if name in protected:
                continue
            series = frame[column]
            if drop_all_missing and series.isna().all():
                candidates.append(name)
                continue
            if drop_constant and series.nunique(dropna=False) <= 1:
                candidates.append(name)
        if candidates:
            frame = frame.drop(columns=candidates)

    elif kind == "rename_columns":
        mapping = {str(key): str(value) for key, value in dict(op.get("mapping", {})).items()}
        _require_columns(frame, mapping.keys(), operation=kind)
        targets = [mapping.get(str(column), str(column)) for column in frame.columns]
        if len(set(targets)) != len(targets):
            raise ValueError("rename_columns would create duplicate column names")
        frame = frame.rename(columns=mapping)

    elif kind == "reorder_columns":
        columns = _require_columns(frame, op.get("columns", []), operation=kind)
        if bool(op.get("keep_unlisted", True)):
            columns += [str(column) for column in frame.columns if str(column) not in columns]
        frame = frame.loc[:, columns].copy()

    elif kind == "coerce_types":
        mapping = {str(key): str(value) for key, value in dict(op.get("mapping", {})).items()}
        _require_columns(frame, mapping.keys(), operation=kind)
        errors = str(op.get("errors", "coerce"))
        for column, target in mapping.items():
            frame[column], column_failures = _coerce_series(frame[column], target, errors=errors)
            failures += column_failures

    elif kind == "sentinel_to_missing":
        columns = op.get("columns") or list(frame.columns)
        columns = _require_columns(frame, columns, operation=kind)
        values = op.get("values", [])
        if not isinstance(values, list):
            values = [values]
        for column in columns:
            frame[column] = frame[column].replace(values, np.nan)

    elif kind == "fill_missing":
        columns = _require_columns(frame, op.get("columns", []), operation=kind)
        method = str(op.get("method", "constant"))
        for column in columns:
            if method == "constant":
                frame[column] = frame[column].fillna(op.get("value"))
            elif method in {"mean", "median"}:
                numeric = pd.to_numeric(frame[column], errors="coerce")
                fill = numeric.mean() if method == "mean" else numeric.median()
                frame[column] = numeric.fillna(fill)
            elif method == "mode":
                mode = frame[column].mode(dropna=True)
                if not mode.empty:
                    frame[column] = frame[column].fillna(mode.iloc[0])
            elif method in {"forward", "backward"}:
                frame[column] = frame[column].ffill() if method == "forward" else frame[column].bfill()
            else:
                raise ValueError(f"fill_missing: unsupported method {method}")

    elif kind == "drop_missing_rows":
        columns = op.get("columns") or list(frame.columns)
        columns = _require_columns(frame, columns, operation=kind)
        how = str(op.get("how", "any"))
        if how not in {"any", "all"}:
            raise ValueError("drop_missing_rows how must be any or all")
        frame = frame.dropna(subset=columns, how=how).copy()

    elif kind == "drop_duplicates":
        subset = op.get("columns") or None
        if subset is not None:
            subset = _require_columns(frame, subset, operation=kind)
        keep_raw = op.get("keep", "first")
        keep: str | bool = False if keep_raw in {False, "false", "none", "None"} else str(keep_raw)
        if keep not in {"first", "last", False}:
            raise ValueError("drop_duplicates keep must be first, last, or false")
        frame = frame.drop_duplicates(subset=subset, keep=keep).copy()

    elif kind == "filter_rows":
        frame = _apply_filter(frame, op)

    elif kind == "clean_strings":
        columns = _require_columns(frame, op.get("columns", []), operation=kind)
        for column in columns:
            text = frame[column].astype("string")
            if bool(op.get("strip", True)):
                text = text.str.strip()
            if bool(op.get("collapse_whitespace", False)):
                text = text.str.replace(r"\s+", " ", regex=True)
            case = str(op.get("case", "preserve"))
            if case == "lower":
                text = text.str.lower()
            elif case == "upper":
                text = text.str.upper()
            elif case != "preserve":
                raise ValueError("clean_strings case must be preserve, lower, or upper")
            frame[column] = text

    elif kind == "derive":
        target = str(op.get("target", "")).strip()
        if not target:
            raise ValueError("derive requires target")
        method = str(op.get("method", "regex_extract"))
        if method == "regex_extract":
            source = str(op.get("source", ""))
            _require_columns(frame, [source], operation=kind)
            pattern = str(op.get("pattern", ""))
            if not pattern:
                raise ValueError("derive regex_extract requires pattern")
            extracted = frame[source].astype("string").str.extract(pattern, expand=True)
            group = int(op.get("group", 0))
            if group < 0 or group >= extracted.shape[1]:
                raise ValueError("derive regex_extract group is out of range")
            frame[target] = extracted.iloc[:, group]
        elif method == "combine_text":
            columns = _require_columns(frame, op.get("columns", []), operation=kind)
            separator = str(op.get("separator", ""))
            frame[target] = frame[columns].astype("string").fillna("").agg(separator.join, axis=1)
        elif method == "arithmetic":
            left = str(op.get("left", "")); right = str(op.get("right", ""))
            _require_columns(frame, [left, right], operation=kind)
            a = pd.to_numeric(frame[left], errors="coerce"); b = pd.to_numeric(frame[right], errors="coerce")
            operator = str(op.get("operator", "add"))
            if operator == "add": frame[target] = a + b
            elif operator == "subtract": frame[target] = a - b
            elif operator == "multiply": frame[target] = a * b
            elif operator == "divide": frame[target] = a / b.replace(0, np.nan)
            else: raise ValueError("derive arithmetic operator must be add, subtract, multiply, or divide")
        else:
            raise ValueError(f"derive: unsupported method {method}")

    elif kind == "encode_categorical":
        columns = _require_columns(frame, op.get("columns", []), operation=kind)
        method = str(op.get("method", "one_hot"))
        if method == "one_hot":
            frame = pd.get_dummies(frame, columns=columns, prefix=columns, drop_first=bool(op.get("drop_first", False)), dtype=int)
        elif method == "ordinal":
            mappings = op.get("mappings", {})
            if not isinstance(mappings, Mapping):
                raise ValueError("encode_categorical ordinal mappings must be an object")
            for column in columns:
                mapping = mappings.get(column)
                if not isinstance(mapping, Mapping):
                    raise ValueError(f"encode_categorical ordinal mapping missing for {column}")
                frame[column] = frame[column].map(mapping)
        else:
            raise ValueError("encode_categorical method must be one_hot or ordinal")

    elif kind == "scale":
        columns = _require_columns(frame, op.get("columns", []), operation=kind)
        method = str(op.get("method", "standard"))
        for column in columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            if method == "standard":
                center = values.mean(); spread = values.std(ddof=0)
                frame[column] = 0.0 if spread == 0 or pd.isna(spread) else (values - center) / spread
            elif method == "minmax":
                low = values.min(); high = values.max(); span = high - low
                frame[column] = 0.0 if span == 0 or pd.isna(span) else (values - low) / span
            elif method == "robust":
                center = values.median(); q1 = values.quantile(0.25); q3 = values.quantile(0.75); spread = q3 - q1
                frame[column] = 0.0 if spread == 0 or pd.isna(spread) else (values - center) / spread
            else:
                raise ValueError("scale method must be standard, minmax, or robust")

    elif kind == "outliers":
        columns = _require_columns(frame, op.get("columns", []), operation=kind)
        method = str(op.get("method", "clip_iqr"))
        multiplier = float(op.get("iqr_multiplier", 1.5))
        if multiplier <= 0:
            raise ValueError("outliers iqr_multiplier must be positive")
        keep_mask = pd.Series(True, index=frame.index)
        for column in columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            q1 = values.quantile(0.25); q3 = values.quantile(0.75); iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                continue
            low, high = q1 - multiplier * iqr, q3 + multiplier * iqr
            if method == "clip_iqr":
                frame[column] = values.clip(lower=low, upper=high)
            elif method == "drop_iqr":
                keep_mask &= values.between(low, high) | values.isna()
            else:
                raise ValueError("outliers method must be clip_iqr or drop_iqr")
        if method == "drop_iqr":
            frame = frame.loc[keep_mask].copy()

    elif kind == "join":
        right_path = str(op.get("right_path", ""))
        if not right_path:
            raise ValueError("join requires right_path")
        right, _ = _read_frame(right_path, allow_malformed_rows=bool(op.get("allow_malformed_rows", False)))
        kwargs: dict[str, Any] = {"how": str(op.get("how", "left")), "suffixes": tuple(op.get("suffixes", ["_x", "_y"]))}
        if op.get("on"):
            on = op.get("on"); kwargs["on"] = [str(value) for value in on] if isinstance(on, list) else str(on)
        else:
            left_on, right_on = op.get("left_on"), op.get("right_on")
            if not left_on or not right_on:
                raise ValueError("join requires on or both left_on and right_on")
            kwargs["left_on"] = [str(value) for value in left_on] if isinstance(left_on, list) else str(left_on)
            kwargs["right_on"] = [str(value) for value in right_on] if isinstance(right_on, list) else str(right_on)
        frame = frame.merge(right, **kwargs)

    elif kind == "melt":
        id_vars = _require_columns(frame, op.get("id_vars", []), operation=kind)
        value_vars_raw = op.get("value_vars")
        value_vars = _require_columns(frame, value_vars_raw, operation=kind) if value_vars_raw else None
        frame = frame.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name=str(op.get("var_name", "variable")),
            value_name=str(op.get("value_name", "value")),
        )

    elif kind == "pivot":
        index = op.get("index")
        columns = op.get("columns")
        values = op.get("values")
        if not index or not columns or not values:
            raise ValueError("pivot requires index, columns, and values")
        aggfunc = str(op.get("aggfunc", "mean"))
        if aggfunc not in {"mean", "sum", "min", "max", "first", "last", "count"}:
            raise ValueError("unsupported pivot aggfunc")
        frame = frame.pivot_table(index=index, columns=columns, values=values, aggfunc=aggfunc).reset_index()
        frame.columns = ["_".join(str(part) for part in column if str(part) != "") if isinstance(column, tuple) else str(column) for column in frame.columns]

    else:  # pragma: no cover - guarded by recipe validation
        raise ValueError(f"unsupported operation: {kind}")

    added = [str(column) for column in frame.columns if column not in before.columns]
    dropped = [str(column) for column in before.columns if column not in frame.columns]
    diagnostics = {
        "type": kind,
        "rows_before": int(len(before)),
        "rows_after": int(len(frame)),
        "columns_before": int(len(before.columns)),
        "columns_after": int(len(frame.columns)),
        "rows_delta": int(len(frame) - len(before)),
        "columns_added": added,
        "columns_dropped": dropped,
        "changed_cells": _changed_cells(before, frame),
        "coercion_failures": int(failures),
        "missing_before": int(before.isna().to_numpy().sum()),
        "missing_after": int(frame.isna().to_numpy().sum()),
        "warnings": warnings,
    }
    return frame.reset_index(drop=True), diagnostics


def transform_dataframe(frame: pd.DataFrame, recipe: Mapping[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Apply a validated, ordered transformation recipe to a DataFrame copy."""
    normalized = _normalise_recipe(recipe)
    current = frame.copy(deep=True)
    diagnostics: list[dict[str, Any]] = []
    for index, operation in enumerate(normalized["operations"]):
        current, record = _apply_operation(current, operation)
        record["index"] = index
        diagnostics.append(record)
    return current, diagnostics


def preview_transformation(
    path: str | Path,
    recipe: Mapping[str, Any],
    *,
    preview_rows: int = 50,
) -> dict[str, Any]:
    normalized = _normalise_recipe(recipe)
    frame, source_record = _read_frame(path, allow_malformed_rows=normalized["allow_malformed_rows"])
    result, diagnostics = transform_dataframe(frame, normalized)
    warnings: list[str] = []
    if source_record.malformed_row_count:
        warnings.append(f"Skipped {source_record.malformed_row_count} malformed source row(s) by explicit recipe permission.")
    return {
        "schema": _PREVIEW_SCHEMA,
        "source": {
            "path": source_record.path,
            "sha256": source_record.sha256,
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
        },
        "recipe": normalized,
        "summary": {
            "rows_before": int(len(frame)),
            "rows_after": int(len(result)),
            "columns_before": int(len(frame.columns)),
            "columns_after": int(len(result.columns)),
            "operation_count": len(diagnostics),
            "coercion_failures": int(sum(item["coercion_failures"] for item in diagnostics)),
        },
        "operations": diagnostics,
        "preview": _records(result, preview_rows),
        "warnings": warnings,
    }


def _default_output(source: Path) -> Path:
    return Path("ml_lab_results/data/derived") / f"{source.stem}-derived{source.suffix.lower()}"


def apply_transformation(
    path: str | Path,
    recipe: Mapping[str, Any],
    *,
    output: str | Path | None = None,
    overwrite: bool = False,
    preview_rows: int = 50,
) -> dict[str, Any]:
    """Apply a recipe and atomically write a new derived CSV/TSV plus recipe/manifest artifacts."""
    source = Path(path).expanduser().resolve()
    normalized = _normalise_recipe(recipe)
    frame, source_record = _read_frame(source, allow_malformed_rows=normalized["allow_malformed_rows"])
    result, diagnostics = transform_dataframe(frame, normalized)

    destination = Path(output).expanduser() if output is not None else _default_output(source)
    destination = destination.resolve()
    if destination == source:
        raise ValueError("derived dataset output cannot overwrite the raw source path")
    if destination.suffix.lower() not in {".csv", ".tsv"}:
        raise ValueError("derived dataset output must end in .csv or .tsv")
    recipe_path = destination.with_suffix(destination.suffix + ".recipe.json")
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    provenance_path = provenance_sidecar_path(destination)
    existing = [candidate for candidate in (destination, recipe_path, manifest_path, provenance_path) if candidate.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "derived output artifact already exists: " + ", ".join(str(candidate) for candidate in existing) +
            "; pass overwrite=true explicitly"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    delimiter = "\t" if destination.suffix.lower() == ".tsv" else ","
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent))
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        result.to_csv(temporary, index=False, sep=delimiter)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    recipe_path.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    created_at = datetime.now(timezone.utc).isoformat()
    transformation_warnings = ([f"Skipped {source_record.malformed_row_count} malformed source row(s)."] if source_record.malformed_row_count else [])
    provenance = persist_transformation_provenance(
        source_path=source,
        derived_path=destination,
        recipe=normalized,
        operations=diagnostics,
        recipe_path=recipe_path,
        manifest_path=manifest_path,
        warnings=transformation_warnings,
        created_at=created_at,
    )
    manifest = {
        "schema": _DERIVED_SCHEMA,
        "created_at": created_at,
        "source": provenance["source"],
        "derived": provenance["derived"],
        "recipe_path": str(recipe_path),
        "recipe": normalized,
        "operations": diagnostics,
        "warnings": provenance["warnings"],
        "provenance": {
            "schema": provenance["schema"],
            "event_id": provenance["event_id"],
            "path": provenance["provenance_path"],
            "parent": provenance["parent"],
        },
        "preview": _records(result, preview_rows),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=_jsonable) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def load_recipe(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return _normalise_recipe(raw)


__all__ = [
    "apply_transformation",
    "load_recipe",
    "preview_transformation",
    "transform_dataframe",
]
