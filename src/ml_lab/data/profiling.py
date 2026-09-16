from __future__ import annotations

import csv
import math
import random
import re
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score

from .intake import inspect_file, inspect_paths
from .types import (
    ColumnProfile,
    DataProfile,
    DataProfileCollection,
    DataQualityIssue,
    RelationshipProfile,
)

_MISSING_TOKENS = {"", "na", "n/a", "null", "none", "nan", "missing", "<na>"}
_BOOLEAN_TRUE = {"true", "yes", "y", "1"}
_BOOLEAN_FALSE = {"false", "no", "n", "0"}
_ID_NAME = re.compile(r"(^id$|_id$|^id_|identifier|uuid|guid|primary[_ -]?key|^key$)", re.IGNORECASE)
_DATE_NAME = re.compile(r"date|time|timestamp|datetime|created|updated", re.IGNORECASE)
_DATE_VALUE = re.compile(r"(?:\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b|T\d{1,2}:\d{2}|\b\d{1,2}:\d{2}(?::\d{2})?\b)")


def _clean_cell(value: str) -> str | None:
    stripped = value.strip()
    if stripped.lower() in _MISSING_TOKENS:
        return None
    return value


def _reservoir_rows(
    path: Path,
    *,
    encoding: str,
    delimiter: str,
    has_header: bool,
    expected_width: int,
    max_rows: int | None,
    random_state: int,
) -> tuple[list[list[str | None]], int, int]:
    """Read rectangular rows, optionally keeping a deterministic reservoir sample.

    Returns (profiled_rows, valid_source_rows, skipped_malformed_rows).
    """
    rng = random.Random(random_state)
    rows: list[list[str | None]] = []
    valid_rows = 0
    skipped = 0
    limit = None if max_rows is None or max_rows <= 0 else int(max_rows)

    with path.open("r", encoding=encoding, newline="", errors="strict") as handle:
        reader = csv.reader(handle, delimiter=delimiter, quotechar='"', doublequote=True, strict=True)
        if has_header:
            try:
                next(reader)
            except StopIteration:
                return rows, 0, 0
        for row in reader:
            if len(row) != expected_width:
                skipped += 1
                continue
            normalized = [_clean_cell(value) for value in row]
            valid_rows += 1
            if limit is None:
                rows.append(normalized)
            elif len(rows) < limit:
                rows.append(normalized)
            else:
                replacement = rng.randrange(valid_rows)
                if replacement < limit:
                    rows[replacement] = normalized
    return rows, valid_rows, skipped


def _load_frame(record: Any, *, max_rows: int | None, random_state: int) -> tuple[pd.DataFrame, int, int]:
    if not record.supported or record.format != "delimited_text":
        raise ValueError(f"unsupported profile source: {record.path}")
    if record.column_count is None or record.encoding is None or record.delimiter is None:
        raise ValueError(f"source could not be structurally parsed: {record.path}")
    rows, valid_rows, skipped = _reservoir_rows(
        Path(record.path),
        encoding=record.encoding,
        delimiter=record.delimiter,
        has_header=bool(record.has_header),
        expected_width=int(record.column_count),
        max_rows=max_rows,
        random_state=random_state,
    )
    frame = pd.DataFrame(rows, columns=list(record.columns), dtype=object)
    return frame, valid_rows, skipped


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _datetime_series(values: pd.Series) -> pd.Series | None:
    text = values.astype(str)
    if text.empty:
        return None
    if not (text.str.contains(_DATE_VALUE).mean() >= 0.5):
        return None
    try:
        parsed = pd.to_datetime(text, errors="coerce", utc=False)
    except (ValueError, TypeError, OverflowError):
        return None
    if parsed.notna().mean() < 0.9:
        return None
    return parsed


def _infer_type(name: str, values: pd.Series) -> tuple[str, pd.Series | None]:
    non_missing = values.dropna()
    if non_missing.empty:
        return "empty", None

    normalized = non_missing.astype(str).str.strip().str.lower()
    bool_tokens = _BOOLEAN_TRUE | _BOOLEAN_FALSE
    if set(normalized.unique()).issubset(bool_tokens):
        return "boolean", normalized.map(lambda value: value in _BOOLEAN_TRUE)

    numeric = pd.to_numeric(non_missing, errors="coerce")
    if numeric.notna().mean() >= 0.95:
        numeric = numeric.dropna().astype(float)
        finite = numeric[np.isfinite(numeric)]
        if not finite.empty and np.allclose(finite, np.round(finite), rtol=0.0, atol=1e-12):
            return "integer", numeric
        return "float", numeric

    if _DATE_NAME.search(name) or non_missing.astype(str).str.contains(_DATE_VALUE).mean() >= 0.8:
        parsed = _datetime_series(non_missing)
        if parsed is not None:
            return "datetime", parsed

    unique_count = int(non_missing.nunique(dropna=True))
    count = int(len(non_missing))
    ratio = unique_count / count if count else 0.0
    categorical_limit = max(20, min(100, int(math.sqrt(max(count, 1)) * 2)))
    if unique_count <= categorical_limit or ratio <= 0.2:
        return "categorical", None
    return "text", None


def _likely_identifier(name: str, inferred_type: str, non_missing: pd.Series, exported_indexes: set[str]) -> bool:
    count = len(non_missing)
    if count < 2:
        return False
    unique_ratio = non_missing.nunique(dropna=True) / count
    if name in exported_indexes and unique_ratio >= 0.8:
        return True
    if _ID_NAME.search(name) and unique_ratio >= 0.8:
        return True
    return inferred_type in {"integer", "text"} and unique_ratio == 1.0 and bool(_ID_NAME.search(name))


def _top_values(values: pd.Series, limit: int = 5) -> tuple[dict[str, Any], ...]:
    counts = values.dropna().astype(str).value_counts(dropna=True).head(limit)
    return tuple({"value": str(value), "count": int(count)} for value, count in counts.items())


def _numeric_summary(series: pd.Series, *, outlier_iqr_multiplier: float) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    numeric = numeric[np.isfinite(numeric)]
    if numeric.empty:
        return {}
    q1 = float(numeric.quantile(0.25))
    median = float(numeric.quantile(0.5))
    q3 = float(numeric.quantile(0.75))
    iqr = q3 - q1
    if iqr > 0:
        low = q1 - outlier_iqr_multiplier * iqr
        high = q3 + outlier_iqr_multiplier * iqr
        outliers = int(((numeric < low) | (numeric > high)).sum())
    else:
        low = q1
        high = q3
        outliers = 0
    std = float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0
    skew = _safe_float(numeric.skew()) if len(numeric) > 2 else None
    return {
        "count": int(len(numeric)),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "mean": float(numeric.mean()),
        "std": std,
        "q1": q1,
        "median": median,
        "q3": q3,
        "iqr": iqr,
        "skew": skew,
        "zero_count": int((numeric == 0).sum()),
        "negative_count": int((numeric < 0).sum()),
        "outlier_iqr_multiplier": float(outlier_iqr_multiplier),
        "outlier_lower_bound": low,
        "outlier_upper_bound": high,
        "outlier_count": outliers,
        "outlier_rate": outliers / len(numeric),
    }


def _text_summary(series: pd.Series) -> dict[str, Any]:
    text = series.dropna().astype(str)
    if text.empty:
        return {}
    lengths = text.str.len()
    return {
        "min_length": int(lengths.min()),
        "max_length": int(lengths.max()),
        "mean_length": float(lengths.mean()),
        "median_length": float(lengths.median()),
    }


def _datetime_summary(series: pd.Series) -> dict[str, Any]:
    parsed = _datetime_series(series.dropna())
    if parsed is None or parsed.empty:
        return {}
    return {"min": parsed.min().isoformat(), "max": parsed.max().isoformat()}


def _column_profiles(
    frame: pd.DataFrame,
    *,
    exported_indexes: set[str],
    outlier_iqr_multiplier: float,
) -> tuple[list[ColumnProfile], list[DataQualityIssue]]:
    profiles: list[ColumnProfile] = []
    issues: list[DataQualityIssue] = []
    total = len(frame)

    for position, name in enumerate(frame.columns):
        values = frame[name]
        non_missing = values.dropna()
        missing_count = total - len(non_missing)
        missing_rate = missing_count / total if total else 0.0
        unique_count = int(non_missing.nunique(dropna=True))
        cardinality_ratio = unique_count / len(non_missing) if len(non_missing) else 0.0
        inferred_type, _ = _infer_type(str(name), values)
        constant = bool(len(non_missing) > 0 and unique_count <= 1)
        likely_identifier = _likely_identifier(str(name), inferred_type, non_missing, exported_indexes)
        dominant_rate = 0.0
        if len(non_missing):
            dominant_rate = float(non_missing.astype(str).value_counts(normalize=True).iloc[0])

        numeric_summary = _numeric_summary(values, outlier_iqr_multiplier=outlier_iqr_multiplier) if inferred_type in {"integer", "float"} else {}
        if inferred_type in {"text", "categorical"}:
            detail_summary = _text_summary(values)
        elif inferred_type == "datetime":
            detail_summary = _datetime_summary(values)
        else:
            detail_summary = {}

        profile = ColumnProfile(
            name=str(name),
            position=position,
            inferred_type=inferred_type,
            non_missing_count=int(len(non_missing)),
            missing_count=int(missing_count),
            missing_rate=float(missing_rate),
            unique_count=unique_count,
            cardinality_ratio=float(cardinality_ratio),
            constant=constant,
            likely_identifier=likely_identifier,
            dominant_value_rate=dominant_rate,
            top_values=_top_values(values),
            numeric_summary=numeric_summary,
            detail_summary=detail_summary,
        )
        profiles.append(profile)

        if inferred_type == "empty":
            issues.append(DataQualityIssue("error", "all_missing", str(name), "column contains no non-missing values"))
        elif missing_rate >= 0.5:
            issues.append(DataQualityIssue("warning", "high_missingness", str(name), f"{missing_rate:.1%} of profiled values are missing"))
        elif missing_rate >= 0.2:
            issues.append(DataQualityIssue("info", "moderate_missingness", str(name), f"{missing_rate:.1%} of profiled values are missing"))
        if constant:
            issues.append(DataQualityIssue("warning", "constant_column", str(name), "column has one unique non-missing value"))
        if likely_identifier:
            issues.append(DataQualityIssue("info", "likely_identifier", str(name), "column appears identifier-like and is excluded from relationship profiling"))
        if inferred_type in {"categorical", "boolean"} and unique_count > 1 and dominant_rate >= 0.9:
            issues.append(DataQualityIssue("warning", "class_imbalance", str(name), f"dominant value accounts for {dominant_rate:.1%} of non-missing values"))
        if numeric_summary and numeric_summary.get("outlier_rate", 0.0) >= 0.05:
            issues.append(DataQualityIssue("info", "iqr_outliers", str(name), f"{numeric_summary['outlier_rate']:.1%} of numeric values fall outside the IQR outlier fences"))

    return profiles, issues


def _discretize(series: pd.Series, inferred_type: str) -> pd.Series:
    values = series.copy()
    if inferred_type in {"integer", "float"}:
        numeric = pd.to_numeric(values, errors="coerce")
        unique_count = int(numeric.nunique(dropna=True))
        if unique_count <= 1:
            return pd.Series([pd.NA] * len(series), index=series.index, dtype="object")
        bins = min(10, max(2, int(math.sqrt(max(int(numeric.notna().sum()), 1)))))
        try:
            return pd.qcut(numeric, q=min(bins, unique_count), duplicates="drop").astype("object")
        except (ValueError, TypeError):
            return numeric.round(8).astype("object")
    return values.astype("object")


def _relationships(
    frame: pd.DataFrame,
    columns: list[ColumnProfile],
    *,
    max_columns: int,
    max_pairs: int,
    relationship_rows: int,
    random_state: int,
) -> tuple[list[RelationshipProfile], list[str], int]:
    warnings: list[str] = []
    candidates = [
        profile
        for profile in columns
        if not profile.constant
        and not profile.likely_identifier
        and profile.inferred_type in {"integer", "float", "boolean", "categorical"}
        and profile.non_missing_count >= 3
    ]
    candidates.sort(key=lambda item: (item.missing_rate, item.position))
    if len(candidates) > max_columns:
        warnings.append(f"relationship profiling limited to {max_columns} of {len(candidates)} eligible columns")
        candidates = candidates[:max_columns]

    relation_frame = frame
    if relationship_rows > 0 and len(frame) > relationship_rows:
        relation_frame = frame.sample(n=relationship_rows, random_state=random_state, replace=False)
        warnings.append(f"relationship metrics use a deterministic {relationship_rows}-row sample")

    type_by_name = {profile.name: profile.inferred_type for profile in candidates}
    relationships: list[RelationshipProfile] = []
    for left, right in combinations([profile.name for profile in candidates], 2):
        pair = relation_frame[[left, right]].dropna()
        pair_count = len(pair)
        if pair_count < 3:
            continue

        left_type = type_by_name[left]
        right_type = type_by_name[right]
        pearson: float | None = None
        spearman: float | None = None
        if left_type in {"integer", "float"} and right_type in {"integer", "float"}:
            left_numeric = pd.to_numeric(pair[left], errors="coerce")
            right_numeric = pd.to_numeric(pair[right], errors="coerce")
            numeric_pair = pd.DataFrame({"left": left_numeric, "right": right_numeric}).dropna()
            if len(numeric_pair) >= 3 and numeric_pair["left"].nunique() > 1 and numeric_pair["right"].nunique() > 1:
                pearson = _safe_float(numeric_pair["left"].corr(numeric_pair["right"], method="pearson"))
                spearman = _safe_float(numeric_pair["left"].corr(numeric_pair["right"], method="spearman"))

        left_discrete = _discretize(pair[left], left_type)
        right_discrete = _discretize(pair[right], right_type)
        valid = left_discrete.notna() & right_discrete.notna()
        left_codes = left_discrete[valid].astype(str)
        right_codes = right_discrete[valid].astype(str)
        mi: float | None = None
        nmi: float | None = None
        if len(left_codes) >= 3 and left_codes.nunique() > 1 and right_codes.nunique() > 1:
            mi = float(mutual_info_score(left_codes, right_codes))
            nmi = float(normalized_mutual_info_score(left_codes, right_codes, average_method="arithmetic"))

        relationships.append(
            RelationshipProfile(
                left=left,
                right=right,
                pair_count=pair_count,
                pearson=pearson,
                spearman=spearman,
                mutual_information=mi,
                normalized_mutual_information=nmi,
                mi_method="quantile_discretized" if mi is not None else None,
            )
        )

    relationships.sort(
        key=lambda item: (
            -1.0 if item.normalized_mutual_information is None else item.normalized_mutual_information,
            -1.0 if item.pearson is None else abs(item.pearson),
        ),
        reverse=True,
    )
    if max_pairs > 0 and len(relationships) > max_pairs:
        warnings.append(f"relationship output limited to the strongest {max_pairs} of {len(relationships)} computed pairs")
        relationships = relationships[:max_pairs]
    return relationships, warnings, len(relation_frame)


def profile_file(
    path: str | Path,
    *,
    root: str | Path | None = None,
    max_rows: int | None = 100_000,
    relationship_rows: int = 5_000,
    max_relationship_columns: int = 25,
    max_relationship_pairs: int = 200,
    outlier_iqr_multiplier: float = 1.5,
    random_state: int = 42,
) -> DataProfile:
    if max_rows is not None and max_rows < 0:
        raise ValueError("max_rows cannot be negative; use 0 or None to profile all valid rows")
    if max_relationship_columns < 2:
        raise ValueError("max_relationship_columns must be at least 2")
    if max_relationship_pairs < 0:
        raise ValueError("max_relationship_pairs cannot be negative")
    if relationship_rows < 0:
        raise ValueError("relationship_rows cannot be negative")
    if outlier_iqr_multiplier <= 0:
        raise ValueError("outlier_iqr_multiplier must be positive")

    record = inspect_file(path, root=root)
    if not record.supported:
        raise ValueError(f"unsupported file type for profiling: {record.path}")
    if record.parse_status == "failed":
        raise ValueError(f"cannot profile structurally failed file: {record.error or record.path}")

    frame, valid_source_rows, skipped_malformed = _load_frame(record, max_rows=max_rows, random_state=random_state)
    profiled_rows = len(frame)
    sampled = profiled_rows < valid_source_rows
    warnings = list(record.warnings)
    if sampled:
        warnings.append(f"profile statistics use a deterministic reservoir sample of {profiled_rows} from {valid_source_rows} valid rows")
    if skipped_malformed:
        warnings.append(f"excluded {skipped_malformed} malformed-width row(s) from statistical profiling")

    exported_indexes = set(record.likely_exported_index_columns)
    columns, issues = _column_profiles(
        frame,
        exported_indexes=exported_indexes,
        outlier_iqr_multiplier=outlier_iqr_multiplier,
    )
    duplicate_count = int(frame.duplicated(keep="first").sum()) if len(frame) else 0
    duplicate_rate = duplicate_count / len(frame) if len(frame) else 0.0
    if duplicate_count:
        issues.append(DataQualityIssue("warning", "duplicate_rows", None, f"{duplicate_count} duplicate row(s) found in the profiled population ({duplicate_rate:.1%})"))
    if skipped_malformed:
        issues.append(DataQualityIssue("warning", "malformed_rows", None, f"{skipped_malformed} malformed-width row(s) were excluded from statistical profiling"))

    relationships, relation_warnings, relationship_row_count = _relationships(
        frame,
        columns,
        max_columns=max_relationship_columns,
        max_pairs=max_relationship_pairs,
        relationship_rows=relationship_rows,
        random_state=random_state,
    )
    warnings.extend(relation_warnings)

    return DataProfile(
        path=record.path,
        relative_path=record.relative_path,
        source_sha256=record.sha256,
        source_row_count=valid_source_rows,
        profiled_row_count=profiled_rows,
        column_count=len(frame.columns),
        sampled=sampled,
        sample_strategy="reservoir" if sampled else "full",
        relationship_row_count=relationship_row_count,
        duplicate_row_count=duplicate_count,
        duplicate_row_rate=duplicate_rate,
        columns=tuple(columns),
        quality_issues=tuple(issues),
        relationships=tuple(relationships),
        warnings=tuple(warnings),
    )


def profile_paths(
    paths: Iterable[str | Path],
    *,
    recursive: bool = True,
    include_hidden: bool = False,
    preview_rows: int = 20,
    max_rows: int | None = 100_000,
    relationship_rows: int = 5_000,
    max_relationship_columns: int = 25,
    max_relationship_pairs: int = 200,
    outlier_iqr_multiplier: float = 1.5,
    random_state: int = 42,
) -> DataProfileCollection:
    inventory = inspect_paths(
        paths,
        recursive=recursive,
        include_hidden=include_hidden,
        preview_rows=preview_rows,
    )
    profiles: list[DataProfile] = []
    warnings = list(inventory.warnings)
    for record in inventory.files:
        if not record.supported or record.parse_status == "failed":
            continue
        try:
            absolute = Path(record.path)
            relative = Path(record.relative_path)
            profile_root = absolute
            for _ in relative.parts:
                profile_root = profile_root.parent
            profiles.append(
                profile_file(
                    record.path,
                    root=profile_root,
                    max_rows=max_rows,
                    relationship_rows=relationship_rows,
                    max_relationship_columns=max_relationship_columns,
                    max_relationship_pairs=max_relationship_pairs,
                    outlier_iqr_multiplier=outlier_iqr_multiplier,
                    random_state=random_state,
                )
            )
        except (OSError, UnicodeError, csv.Error, ValueError) as exc:
            warnings.append(f"profile failed for {record.relative_path}: {type(exc).__name__}: {exc}")
    return DataProfileCollection(
        requested_paths=inventory.requested_paths,
        inventory=inventory,
        profiles=tuple(profiles),
        warnings=tuple(warnings),
    )
