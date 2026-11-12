"""Bounded views over the existing A2 DataProfile engine, not a second profiler.

Builds UI-only distributions and type-aware pair previews on the same
reproducible reservoir sample. Source bytes and persisted profile artifacts
are not modified. No UI, Flask or PAH dependency.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .intake import inspect_file
from .profiling import _load_frame, profile_file

MAX_PLOT_POINTS = 240
MAX_CATEGORIES = 8


def suggested_role(column: Any) -> str:
    if column.likely_identifier:
        return 'identifier'
    if column.inferred_type in {'boolean', 'categorical'}:
        return 'discrete'
    if column.inferred_type == 'integer':
        # Small support often represents discrete numeric categories.
        if column.unique_count <= 12 and column.cardinality_ratio <= 0.35:
            return 'discrete'
        return 'continuous'
    if column.inferred_type == 'float':
        return 'continuous'
    if column.inferred_type in {'text'} and column.unique_count <= 12:
        return 'discrete'
    return 'exclude'


def _numeric(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors='coerce')
    return vals.where(np.isfinite(vals))


def _histogram(series: pd.Series) -> dict[str, Any]:
    nums = _numeric(series).dropna()
    if nums.empty:
        return {'kind': 'empty', 'bins': []}
    counts, edges = np.histogram(nums.to_numpy(dtype=float), bins=min(12, max(1, int(math.sqrt(len(nums))))))
    return {'kind': 'histogram', 'bins': [
        {'label': f'{edges[i]:.4g}–{edges[i+1]:.4g}', 'count': int(count)}
        for i, count in enumerate(counts)
    ]}


def _categories(series: pd.Series) -> dict[str, Any]:
    counts = series.dropna().astype(str).value_counts().head(12)
    return {'kind': 'categories', 'bins': [
        {'label': str(value), 'count': int(count)} for value, count in counts.items()
    ]}


def _pair_preview(frame: pd.DataFrame, left: str, right: str, roles: Mapping[str, str]) -> dict[str, Any]:
    a, b = roles[left], roles[right]
    result: dict[str, Any] = {'left': left, 'right': right, 'roles': [a, b], 'kind': 'unsupported', 'count': 0}
    if a not in {'continuous', 'discrete'} or b not in {'continuous', 'discrete'}:
        result['message'] = 'Both fields must be continuous or discrete to visualize this pair.'
        return result
    pair = frame[[left, right]].dropna()
    if a == 'continuous':
        pair[left] = _numeric(pair[left])
    if b == 'continuous':
        pair[right] = _numeric(pair[right])
    pair = pair.dropna()
    result['count'] = len(pair)
    if pair.empty:
        result['message'] = 'No complete observations are available for this pair.'
        return result
    if a == b == 'continuous':
        points = pair.sample(n=min(len(pair), MAX_PLOT_POINTS), random_state=42) if len(pair) > MAX_PLOT_POINTS else pair
        x = points[left].astype(float)
        y = points[right].astype(float)
        xlo, xhi = float(x.min()), float(x.max())
        ylo, yhi = float(y.min()), float(y.max())
        dx, dy = xhi - xlo, yhi - ylo
        result.update(kind='scatter', sampled=len(points) < len(pair),
                      x_min=xlo, x_max=xhi, y_min=ylo, y_max=yhi,
                      points=[{'x': float(36 + 520 * ((vx-xlo)/dx if dx else 0.5)),
                               'y': float(282 - 245 * ((vy-ylo)/dy if dy else 0.5))}
                              for vx, vy in zip(x, y)])
        return result
    if a == b == 'discrete':
        xvals = pair[left].astype(str)
        yvals = pair[right].astype(str)
        xcats = list(xvals.value_counts().head(MAX_CATEGORIES).index)
        ycats = list(yvals.value_counts().head(MAX_CATEGORIES).index)
        # A full crosstab allocates a dense (#unique left x #unique right)
        # matrix, although the UI displays only 8x8 cells. Keep counts sparse.
        # Retain the global max_cell, including omitted categories, so the
        # existing display scaling and visible cell values are unchanged.
        sparse = pd.DataFrame({'left': xvals, 'right': yvals}).groupby(
            ['left', 'right'], sort=False
        ).size()
        result.update(kind='contingency', x_categories=xcats, y_categories=ycats,
                      max_cell=int(sparse.max()),
                      rows=[{'label': xcat, 'cells': [int(sparse.get((xcat, ycat), 0))
                             for ycat in ycats]} for xcat in xcats],
                      truncated=xvals.nunique() > MAX_CATEGORIES or yvals.nunique() > MAX_CATEGORIES)
        return result
    category, numeric = (left, right) if a == 'discrete' else (right, left)
    cats = pair[category].astype(str)
    top = list(cats.value_counts().head(MAX_CATEGORIES).index)
    groups = []
    for label in top:
        values = pair.loc[cats == label, numeric].astype(float)
        groups.append({'label': label, 'count': len(values),
                       'q1': float(values.quantile(.25)), 'median': float(values.median()),
                       'q3': float(values.quantile(.75))})
    result.update(kind='groups', category=category, numeric=numeric, groups=groups,
                  truncated=len(cats.unique())>MAX_CATEGORIES)
    return result


def explore_features(
    path: str, *, source_sha256: str, overrides: Mapping[str, str] | None = None,
    selected: str | None = None, left: str | None = None, right: str | None = None,
    max_rows: int = 5000,
) -> dict[str, Any]:
    if not 1 <= max_rows <= 20_000:
        raise ValueError('max_rows must be 1–20,000')
    # Existing A2 profile is authoritative for inference, quality and relationships.
    profile = profile_file(path, max_rows=max_rows, relationship_rows=max_rows,
                           max_relationship_columns=20, max_relationship_pairs=0)
    if profile.source_sha256 != source_sha256:
        raise ValueError('Dataset changed since workbench selection; reload the dataset')
    record = inspect_file(path)
    if record.sha256 != source_sha256:
        raise ValueError('Dataset changed during profiling; reload the dataset')
    frame, _, _ = _load_frame(record, max_rows=max_rows, random_state=42)
    overrides = overrides or {}
    columns = []
    role_map = {}
    for col in profile.columns:
        default = suggested_role(col)
        role = overrides.get(col.name, default)
        if role not in {'continuous', 'discrete', 'identifier', 'exclude'}:
            raise ValueError('Unknown feature role')
        if role == 'continuous' and col.inferred_type not in {'integer','float'}:
            raise ValueError(f'Cannot treat nonnumeric column {col.name} as continuous')
        role_map[col.name] = role
        item = col.to_record()
        zero_rate = col.numeric_summary.get('zero_rate')
        if col.inferred_type == 'boolean':
            # The A2 profiler intentionally leaves numeric_summary empty for
            # booleans; preserve that output and expose the false/zero rate
            # only as supplementary workbench metadata.
            observed = frame[col.name].dropna().astype(str).str.strip().str.lower()
            zero_rate = (float(observed.isin({'false', 'no', 'n', '0'}).mean())
                         if len(observed) else None)
        item.update(default_role=default, role=role,
                    overridden=col.name in overrides,
                    variance=col.numeric_summary.get('variance'),
                    zero_rate=zero_rate)
        columns.append(item)
    names = list(role_map)
    selected = selected if selected in role_map else (names[0] if names else None)
    left = left if left in role_map else (names[0] if names else None)
    right = right if right in role_map and right != left else (next((n for n in names if n != left), None))
    distribution = None
    if selected is not None:
        role = role_map[selected]
        distribution = _histogram(frame[selected]) if role == 'continuous' else (
            _categories(frame[selected]) if role == 'discrete' else {'kind': 'empty', 'bins': []})
    pair = _pair_preview(frame, left, right, role_map) if left and right else None
    return {'schema': 'ml-lab.workbench-feature-explorer@1',
            'summary': profile.to_record()['summary'], 'columns': columns,
            'quality_issues': [issue.to_record() for issue in profile.quality_issues],
            'relationships': [rel.to_record() for rel in profile.relationships],
            'warnings': list(profile.warnings), 'selected': selected,
            'distribution': distribution, 'left': left, 'right': right, 'pair': pair,
            'sample_limit': max_rows}


def explore_cached_views(
    path: str, *, source_sha256: str, baseline: Mapping[str, Any],
    selected: str | None, left: str | None, right: str | None,
    max_rows: int = 5000,
) -> dict[str, Any]:
    """Refresh selection-specific plots without rerunning A2 relationships.

    ``baseline`` must come from the same verified cache identity as the active
    workbench and its role overrides. No sampled source rows are saved to SQLite.
    Source bytes are inspected anew; callers revalidate after this scan.
    """
    if not 1 <= max_rows <= 20_000:
        raise ValueError('max_rows must be 1–20,000')
    if (baseline.get('schema') != 'ml-lab.workbench-feature-explorer@1'
            or not isinstance(baseline.get('columns'), list)
            or not isinstance(baseline.get('relationships'), list)):
        raise ValueError('Invalid cached profile; recompute')
    record = inspect_file(path)
    if record.sha256 != source_sha256:
        raise ValueError('Dataset changed during profiling; reload the dataset')
    frame, _, _ = _load_frame(record, max_rows=max_rows, random_state=42)
    roles = {column['name']: column['role'] for column in baseline['columns']}
    names = list(roles)
    selected = selected if selected in roles else (names[0] if names else None)
    left = left if left in roles else (names[0] if names else None)
    right = right if right in roles and right != left else next(
        (name for name in names if name != left), None
    )
    if list(frame.columns) != names:
        raise ValueError('Cached profile columns disagree with verified source; recompute')
    distribution = None
    if selected is not None:
        role = roles[selected]
        distribution = _histogram(frame[selected]) if role == 'continuous' else (
            _categories(frame[selected]) if role == 'discrete' else {'kind': 'empty', 'bins': []}
        )
    result = dict(baseline)
    result.update(selected=selected, left=left, right=right,
                  distribution=distribution,
                  pair=_pair_preview(frame, left, right, roles) if left and right else None,
                  sample_limit=max_rows)
    result.pop('matrix', None)
    result.pop('cache', None)
    return result
