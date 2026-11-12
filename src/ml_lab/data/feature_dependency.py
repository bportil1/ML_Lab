"""Read-only, bounded association matrices derived from A2 relationship records.

No statistical metric is recalculated in this display adapter. A missing value
means that A2 did not produce a valid estimate, not that the association is 0.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

METRIC_LABELS = {
    'normalized_mutual_information': 'Normalized MI',
    'mutual_information': 'Mutual information',
    'pearson': 'Pearson',
    'spearman': 'Spearman',
}
MATRIX_LIMITS = (8, 12, 20)


def dependency_matrix(
    relationships: Sequence[Mapping[str, Any]], columns: Sequence[Mapping[str, Any]],
    *, metric: str = 'normalized_mutual_information', limit: int = 12,
) -> dict[str, Any]:
    if metric not in METRIC_LABELS:
        raise ValueError('Unknown dependency metric')
    if type(limit) is not int or limit not in MATRIX_LIMITS:
        raise ValueError('Matrix size must be 8, 12 or 20')
    # Include only source-inferred A2 candidates for which a relationship was
    # attempted and returned. No identifiers/exclusions are silently coerced.
    linked = {str(item[side]) for item in relationships for side in ('left', 'right')}
    names = [str(column['name']) for column in columns if column['name'] in linked]
    names = names[:limit]
    by_pair = {frozenset((str(item['left']), str(item['right']))): item
               for item in relationships if item['left'] != item['right']}
    amplitudes = [abs(float(item[metric])) for item in relationships
                  if item.get(metric) is not None and
                  item['left'] in names and item['right'] in names
                  and math.isfinite(float(item[metric]))]
    peak = max(amplitudes, default=0.0)
    rows = []
    for left in names:
        cells = []
        for right in names:
            item = None if left == right else by_pair.get(frozenset((left, right)))
            val = item.get(metric) if item else None
            if val is not None and not math.isfinite(float(val)):
                val = None
            cells.append({
                'other': right,
                'value': float(val) if val is not None else None,
                'pair_count': int(item['pair_count']) if item else None,
                'strength': min(4, max(1, math.ceil(4 * abs(float(val)) / peak)))
                if val is not None and peak > 0 else 0,
                'sign': 'negative' if val is not None and float(val) < 0 else 'positive',
            })
        rows.append({'name': left, 'cells': cells})
    return {
        'metric': metric, 'metric_label': METRIC_LABELS[metric],
        'limit': limit, 'names': names, 'rows': rows,
        'eligible_column_count': len(linked),
        'omitted_count': max(0, len(linked) - len(names)),
        'peak_magnitude': peak,
    }
