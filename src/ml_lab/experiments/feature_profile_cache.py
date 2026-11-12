"""Bounded, optional profile-preview cache inside this project's existing DB.

Only JSON summaries and up to 240 scatter dots are stored; never raw CSV rows,
model data or source tables. Cache is deliberately not a cross-project service.
The dataset's full checksum and provenance are reverified by the caller before
any cache read. v3 rollback compatibility is preserved by an optional table.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from .store import ExperimentStore, _utcnow
from .workbench import WorkbenchConflict

_CACHE_VERSION = 'ml-uw-05-v1-full-relationships'
_MAX_PER_INSTANCE = 24
_MAX_JSON_BYTES = 2_000_000


def feature_report_key(context: Mapping[str, Any], *, selected: str | None,
                       left: str | None, right: str | None,
                       overrides: Mapping[str, str]) -> str:
    dataset = context['active_dataset']
    payload = {
        'format': _CACHE_VERSION,
        'instance_id': context['instance_id'],
        'generation': context['generation'],
        'experiment_id': context['experiment_id'],
        'experiment_revision': context['experiment_revision'],
        'path': dataset['path'],
        'dataset_id': dataset['dataset_id'],
        'source_sha256': dataset['sha256'],
        'selected': selected, 'left': left, 'right': right,
        'overrides': dict(overrides),
        'max_rows': 5000,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def load_feature_report(store: ExperimentStore, context: Mapping[str, Any],
                        cache_key: str) -> dict[str, Any] | None:
    with store._connection() as db:
        result = db.execute(
            '''SELECT report_json FROM workbench_feature_profile_cache
               WHERE instance_id=? AND dataset_sha256=? AND cache_key=?''',
            (context['instance_id'], context['active_dataset']['sha256'], cache_key),
        ).fetchone()
    if result is None:
        return None
    try:
        report = json.loads(result['report_json'])
    except (TypeError, ValueError):
        return None
    if (not isinstance(report, dict)
            or report.get('schema') != 'ml-lab.workbench-feature-explorer@1'
            or not isinstance(report.get('relationships'), list)):
        return None
    return report


def save_feature_report(store: ExperimentStore, context: Mapping[str, Any],
                        cache_key: str, report: Mapping[str, Any], *,
                        preserve_cache_key: str | None = None) -> bool:
    """Write only against the same pinned workbench generation/revision."""
    serialized = json.dumps(report, sort_keys=True, separators=(',', ':'), allow_nan=False)
    if len(serialized.encode('utf-8')) > _MAX_JSON_BYTES:
        return False
    with store._connection(write=True) as db:
        current = db.execute(
            '''SELECT generation,experiment_id,experiment_revision FROM workbench_instances
               WHERE instance_id=?''', (context['instance_id'],),
        ).fetchone()
        if (current is None or current['generation'] != context['generation']
                or current['experiment_id'] != context['experiment_id']
                or current['experiment_revision'] != context['experiment_revision']):
            raise WorkbenchConflict('Workbench changed while computing feature profile; reload')
        db.execute(
            '''INSERT INTO workbench_feature_profile_cache
               (instance_id,dataset_sha256,cache_key,report_json,created_at)
               VALUES (?,?,?,?,?)
               ON CONFLICT(instance_id,dataset_sha256,cache_key)
               DO UPDATE SET report_json=excluded.report_json, created_at=excluded.created_at''',
            (context['instance_id'], context['active_dataset']['sha256'],
             cache_key, serialized, _utcnow()),
        )
        # Preserve the expensive scientific baseline when a user explores many
        # cheap view variants. All entries still count toward the same bound.
        db.execute(
            '''DELETE FROM workbench_feature_profile_cache WHERE instance_id=? AND cache_key NOT IN
               (SELECT cache_key FROM workbench_feature_profile_cache WHERE instance_id=?
                ORDER BY CASE WHEN cache_key=? THEN 1 ELSE 0 END DESC,
                         created_at DESC, cache_key DESC LIMIT ?)''',
            (context['instance_id'], context['instance_id'],
             preserve_cache_key, _MAX_PER_INSTANCE),
        )
    return True


def feature_baseline_key(context: Mapping[str, Any], *,
                         overrides: Mapping[str, str]) -> str:
    """Distinct cache namespace for expensive, selection-independent A2 work.

    The same source, generation, experiment revision and roles are part of the
    identity as for view keys. A baseline never crosses project instances.
    """
    view_key = feature_report_key(context, selected=None, left=None,
                                  right=None, overrides=overrides)
    return hashlib.sha256(('scientific-baseline:' + view_key).encode('ascii')).hexdigest()
