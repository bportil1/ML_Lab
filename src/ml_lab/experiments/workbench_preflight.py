"""Read-only pyPIQUE-style artifact preflight for a project-local ML_Lab workbench.

A saved click sequence is never proof that work is complete. Inspect existing
experiment revisions and derived outputs; verify source/provenance before
labelling something reusable. This module never starts a job or selects data.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from .datasets import DatasetSelectionError, verify_selection
from .store import ExperimentStore, ExperimentStoreError

# Small saved datasets may be fully verified during the preflight. Larger
# ones are discoverable without an expensive automatic read on every page.
_PREFLIGHT_VERIFY_BYTES = 4 * 1024 * 1024


def _derived_candidates(root: Path, *, limit: int = 8, examined_limit: int = 128) -> list[str]:
    """List bounded project-owned derived tables without parsing/reprocessing them.

    Ignore symlinks and never descend outside the project root. This discovery
    is intentionally narrow: other data should be selected using Tabular Data.
    """
    derived = root / "analysis" / "data" / "derived"
    if not derived.is_dir() or derived.is_symlink():
        return []
    found: list[tuple[int, str]] = []
    stack: list[tuple[Path, int]] = [(derived, 0)]
    examined = 0
    while stack and examined < examined_limit:
        directory, depth = stack.pop()
        try:
            with os.scandir(directory) as children:
                entries = sorted(children, key=lambda entry: entry.name)
        except OSError:
            continue
        for entry in entries:
            examined += 1
            if examined > examined_limit:
                break
            try:
                if entry.is_symlink() or entry.name.startswith('.'):
                    continue
                item = Path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    if depth < 3:
                        stack.append((item, depth + 1))
                elif entry.is_file(follow_symlinks=False) and item.suffix.lower() in {'.csv', '.tsv'}:
                    stat = entry.stat(follow_symlinks=False)
                    if stat.st_size:
                        found.append((stat.st_mtime_ns, item.relative_to(root).as_posix()))
            except (OSError, ValueError):
                continue
    found.sort(reverse=True)
    return [path for _, path in found[:limit]]


def inspect_existing_work(store: ExperimentStore, context: Mapping[str, Any],
                          *, experiment_limit: int = 6, derived_limit: int = 6) -> dict[str, Any]:
    """Inspect saved artifacts without training, extraction or dataset mutation.

    An existing experiment is marked *ready* only after a full fingerprint and
    lineage check. A standalone derived file is marked *found*, not verified,
    until the existing dataset selection flow checks it on explicit use.
    """
    active = context.get('active_dataset') if context.get('status') == 'ready' else None
    saved: list[dict[str, Any]] = []
    known_paths: set[str] = set()
    for item in store.list_experiments(limit=experiment_limit):
        dataset = item.get('dataset_ref')
        if not isinstance(dataset, dict) or not dataset.get('path'):
            continue
        entry = {
            'experiment_id': item['experiment_id'],
            'revision': item['revision'],
            'name': item['name'],
            'path': dataset['path'],
            'status': 'stale',
            'reason': None,
        }
        try:
            from .datasets import _project_file
            relative = dataset['path']
            if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or '..' in Path(relative).parts:
                raise DatasetSelectionError('Invalid saved dataset path')
            verified_file, _ = _project_file(store, relative)
            if not verified_file.is_file():
                raise DatasetSelectionError('Saved dataset is missing')
            if verified_file.stat().st_size > _PREFLIGHT_VERIFY_BYTES:
                # This is not a trusted usable dataset until its existing
                # activation preview performs the complete integrity check.
                entry['status'] = 'found'
                entry['reason'] = 'Large saved file; verify on selection'
                known_paths.add(relative)
            else:
                confirmed = verify_selection(store, dataset)
                entry['status'] = 'ready'
                entry['rows'] = confirmed['rows']
                entry['columns'] = confirmed['columns']
                known_paths.add(confirmed['path'])
        except (DatasetSelectionError, ExperimentStoreError, OSError, ValueError) as exc:
            entry['reason'] = str(exc)
        saved.append(entry)
    derived = [
        {'path': path, 'status': 'found', 'reason': 'Verify before activation'}
        for path in _derived_candidates(store.project_root, limit=derived_limit)
        if path not in known_paths
    ]
    has_active = active is not None
    profile_cached = False
    recent_runs: list[dict[str, Any]] = []
    if has_active:
        # Reuse the existing verified A2 summary without recalculating A2.
        from .feature_profile_cache import (
            feature_baseline_key, load_feature_report,
        )
        from .feature_roles import feature_roles
        baseline = feature_baseline_key(context, overrides=feature_roles(store, dict(context)))
        profile_cached = load_feature_report(store, context, baseline) is not None
        if context.get('experiment_id'):
            for run in store.list_runs(context['experiment_id'], limit=5):
                executor = run.get('environment', {}).get('executor')
                if executor in {'ml_lab.experiments.training@1',
                                'ml_lab.experiments.optimization@1',
                                'ml_lab.experiments.synthetic@1'}:
                    recent_runs.append({'run_id': run['run_id'], 'status': run['status'],
                                        'revision': run['revision']})
    has_reusable = any(x['status'] in {'ready', 'found'} for x in saved)
    return {
        'schema': 'ml-lab.artifact-preflight@1',
        'active_status': context.get('status', 'empty'),
        'active': {'path': active['path'], 'rows': active['rows'],
                   'columns': active['columns']} if has_active else None,
        'saved': saved,
        'derived': derived,
        'profile_cached': profile_cached,
        'recent_runs': recent_runs,
        'steps': [
            {'id':'data', 'label':'Data',
             'state':'complete' if has_active else ('ready' if (has_reusable or derived) else 'blocked')},
            {'id':'profiling', 'label':'Feature Profiling',
             'state':('complete' if profile_cached else 'ready') if has_active else 'blocked'},
            {'id':'representation', 'label':'Representation',
             'state':'ready' if has_active else 'blocked'},
            {'id':'comparison', 'label':'Comparison',
             'state':'ready' if has_active else 'blocked'},
            {'id':'tools', 'label':'General ML Tools',
             'state':'ready' if has_active else 'blocked'},
        ],
        'note': 'Saved outputs were inspected; no work was executed or selected.',
    }
