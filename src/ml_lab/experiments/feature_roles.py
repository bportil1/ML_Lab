"""Metadata-only, per-instance feature interpretations for the active dataset.

Does not coerce columns, modify experiment selection, or change downstream inputs.
SQL lives in the existing per-project ExperimentStore, never a second database.
"""
from __future__ import annotations

from typing import Any

from .store import ExperimentStore, _utcnow
from .workbench import WorkbenchConflict, WorkbenchInstanceError, get_workbench_instance

_ALLOWED = frozenset({'auto', 'continuous', 'discrete', 'identifier', 'exclude'})


def feature_roles(store: ExperimentStore, context: dict[str, Any]) -> dict[str, str]:
    if context['status'] != 'ready':
        return {}
    with store._connection() as db:
        rows = db.execute(
            '''SELECT column_name, role FROM workbench_feature_roles
               WHERE instance_id = ? AND dataset_sha256 = ?''',
            (context['instance_id'], context['active_dataset']['sha256']),
        ).fetchall()
    valid = set(context['active_dataset']['column_names'])
    return {row['column_name']: row['role'] for row in rows if row['column_name'] in valid}


def set_feature_role(
    store: ExperimentStore, instance_id: str, *, column: str, role: str,
    expected_generation: int, source_sha256: str,
) -> dict[str, str]:
    context = get_workbench_instance(store, instance_id)
    if context['status'] != 'ready':
        raise WorkbenchInstanceError('No current verified dataset; reselect the source before editing feature roles')
    selected = context['active_dataset']
    if type(expected_generation) is not int or expected_generation != context['generation']:
        raise WorkbenchConflict('Workbench context changed; reload the feature profile')
    if not isinstance(source_sha256, str) or selected['sha256'] != source_sha256:
        raise WorkbenchConflict('Dataset content changed; reload the feature profile')
    if not isinstance(column, str) or column not in selected['column_names']:
        raise WorkbenchInstanceError('Unknown feature column')
    if role not in _ALLOWED:
        raise WorkbenchInstanceError('Feature role must be auto, continuous, discrete, identifier, or exclude')

    if role == 'continuous':
        # The UI proposes continuous only on parsed numeric columns. Validate
        # again at service level so direct API/POST callers cannot mislabel text.
        from ml_lab.data import profile_file
        from .datasets import _project_file

        source, _ = _project_file(store, selected['path'])
        current = profile_file(source, max_rows=5000, max_relationship_pairs=0)
        inferred = next(c.inferred_type for c in current.columns if c.name == column)
        if current.source_sha256 != selected['sha256']:
            raise WorkbenchConflict('Dataset content changed; reload the feature profile')
        if inferred not in {'integer', 'float'}:
            raise WorkbenchInstanceError('A continuous override requires a numeric feature')

    # Compare the pinned revision/generation atomically with the metadata write.
    # Changing the dataset can never race a stale form into a new selection.
    with store._connection(write=True) as db:
        row = db.execute(
            'SELECT generation, experiment_id, experiment_revision FROM workbench_instances WHERE instance_id = ?',
            (instance_id,),
        ).fetchone()
        if (row is None or row['generation'] != context['generation']
                or row['experiment_id'] != context['experiment_id']
                or row['experiment_revision'] != context['experiment_revision']):
            raise WorkbenchConflict('Workbench context changed; reload the feature profile')
        if role == 'auto':
            db.execute(
                'DELETE FROM workbench_feature_roles WHERE instance_id = ? AND dataset_sha256 = ? AND column_name = ?',
                (instance_id, source_sha256, column),
            )
        else:
            db.execute(
                '''INSERT INTO workbench_feature_roles(instance_id,dataset_sha256,column_name,role,updated_at)
                   VALUES (?,?,?,?,?)
                   ON CONFLICT(instance_id,dataset_sha256,column_name) DO UPDATE
                   SET role=excluded.role, updated_at=excluded.updated_at''',
                (instance_id, source_sha256, column, role, _utcnow()),
            )
        db.execute('DELETE FROM workbench_feature_profile_cache WHERE instance_id = ?', (instance_id,))
    return feature_roles(store, get_workbench_instance(store, instance_id))
