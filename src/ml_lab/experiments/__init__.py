"""Project-local experiment storage and opt-in task recording."""

from .artifacts import ArtifactError, ArtifactNotFound, ArtifactRegistry
from .datasets import (
    DatasetSelectionError,
    describe_selection,
    prepare_model_handoff,
    preview_experiment_dataset,
    select_experiment_dataset,
    verify_selection,
)
from .legacy import RecordedTaskError, execute_recorded_task
from .store import (
    ExperimentNotFound,
    ExperimentStore,
    ExperimentStoreError,
    InvalidRunTransition,
    SchemaVersionError,
)

__all__ = [
    "DatasetSelectionError",
    "describe_selection",
    "prepare_model_handoff",
    "preview_experiment_dataset",
    "select_experiment_dataset",
    "verify_selection",
    "ArtifactError",
    "ArtifactNotFound",
    "ArtifactRegistry",
    "ExperimentNotFound",
    "ExperimentStore",
    "ExperimentStoreError",
    "InvalidRunTransition",
    "RecordedTaskError",
    "execute_recorded_task",
    "SchemaVersionError",
]
