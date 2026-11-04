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
from .models import (
    ModelConfigurationError,
    configure_experiment_model,
    describe_model_method,
    list_model_families,
    list_model_methods,
    prepare_configured_model_handoff,
    validate_model_configuration,
)
from .training import (
    ConfiguredRunError, execute_configured_run, get_configured_run_status,
    queue_configured_run, run_configured_experiment,
)
from .optimization import (
    OPTIMIZER_EXECUTOR, OPTIMIZER_SCHEMA, execute_optimization_run,
    optimize_configured_experiment, queue_optimization_run,
    validate_optimization_plan, list_optimization_controls, optimization_metrics,
)
from .comparison import COMPARISON_SCHEMA, compare_experiment_runs, comparison_csv
from .visualizations import (VISUALIZATION_SCHEMA, build_visualization_report,
                             chart_csv, chart_svg)
from .run_control import request_run_cancellation, recover_interrupted_runs
from .synthetic import (SYNTHETIC_EXECUTOR, SYNTHETIC_SCHEMA, ALL_METHODS,
                        validate_synthetic_plan, queue_synthetic_run,
                        execute_synthetic_run, generate_synthetic_dataset)
from .store import (
    ExperimentNotFound,
    ExperimentStore,
    ExperimentStoreError,
    InvalidRunTransition,
    SchemaVersionError,
)

__all__ = [
    "COMPARISON_SCHEMA", "compare_experiment_runs", "comparison_csv",
    "VISUALIZATION_SCHEMA", "build_visualization_report", "chart_csv", "chart_svg",
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
    "OPTIMIZER_EXECUTOR", "OPTIMIZER_SCHEMA", "queue_optimization_run",
    "execute_optimization_run", "optimize_configured_experiment", "validate_optimization_plan",
    "list_optimization_controls", "optimization_metrics",
    "request_run_cancellation", "recover_interrupted_runs",
    "SYNTHETIC_EXECUTOR", "SYNTHETIC_SCHEMA", "ALL_METHODS",
    "validate_synthetic_plan", "queue_synthetic_run", "execute_synthetic_run",
    "generate_synthetic_dataset",
    "ConfiguredRunError", "queue_configured_run", "execute_configured_run",
    "run_configured_experiment", "get_configured_run_status",
    "ModelConfigurationError",
    "configure_experiment_model",
    "describe_model_method",
    "list_model_families",
    "list_model_methods",
    "prepare_configured_model_handoff",
    "validate_model_configuration",
]
