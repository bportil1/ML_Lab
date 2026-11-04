# Unified Experiment — Sprint 2.1 Dataset Selection and Model Handoff

**Status:** Implemented opt-in headless Python API. No existing Data Lab routes,
legacy scientific task envelopes, job runners, or PAH integration routes change.
Sprints 2.2–2.3 will supply the visible matrix editing and transformation-to-experiment
workflow; S3 provides unified model configuration and execution.

## Why this exists

Previously, Data Lab could inspect and derive datasets, but model tasks accepted
inline `X`/`y` arrays with no reference to the inspected source. S2.1 establishes
a pinned, project-local dataset selection and a bridge to those same task input
shapes. It uses **Data Lab's authoritative inspection, table loading, identity,
and transformation provenance**, rather than creating a second CSV parser or
inventing another dataset lineage scheme.

### Minimal example

```python
from ml_lab.experiments import (
    ExperimentStore, select_experiment_dataset,
    preview_experiment_dataset, prepare_model_handoff,
)

store = ExperimentStore("/path/to/project")
experiment = select_experiment_dataset(
    store,
    "datasets/measurements.csv",
    experiment_name="Measurements baseline",
    feature_columns=["temperature", "pressure"],
    target_columns=["failure_type"],
)
page = preview_experiment_dataset(store, experiment["experiment_id"], page_size=25)
handoff = prepare_model_handoff(
    store, experiment["experiment_id"], "classification",
    settings={"estimators": ["ridge"], "config": {"cv_folds": 2}},
)
# handoff["payload"] is compatible with ml_lab.application.execute_task.
# S3.1 attaches a validated model configuration; S4 will add recorded execution on the configured experiment.
```

The API also selects transformed CSV/TSV outputs, including a Data Lab
transformation's `.provenance.json` sidecar, without copying the source file.
XML must first be materialized as a tabular dataset through Data Lab's existing
XML extraction flow. No implicit XML flattening is performed.

## Data identity and pinning

`ml-lab.experiment-dataset@1` stores only:

- Project-relative path; Data Lab logical dataset ID and exact file SHA-256.
- Logical table fingerprint, row/column shape, columns and inferred dtypes.
- Explicit ordered `feature_columns` and `target_columns`.
- If present, the existing Data Lab transformation event ID, sidecar relative
  path, and sidecar SHA-256, verified against the derived dataset identity.

Only existing, non-symlinked files contained within the supplied project are
eligible. Paths with `..`, external files, directories, XML sources, malformed
CSV/TSV records, empty tables, duplicate/ambiguous columns, and missing
feature selections are rejected. The original file is not imported, changed,
truncated, copied, or automatically corrected. A project-local dataset can be
inspected and selected in either standalone or PAH-hosted use, provided the
caller passes the correct project to `ExperimentStore`.

A previously pinned selection is revalidated before preview or handoff. An
edited file or altered provenance sidecar is treated as **stale**: the caller
must explicitly select a new experiment revision. An identical selection is
idempotent. A changed selection publishes a new immutable revision, resetting
model/evaluation configuration while preserving completed historical runs.
Legacy-recorded task experiments are not silently repurposed.

`preview_experiment_dataset` uses Data Lab's bounded table-page API. It never
stores a filtered or sorted viewport as the model input. `prepare_model_handoff`
reads the original full dataset in source order, retains the selected feature
order, and returns the task's expected numeric `X`, feature names, and optional
labels. No arbitrary row sampling, missing-value imputation, feature encoding,
scaling, or coercion to a nonnumeric model feature is performed silently.

## Supported first-pass model mapping

| Task | Features | Targets |
|---|---|---|
| `classification` | numeric, finite 2-D `X` | exactly one explicit label column as `y` |
| `regression` | numeric, finite 2-D `X` | exactly one numeric target as `y` |
| `clustering` | numeric, finite 2-D `X` | optional reference-label column as `y_true` |
| `representation`, `gan`, `rbm` | numeric, finite 2-D `X` | none |

Multiple regression targets and sequence-shaped data remain future typed
configuration work, not accidental behavior. Additional settings are passed
through unchanged but cannot replace `X`, `y`, or the chosen column mapping.
Unsupported task IDs and nonfinite or nonnumeric features are rejected prior
to model execution.

## Boundaries and upcoming work

- This is a headless dataset-selection contract and input materialization
  layer. It does **not** start training or bind model runs to the selection.
  Do not call the existing legacy recorder with a selected experiment ID: the
  S1.3 legacy adapter's experiment revisions use a distinct schema.
- No new SQLite schema/table is necessary: the existing `dataset_ref_json`
  field holds the pinned selection. Matrices remain in project files and
  runtime memory, not in SQLite.
- S2.2 adds pyPIQUE-style generic matrix editing with derived revisions.
  S2.3 integrates selected transformed outputs into the visible handoff.
  S3.1 adds headless validated model configuration; S3.2 adds the UI; S4 records model runs.
- PAH remains a mediator, not a code/database dependency; HSQA_DBN retains
  ownership of its tangent-manifold research implementation.

## Local validation

```bash
PYTHONPATH=src python -m pytest -q tests/test_experiment_dataset_selection.py \
    tests/test_experiment_store.py tests/test_experiment_artifacts.py \
    tests/test_experiment_legacy_adapter.py tests/test_data_transform.py
```
