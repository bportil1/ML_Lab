# Sprint 2.3 — Explicit dataset-to-experiment handoff

**Status:** Implemented opt-in UI and existing Python selection API extension.
No model training, task dispatch, legacy experiment recording, or PAH code changes.

## Why

Data Lab already creates validated derived CSV/TSV tables (including S2.2
matrix drafts) with authoritative transformation provenance. S2.1 can select
those tables through Python, but previously the user had to repeat the file
path and column mapping elsewhere. S2.3 exposes a **Use in experiment** action
on Data Lab's read-only source page, after a successful transformation, and
in the result of a saved matrix draft.

## Workflow

1. Inspect or transform a local CSV/TSV dataset, or **Edit a copy as matrix**.
2. For a saved derived dataset, choose **Use in experiment**. This action
   opens a **selection form**; saving the derived file does *not* modify any
   experiment. You can also select a project-local raw CSV/TSV source directly.
3. Create a new experiment, or pick an existing non-legacy experiment in the
   current project. When revising, the form proposes matching previously
   selected feature/target column names. Review all selected columns; renamed
   or dropped columns require explicit remapping.
4. Submit the selection. The experiment stores the exact dataset identity,
   byte SHA-256, column mapping, and any Data Lab provenance sidecar reference.
   A changed selection creates an immutable new revision, resetting old
   model/evaluation configuration. Identical selections are idempotent.
5. Later model-configuration work (Sprint 3) consumes the selected revision
   using `prepare_model_handoff(...)`. Training is **not** launched by this form.

The matrix editor remains a neutral scientific grid. The handoff action appears
**after** saving, outside the grid, rather than introducing PAH-style controls
inside the matrix itself.

## Project isolation and stale checks

Standalone launch defaults the experiment project to the process working
folder (the same default as `ExperimentStore`). Hosts can explicitly set
`experiment_project_root=` in `create_ui_blueprint`, or configure
`ML_LAB_EXPERIMENT_PROJECT_ROOT` in `create_app`. **PAH must provide the selected
project root** for a project-aware mount; there is no implicit shared database,
PAH dependency, or cross-module access. The UI does not offer experiment
selection for out-of-project files. The storage API independently enforces
project-local regular files and denies symlinked paths.

The form pins both dataset bytes and provenance SHA-256 when rendered. A stale
file or rewritten sidecar cannot be selected through an outdated form. Existing
experiments carry an `expected_revision` token, compared inside the SQLite
write transaction; a different writer's newer revision is never overwritten.
Legacy-recorded task experiments cannot be repurposed through this UI.

`ExperimentStore.revise_experiment(..., expected_revision=N)` and
`select_experiment_dataset(..., expected_revision=N)` accept the new optional
compare-and-swap argument. Callers that omit it retain the original S1.1/S2.1
API behavior; the selection UI always supplies it when revising an experiment.

The underlying Data Lab sidecar remains the authoritative transformation
record. The experiment selection references and verifies it; it does not
reimplement the transformation pipeline. The S1.2 artifact registry remains
**run-scoped**; selecting a dataset does not fabricate a training run or output
artifact.

## Scope and limitations

- UI destination picker lists up to 100 current non-legacy experiments; a full
  experiment management interface and filtering belong in Sprint 3.
- Source CSV/TSV must be fully parsed, with a header and unique columns. The
  numerical feature compatibility rules from S2.1 still apply at model handoff.
- Saving a matrix does not automatically select it. No historical run is
  deleted, rerun, or silently redirected to new data.
- Project-local output must be configured in PAH for the handoff action to
  appear after saving. Data Lab can still inspect external files, but the
  experiment catalog intentionally rejects them.
- No migrations beyond the existing experiment SQLite schema v2 are needed.
- UI tests require the optional Flask dependency. The backend selection,
  concurrency, lineage, and compatibility tests run without Flask.

## Local checks

```bash
PYTHONPATH=src python -m pytest -q \
  tests/test_experiment_dataset_handoff.py \
  tests/test_experiment_dataset_selection.py \
  tests/test_experiment_store.py \
  tests/test_experiment_dataset_ui.py
node --check src/ml_lab/ui/static/ml_lab_experiment_dataset.js
```
