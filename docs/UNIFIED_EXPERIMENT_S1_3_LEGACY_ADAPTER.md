# Unified Experiment — S1.3 Legacy Task Adapter

**Status:** Implemented opt-in headless API. No automatic interception of old
calls, no UI/Flask/CLI/API route changes, and no PAH dependency.

## Purpose and call surface

`ml_lab.application.execute_task(task, payload)` remains unchanged: task IDs,
input shapes, exception behavior, and `result`/`results` response envelopes are
identical for existing callers. Use the opt-in wrapper when persistent
experiment/run evidence is required:

```python
from ml_lab.experiments import execute_recorded_task, ExperimentStore

recorded = execute_recorded_task(
    "representation",
    {"X": [[1, 2], [2, 3], [3, 5]], "method": "pca", "config": {"n_components": 1}},
    project_root="/path/to/project",
    experiment_name="PCA baseline",
)
assert recorded["response"]["task"] == "representation"
store = ExperimentStore("/path/to/project")
assert store.get_run(recorded["run_id"])["status"] == "succeeded"
```

Return schema `ml-lab.recorded-legacy-task@1` carries `experiment_id`,
`revision`, `run_id`, `status`, the **unchanged** `response` envelope, and the
registered `artifacts` records. The wrapper calls the existing dispatcher
once, synchronously. It does not schedule jobs, invent checkpoint support,
resolve dataset IDs, or change task scientific behavior.

## Persistence and provenance

For each new invocation, a new run is created in the S1.2 project-local SQLite
store. Strict JSON-compatible input (including NumPy array/scalar values and
paths) is captured in a unique file beneath
`analysis/ml_lab/recorded_runs/<run_id>/input.json`. The complete returned
legacy envelope is stored as `result.json`. Input and result files are
registered as `payload` and `result` artifact types with SHA-256 fingerprints;
the result's direct source links include the registered input snapshot.

Large feature/target arrays are **not copied into SQLite**. The experiment's
immutable revision contains the input SHA-256, a descriptive task/config
snapshot excluding the inline feature and label arrays, and the task identity.
For directly referenced project-local **files**, the experiment revision
also pins their SHA-256 and byte size: a changed file at an unchanged path
cannot silently reuse the previous experiment revision. Project-local input
files are checked again at the end of a successful invocation on a best-effort
basis. The full input snapshot is file-based; it is not a new independent
dataset revision. Directory trees, external paths, and secondary files
referenced inside recipes are **not** yet content-pinned: a path string alone
does not guarantee that those inputs can be reproduced later.
The wrapper refuses lossy, non-JSON input (such as arbitrary objects and NaN)
rather than silently stringifying it. Task outputs must likewise be strictly
serializable to qualify as successfully recorded.

For Data Lab tasks, existing **project-local regular files** named in the
`path`/`paths` inputs are registered as `input` `dataset` artifacts. For
`data.transform.apply`, returned derived dataset, recipe, manifest, and
provenance sidecars are also registered as outputs *if* inside the project.
All other task types currently register their full JSON result, not an
invented model checkpoint or additional output file. Legacy paths outside
the selected project are still usable by `execute_task`, but the adapter will
not copy or automatically register those external files. The payload records
the path string, not its external contents; complete reproducibility therefore
requires a stable, independently managed source or a later project-local
import. The Data Lab sidecars retain their own authoritative lineage.

Optional `source_artifact_ids` may refer to already registered files in the
**same project**. These are recorded as sources of the input payload artifact.
Cross-project IDs are rejected before task execution. Each invocation has a
separate run directory; existing run snapshots are never overwritten.

## Experiment revisions and failures

By default, every call creates an independent experiment at revision 1. Supply
`experiment_id` to re-run the *same* payload on an existing experiment, with
optional `revision` to pin a historical snapshot. A stored revision must have
the same adapter identity, task, and input SHA-256; changed payloads are not
silently assigned to an old configuration. A caller can explicitly publish a
complete matching revision using `ExperimentStore.revise_experiment` or start
a new experiment. The optional `seed` argument is **metadata only**, not a
replacement for the selected algorithm's random-state configuration.

A task or registration error after run creation records `failed` with the
error text and raises `RecordedTaskError` (chained from the original exception).
The exception has `run_id` and `experiment_id` for inspection. Previously
registered input artifacts remain inspectable. A failure during output
registration can occur after the underlying task performed side effects:
`failed` means that the *recorded invocation* did not complete. No automatic
rollback or cancellation/resume promise is made. Process termination may
leave a `running` record for later reconciliation.

## Boundaries and next steps

- Importing the storage API does not require PAH/Flask. The scientific
  dispatcher and its normal optional dependencies load only when invoked.
- Existing user-facing routes and CLI commands remain unchanged; subsequent
  workflow sprints will use this opt-in adapter deliberately.
- S2 provides dataset identity/resolution, selection, and matrix edits. S3
  provides model-configuration controls; S4 provides proper training/optimizer
  jobs, family-specific files/checkpoints, and lifecycle support.
- PAH accesses public ML_Lab APIs and artifacts, **not** its private SQLite.
  HSQA_DBN retains the authoritative tangent-manifold augmentation method.

## Validation

```bash
PYTHONPATH=src python -m pytest -q tests/test_experiment_legacy_adapter.py \
  tests/test_experiment_store.py tests/test_experiment_artifacts.py \
  tests/test_flask_adapter_service.py tests/test_data_transform.py
```

Tests cover unchanged legacy envelopes, numpy input snapshotting, run reuse,
failure persistence, strict serialization, project isolation, existing Data
Lab output sidecars, source links, and concurrent invocation directories.
