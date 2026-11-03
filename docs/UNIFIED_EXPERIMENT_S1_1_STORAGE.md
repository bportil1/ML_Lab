# Unified Experiment — S1.1 Metadata Store

**Historical implementation:** This document describes the S1.1 schema-v1
baseline. S1.2 adds the [artifact registry and schema-v2 migration](UNIFIED_EXPERIMENT_S1_2_ARTIFACTS.md).

**Implementation status:** Implemented as an **opt-in, standalone API** in
`ml_lab.experiments.ExperimentStore`. No existing model tasks, CLI commands,
Flask screens, or PAH calls are redirected in S1.1.

## Storage ownership and initialization

`ExperimentStore(project_root)` opens the private file
`<project_root>/analysis/ml_lab/experiments.sqlite3`. Omitting `project_root`
uses the current working directory **at construction time** for standalone
scripts. PAH integration may pass a resolved project root in a future sprint,
but PAH must never access these SQLite tables directly. There is no cross-module
or global user catalog. Large arrays, model weights, and derived datasets stay
in their existing file-based artifact stores.

The database uses foreign keys, WAL, a busy timeout, short per-operation
connections, and atomic writer transactions. The schema is identified using
SQLite `user_version=1`. Newer schema versions and nonempty unversioned databases
are refused rather than migrated or overwritten. **No backup, schema upgrade,
or production migration interface is claimed in this release.**

## Public API and state semantics

```python
from ml_lab.experiments import ExperimentStore

store = ExperimentStore("/path/to/project")
experiment = store.create_experiment(
    "RBM trial",
    dataset_ref={"dataset_id": "dataset-1", "sha256": "..."},
    model_spec={"family": "rbm", "hidden_units": 32},
    evaluation_plan={"metrics": ["reconstruction_error"]},
)
run = store.create_run(experiment["experiment_id"], seed=42)
# A real executor, once integrated, reports observed state changes:
store.update_run_status(run["run_id"], "running")
store.update_run_status(run["run_id"], "succeeded")
```

Experiment records are named, versioned metadata snapshots. Use
`revise_experiment(id, dataset_ref=..., model_spec=..., evaluation_plan=...,
metadata=...)` to create a **complete new revision**. Omitted mappings are saved
as empty objects, **not silently inherited** from the previous revision.
Existing run records continue to point at their historical revision. Use
`get_experiment(id, revision=n)` to recover that input/configuration snapshot.
The store records caller-supplied dataset references but does not independently
verify their fingerprints or load and validate the datasets in S1.1.

`create_run` records a new queued invocation, optionally pinning a historical
revision, seed, and environment metadata. `update_run_status` accepts only:

- `queued` → `running`, `succeeded`, `failed`, or `cancelled`;
- `running` → `succeeded`, `failed`, or `cancelled`;
- terminal states have no transitions.

All transitions are explicit; creating a record **does not execute, schedule,
cancel, or resume anything**. `started_at` and `finished_at` are recorded as
applicable. `succeeded` from `queued` is supported for synchronous, non-training
operations. Caller-provided errors are allowed only for failed runs. New
metadata schemas are `ml-lab.experiment@1` and `ml-lab.experiment-run@1`.

`list_experiments(limit=100, offset=0)` and
`list_runs(experiment_id, limit=100, offset=0)` support bounded pagination
(maximum 500 records per call). The store requires JSON-compatible mapping
metadata; it rejects nonserializable values rather than converting them to
unreliable string representations. The API is synchronous and does not require
Flask, PAH, PyTorch, or a network connection.

## Boundaries and follow-ups

- **S1.2** registers typed artifact references, provenance, and output files.
- **S1.3** introduces compatibility adapters around existing task execution;
  current `execute_task` payloads and envelopes remain untouched for now.
- **S2** resolves dataset references to validated feature/target inputs.
- **S4** connects actual executors, checkpoints, progress, and optional PAH
  job visibility; S1.1 does not advertise lifecycle capabilities that the
  underlying executors cannot provide.
- Independent projects use independent databases. Copying an SQLite file into
  another project is not a substitute for an artifact handoff and is outside
  the supported contract.

## Verification

Run locally: `PYTHONPATH=src pytest -q tests/test_experiment_store.py`.
Tests exercise persistence/reopen, historical snapshot pinning, project
isolation, schema safety, JSON validation, invalid transitions, and concurrent
writers without requiring optional ML or UI packages.
