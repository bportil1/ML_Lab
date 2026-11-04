# Sprint 4.2 — Bounded model optimization and cooperative run control

**Status:** Opt-in ML_Lab implementation. The PAH, pyPIQUE, HSQA_DBN,
GAN_Stabilization_Sandbox and EBGAN-Playground repositories are unchanged.

## Design

The experiment's revision, dataset SHA-256, feature/target mapping and model
configuration remain immutable. The new **Optimize saved configuration** action
runs serial Firefly optimization over exactly one scalar numeric hyperparameter
selected from the existing S3.1 typed model registry. The optimizer reuses
`ml_lab.optimization.optimize`, `FunctionEvaluator`, `SearchSpace` and the
existing `execute_task` engines. It does **not** implement or fork a second
model training algorithm, update the experiment's configured model, or promote
the best result to a new revision automatically.

Only model families with a consistent, named numeric evaluation metric are
currently exposed: **classification, regression, clustering and
representation**. RBM and GAN trials require validated metric contracts and
method-specific cost controls before enabling the same UI. No assumption is
made that Firefly is universally appropriate for every hyperparameter.

## Headless entry points

```python
from ml_lab.experiments import (
    ExperimentStore, queue_optimization_run, execute_optimization_run,
    optimize_configured_experiment, get_configured_run_status,
    request_run_cancellation, recover_interrupted_runs,
)
store = ExperimentStore("/path/to/project")
plan = {
    "parameter": "config.n_components", "lower": 1, "upper": 2,
    "metric": "rmse", "goal": "minimize", "population_size": 2,
    "iterations": 0, "seed": 42,
}
queued = queue_optimization_run(store, experiment_id, plan, expected_revision=revision)
result = execute_optimization_run(store, queued["run_id"])
```

The UI has an opt-in POST launch route, an existing run status page, verified
JSON result access, and an explicit POST cancellation action. Trial data can be
inspected from the artifact registry or optimization result directory.

The accepted plan intentionally limits evaluation count to **32** and
population to **2–8**. Iterations are **0–5**. Search parameters are selected
from numeric controls and checked using the corresponding stable model
configuration validator. Numeric union fields such as PCA `n_components`
accept either integral count bounds (>=1) or fractional thresholds in (0,1].
Neither arbitrary paths nor checkpoint locations are selectable. Model-specific
fit constraints may still reject a candidate: these failures become recorded
trials. No arbitrary Python evaluator expression is accepted from the UI.

The model `seed` from S4.1 remains *provenance only*. The Firefly population
seed is reproducible, but this patch does not impose seeds on every scientific
engine or guarantee bit-for-bit repeatability of GPU training.

## Artifacts and provenance

```
<project>/analysis/ml_lab/experiments.sqlite3
<project>/analysis/ml_lab/training_runs/<run_id>/
    optimization_input.json
    progress.json
    trial_0001.json ... trial_N.json
    optimizer_outputs/
        optimization_search_space.json
        optimization_history.csv
        firefly_history.json
        firefly_evaluations.json
        optimization_result.json
<project>/analysis/ml_lab/run_controls/<run_id>.cancel
<project>/analysis/ml_lab/run_controls/<run_id>.owner
```

The dataset and optional Data Lab provenance file are registered as inputs;
the plan manifest, each trial, and Firefly's final search-space / history /
result files are registered with SHA-256 hashes and source links. Trials contain
scalar metrics, parameter values and failure messages, **not** copies of the
entire original dataset or all model predictions. A failed optimization remains
failed even if it generated some valid trial artifacts.

## Cancellation and interruption

- **Queued:** cancellation is an atomic terminal `cancelled` transition. It
  prevents a later worker claim. No process is launched.
- **Running:** cancellation records an on-disk marker. Execution checks it
  between model runs and before committing success. An active training call
  **cannot be interrupted mid-kernel** and may continue until that call ends.
  The UI reports a request, not an immediate cancellation success.
- **Recovery:** `recover_interrupted_runs(store, experiment_id)` is an
  **explicit** conservative Linux recovery operation. Only a locally verified
  dead owner process, identified using PID and `/proc` start ticks, is marked
  failed. Unknown owners, another host, permission-limited `/proc`, and runs
  without an owner are left unchanged. It is **not** automatic restart,
  replay, checkpoint resume or a license to kill an unfamiliar process.
- Runs are never reused for retry. Start a new revision-pinned run instead.

There is no durable external queue. In-process UI worker shutdowns may still
leave queued or unverifiable running records. Worker cancellation is not a hard
kill, and optimizer progress is at the *lifecycle* boundary rather than at
every model epoch. A future job-runner sprint can improve recoverability.

## Non-goals

No PAH database sharing, no cross-module optimizer scheduling, no revised
matrix presentation, no generative sampler or tangent-manifold transplant, no
automatic best-parameter adoption, and no RBM/GAN hyperparameter optimizer UI
in this patch. These remain distinct future work items.
