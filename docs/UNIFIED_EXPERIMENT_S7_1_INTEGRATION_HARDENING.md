# Sprint 7.1 — Standalone / PAH integration and hardening

**Scope:** ML_Lab only. No changes to PAH, HSQA_DBN, pyPIQUE, or model mathematics.
This sprint consolidates the operational boundaries established in Sprints 0–6
without removing legacy APIs or the existing Data Lab workbenches.

## Binding the selected project

- Standalone `ml_lab.ui.create_app()` keeps its existing default: an experiment
  database under the process current working directory, or the explicitly
  configured `ML_LAB_EXPERIMENT_PROJECT_ROOT`.
- The PAH runtime adapter now **always passes the selected project's root** to
  the experiment UI, as well as the profile/derived output paths. Previously
  only the latter were configured, so hosted experiments could accidentally
  use the PAH server working directory.
- Hosted file-path autocomplete suggests locations under the selected project,
  not PAH's cwd or the account's home folder. The general Data Lab intake UI
  still supports explicitly supplied external sources where its existing
  interfaces allow that. Experiment dataset registration independently refuses
  out-of-project sources.
- Runtime `status(context=...)` indicates *running for this context* only when
  host, port, project root, and configured results root all match. Switching
  projects or results roots causes `launch` to rebuild the UI binding. A
  running server for a different project is not misreported as active for the
  newly selected one.
- A PAH shutdown stops the hosted HTTP server; it does **not** forcibly stop an
  in-progress scientific training thread. Running cancellation is cooperative,
  and explicit Linux dead-worker recovery remains available through S4.2.

## Discovery for PAH

Existing Data Lab JSON manifest discovery remains available. New experiment
outputs are discovered through **ML_Lab's own `ExperimentStore` and artifact
registry**, not by scanning unrelated project files or reading a sibling
module's SQLite database.

- Only registered **output** artifacts of **succeeded** training, optimization,
  or synthetic-data runs are published. This includes result files, recorded
  optimization histories, and generated datasets; input datasets, unfinished
  runs, and intermediate candidate files are not separately published.
- Registered outputs are verified against their recorded SHA-256 digest and
  project-scoped path before exposure. Modified, missing, or unsafe files are
  not advertised as validated artifacts.
- Registered artifacts carry stable identifiers, source run/revision,
  experiment ID, media type, digest, and producer capabilities.
- Existing Data Lab sidecars cannot point PAH discovery outside the selected
  project or configured result tree. Symlinked manifest files are ignored.
- Discovery does not create a new experiment database in projects that lack
  one. A missing custom results directory does not suppress available
  registered experiment outputs in the selected project's own database.
- Host enumeration is intentionally bounded to 500 experiments, the 100
  most recent runs per experiment, and up to 100 artifacts per run, with an
  overall 250-output cap. This is a
  discovery window, **not** a claim of exhaustive project history.

## Local regression contract

The new `tests/test_s7_1_pah_hardening.py` covers the launch configuration
using an in-process fake WSGI server (no external network, PAH installation,
or Flask dependency required), project/results-root switching, verified
training and synthetic output discovery, missing/custom results roots,
non-completed run exclusion, tampered outputs, and forged or symlinked
manifests. Existing `tests/test_pah_integration.py`, experiment storage,
artifact, training, Data Lab, and synthetic tests remain authoritative.

Run locally from the ML_Lab repository:

```bash
PYTHONPATH=src python -m pytest -q \
  tests/test_s7_1_pah_hardening.py tests/test_pah_integration.py \
  tests/test_experiment_store.py tests/test_experiment_artifacts.py \
  tests/test_experiment_training.py tests/test_experiment_synthetic.py
```

## Deliberately not claimed as complete

- Browser end-to-end testing requires the optional Flask installation and a
  manual standalone and PAH-hosted smoke test on the target machine.
- PAH's job-monitoring implementation and its thread-context propagation are
  not modified. The ML_Lab experiment status page remains the local source
  for run state; no cross-process durable queue is introduced.
- In-process execution is not resumable after a server restart. Queued
  cancellation and cooperative running cancellation retain S4.2 semantics.
- No automatic checkpoint reuse for GAN/RBM generation is added.
- Matrix editor appearance, numerical precision, and provenance behavior from
  S2.2 are unchanged; cross-repository code deduplication remains deferred.
