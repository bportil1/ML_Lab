# Unified Experiment Workflow — S0.1 Architecture Contract

**Stage:** Proposed design contract, subject to implementation in Sprints 1–7.
**Baseline:** ML_Lab 0.32.0. No new API, schema, database, or UI described here
should be interpreted as implemented in this patch.
**Inventory:** [`UNIFIED_EXPERIMENT_S0_1_INVENTORY.md`](UNIFIED_EXPERIMENT_S0_1_INVENTORY.md).
**S0.2 follow-up:** [UI conventions](UNIFIED_EXPERIMENT_S0_2_UI_CONVENTIONS.md)
and [scientific matrix editor specification](UNIFIED_EXPERIMENT_S0_2_MATRIX_EDITOR.md).

## Goal and explicit non-goals

Build **one user-facing experiment workspace** over independently testable
scientific engines. The navigational stages are **Data → Model → Training (with
optional optimization) → Evaluation → Results**. Stages may be skipped or
revisited where mathematically appropriate: PCA need not train, synthetic
sampling can create a dataset for another experiment, and optimization is
optional rather than a prerequisite.

This effort does **not** merge every algorithm into one Python package, move
HSQA-specific research algorithms into ML_Lab, introduce a PAH runtime
requirement, redesign pyPIQUE's scoring/model interpretation, or convert
read-only observations into untracked in-place edits.

## Authority and dependency rules

| Owner | May own | Must not own |
| --- | --- | --- |
| ML_Lab | Experiment lifecycle, dataset/model references, local run metadata, generic training/optimization adapters, model output and provenance schemas, standalone UI | PAH's cross-module coordination; HSQA-specific augmentation policy; pyPIQUE quality-model logic |
| PAH | Project selection/context, optional host launch, cross-module job visibility, artifact mediation and handoff | Direct writes into ML_Lab's private database or algorithm state; copied ML_Lab implementations |
| HSQA_DBN | Its domain-specific EBM experimentation, tangent-manifold augmentation implementation, validated research outputs | An implicit dependency of standalone ML_Lab |
| pyPIQUE | Quality-model inputs, MI/research interpretation, domain visualization and evaluations | An implicit ML_Lab training-engine or UI dependency |

**Data isolation:** Use ML_Lab-owned per-project run metadata (SQLite is the
planned candidate, **not yet implemented**) and file-based dataset, weight, and
output artifacts. A host can pass an explicit project/results context through
`ml_lab.pah_integration`, but no module may query another module's private DB.
Default paths must remain project-scoped; standalone execution must provide
local defaults. Avoid storing binary model state or large matrices directly in
run-record rows. PAH discovers outputs through public artifact descriptions.

**Host interface:** Retain the existing `pah.modules` and `pah.runtimes` entry
points and the `module_manifest`, `runtime_adapter`, and `discover_artifacts`
interfaces. Do not introduce PAH imports into the scientific core. Optional
`pah_jobs` integration must preserve the current standalone fallback. PAH is a
mediator, not an implementation of ML_Lab model stages.

## Proposed ML_Lab-owned records (design, not runtime schemas)

The exact serialized field types and schema versions are Sprint 1 deliverables.
The minimum *semantic* contracts are:

| Record | Minimum meaning | Identity / linkage |
| --- | --- | --- |
| Dataset reference | Source/derived location, format, shape, names/types, fingerprint and any authoritative transformation lineage | Stable content/dataset identity where already available; do not infer truth from filename alone |
| Experiment | User's repeatable goal with dataset reference, model family, config snapshot, evaluation plan and intended output collection | Stable experiment ID independent of a single execution |
| Run | One invocation of that experiment, random seed(s), resolved dependencies/environment, start/end/status, errors and outputs | Unique run ID; points to immutable experiment/config snapshot |
| Model configuration | Family and method name, typed parameters, shape/target constraints and optional training parameters | Config snapshot/version; model family remains authoritative for validation |
| Optimization trial | Evaluator, objective/metric/direction, search bounds, seed, trial history and candidate result | Linked to run; absent when an experiment has no optimizer |
| Artifact reference | Kind, schema/version, producing run, location, fingerprints and provenance/validation state | Stable artifact ID and producer identity; opaque to PAH beyond routing metadata |

**Immutability:** An experiment may be reconfigured by creating a new revision;
a completed run's resolved input/config/seed and outputs are never silently
reinterpreted. Raw source files remain non-destructive by default. A manual
edit to a dataset must create an explicit derived revision with traceable
changes; it is not a mutation of historical training evidence.

**Data handoff:** A selected dataset reference resolves to the feature matrix
`X`, and where applicable labels/targets `y`, feature names, task-specific
split/selection policy and validation results. Existing Data Lab fingerprints
and provenance are reused instead of introducing competing identity logic.
Model executors should not receive an unvalidated arbitrary UI table state.
They may continue to receive in-memory arrays through the current headless
APIs, which remain valid after the new adapters are introduced.

**Result contract:** Preserve raw task-specific reports and the semantics of
current outputs. The experiment layer wraps them with run metadata and typed
artifact references; it does not redefine clustering agreement, likelihood,
reconstruction, class scores, or optimization objectives as one generic score.
Use schema/version markers for new persisted records and plan migrations
explicitly; no claimed backward compatibility for a schema that does not yet
exist.

## Proposed state model

Workflow-stage readiness is **derived** from required input/config/artifact
availability and is not identical to job-execution status:

| Dimension | Proposed values | Meaning |
| --- | --- | --- |
| Stage | `unconfigured`, `blocked`, `ready`, `complete`, `stale` | Whether that stage can run or its recorded result matches the selected inputs |
| Execution | `queued`, `running`, `succeeded`, `failed`, `cancelled` | Outcome for a specific run/trial; `cancelled` only when the underlying engine supports it |

Revising a dataset or model config does not delete completed historical runs;
it invalidates only the active stage's *reuse claim*. A failed run remains
inspectable, and progress or checkpoints are displayed only to the extent the
specific engine supports them. No optimistic resume/cancel guarantees.

## Compatibility-first migration order

1. **S0.2 UI contract:** Declare workflow rail/status conventions and generic
   spreadsheet-like matrix behavior. Application chrome follows PAH styling;
   matrices remain neutral and pyPIQUE-inspired. pyPIQUE's existing tables are
   a presentation reference, not evidence of full cell-editing functionality.
2. **S1 experiment adapters:** Add records and local storage behind current
   `execute_task` and model public APIs. Preserve known task IDs, request
   payloads, response envelopes, CLI commands, Flask routes, and optional
   dependency/import behavior; add new workflow endpoints rather than silently
   replacing them.
3. **S2 dataset selection:** Resolve dataset IDs/derived files to validated
   feature/target inputs without asking users to manually recopy arrays.
   Keep Data Lab's XML extraction and provenance contracts intact.
4. **S3 model selection:** Register model family capabilities/validation
   separately from the old estimator-only registry; provide typed controls
   and allow family-specific advanced settings.
5. **S4 training and optimization:** Orchestrate compatible engines with
   family-aware metrics, checkpoint capabilities and job reporting; reuse
   generic Firefly only through a valid evaluator and search-space contract.
6. **S5–S6 evaluation/synthesis:** Record comparable runs and generated
   dataset outputs, distinguishing generative model training from sampling.
7. **S7 host testing:** Verify the same ML_Lab workflow standalone and via
   PAH, including artifact mediation and failure cases, before retiring legacy
   UI routes or duplicated front-end code.

## Required regression gates

- Existing `execute_task` IDs and response shapes stay intact (including the
  `result` / `results` distinctions), as do CLI entry points and standalone UI.
- Importing ML_Lab core remains independent of Flask, PAH, and torch-only extras
  when those optional dependencies are not selected.
- Dataset edits and transformations cannot silently overwrite raw inputs or
  erase recorded data provenance.
- Multiple PAH projects cannot share mutable ML_Lab run state or private
  database files by accident; parallel runs cannot overwrite each other's
  artifacts under default naming.
- Model-specific metrics are not replaced by a synthetic universal score.
- pyPIQUE and HSQA_DBN do not change as a prerequisite for this workflow.
- PAH only consumes public module APIs and artifact metadata; it does not
  reach into ML_Lab internal database tables.
- UI remains functional at useful widths without fixed side panels; matrices
  are restrained, conventional, keyboard-oriented scientific controls.

## Deferred decisions and open verification tasks

- Determine exact versioned schema, per-project SQLite schema, migration
  strategy, content hashing, artifact locking and concurrent-run policy in S1.
- Define read-only vs editable matrix behavior, selection/copy/paste, typing,
  numerical precision, pagination/virtualization and import/export in S0.2.
- Specify each family's supported training/cancellation/resume behavior before
  displaying those controls in S4.
- Verify which, if any, EBGAN-Playground methods are absent from ML_Lab before
  scheduling migration. Preserve the already adapted GAN stabilization work.
- Defer generic tangent-manifold extraction/ownership and cross-repository
  numerical-equivalence testing to the separate future consolidation effort.

## S1.1 implementation checkpoint

The optional versioned local experiment metadata store is now implemented in
`ml_lab.experiments`. See [S1.1 storage and lifecycle contract](UNIFIED_EXPERIMENT_S1_1_STORAGE.md).
It covers immutable experiment revisions, per-project databases, and caller-reported
run states only. Dataset resolution, artifact registration, model adapters, and UI
are deferred to their scheduled sprints.
