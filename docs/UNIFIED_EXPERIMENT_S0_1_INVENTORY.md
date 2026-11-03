# Unified Experiment Workflow — S0.1 Capability Inventory

**Scope:** `ML_Lab` 0.32.0 source snapshot, supplemented by the locally supplied
`Project_Assistant_Host`, `HSQA_DBN`, and `pyPIQUE` snapshots.
**Status:** Audited implementation map, not a claim that the unified workflow exists.
**Follow-up:** S0.2 defines UI behavior; Sprints 1–7 implement the workflow.

This inventory distinguishes **implemented scientific capabilities** from **current
user-facing access** and the **integration gaps**. A model's existence in Python
does not imply that it is selectable in the UI, can consume a Data Lab dataset
without re-entry, or produces a persistent experiment record.

## Existing entry points and ownership

| Surface | Current entry point | Current responsibility | Important limit |
| --- | --- | --- | --- |
| Task dispatch | `src/ml_lab/application.py::execute_task` | Headless JSON-shaped application tasks | No first-class experiment or dataset reference is passed between tasks |
| Capability catalog | `src/ml_lab/ui/service.py` | UI groups, required dependencies, sample payloads, readiness | Capabilities are displayed separately; optimization is `library`, experimental is gated |
| Core model registries | `classification/registry.py`, `regression/registry.py`, `clustering/registry.py` | Estimator discovery by task | No one registry describes all model families |
| Standalone web UI | `src/ml_lab/ui/blueprint.py`, `ui/templates/ml_lab_ui/` | Data Lab pages and generic model task pages | Most model pages use `task.html` and a JSON textarea |
| CLI | `src/ml_lab/cli.py` | Direct data, training, and model commands | Separate command invocations do not form one experiment history |
| Existing dataset artifacts | `data/provenance.py`, `data/transform.py`, `data/artifacts.py` | Logical data identities, recipes, derived output and profile artifacts | Data lineage exists; model/training lineage is not linked to it |
| Generic optimizer | `optimization/api.py::optimize` | Search-space/evaluator based Firefly optimization | No `execute_task` dispatcher entry or visual optimization workbench |
| UI profiling jobs | `ui/jobs.py::ProfileJobManager` | Asynchronous profile status and retained profile outputs | Profile-specific, in-process job lifecycle; not a shared training-run record |
| PAH boundary | `pah_integration.py` and `flask_adapter/` | Module manifest, runtime, artifact discovery, optional host integration | PAH artifact discovery recognizes a set of data schemas, not a generic experiment bundle |
| Optional experimental catalog | `experimental/catalog.py`, `experimental/registry.py` | Research pipelines behind explicit opt-in | Must not be implicitly treated as stable model families |

## Capability-to-workflow map

**UI access** describes this snapshot, not a proposed feature.

| Capability | Implemented owner / API | Current access | Proposed workflow position | Integration requirement |
| --- | --- | --- | --- | --- |
| CSV/TSV inventory, source browsing | `data/intake.py`, `data/table.py` | Typed Data Lab UI, CLI, `data.inspect`/`data.table` | Data | Preserve read-only raw source and row/column semantics |
| Dataset profiling | `data/profiling.py`, `data/artifacts.py` | Data Lab UI with profile jobs, CLI, `data.profile` | Data | Select a profiled dataset for subsequent model stages |
| Comparison | `data/comparison.py` | Data Lab UI, CLI, `data.compare` | Data / Evaluation | Keep heuristic comparisons distinct from authoritative lineage |
| Derived data and lineage | `data/transform.py`, `data/provenance.py` | Data Lab UI, CLI, `data.transform.*`, `data.lineage` | Data | Carry stable derived dataset identity and recipe into an experiment |
| XML structure and tabular extraction | `data/xml_*.py` | Specialized Data Lab UI and API | Data | Preserve existing collection, recipe, export and provenance flow; hand off materialized tables only |
| Classification | `classification/api.py` | `classification` task, CLI, JSON UI | Model → Evaluation | Target selection, splits, presets, run/artifact records |
| Regression | `regression/api.py` | `regression` task, CLI, JSON UI | Model → Evaluation | Preserve single- and multi-target semantics |
| Clustering | `clustering/api.py` | `clustering` task, CLI, JSON UI | Model → Evaluation | Account for unsupervised metrics, stability, agreement |
| PCA | `representation/api.py`, `representation/pca.py` | `representation` task, CLI, JSON UI | Model / Optional representation step | No forced training stage; retain latent and reconstruction output |
| MLP / Transformer autoencoders | `representation/` and `neural/` | `representation` task, CLI, JSON UI; optional neural dependency | Model → Training → Evaluation | Handle learned representations, training history, checkpoints |
| GAN (MLP / Transformer) | `generative/gan/api.py` | `gan` task, CLI, JSON UI; optional neural dependency | Generative model → Training → Sampling | Keep model training distinct from producing reusable synthetic datasets |
| Bernoulli / Gaussian / Student-t PoE RBM | `energy_based/rbm/`, `energy_based/training/` | `rbm` task and `rbm-train` CLI; optional energy dependency | EBM model → Training → Evaluation / Sampling | Preserve family-specific diagnostics and checkpoints |
| Firefly optimizer | `optimization/` | Python library API | Optional optimization loop | Define compatible evaluators, trials, objective direction and outputs |
| Experimental research modules | `experimental/` | Explicit experimental activation; not stable task UI | Opt-in experiments | No promotion or migration without method-specific review |

### What is **not** yet implemented as a unified contract

1. No shared `ExperimentSpec` or `RunRecord` coordinates dataset → model →
   training/optimization → evaluation → result across families.
2. A Data Lab transformation writes recorded datasets, but the generic model UI
   still expects separate payloads with `X` and (where applicable) `y`.
3. Result shapes differ by task: classification/regression return `results`,
   clustering returns `results` plus agreement records, and representation/GAN/
   RBM return `result`. That distinction is an existing compatibility contract.
4. Model output persistence, checkpoint formats, and error/status reporting are
   not unified merely because individual engines support these features.
5. PAH's discovery of known Data Lab JSON schemas does not automatically make
   arbitrary model results discoverable as reusable experiment artifacts.
6. Generic optimization cannot be assumed valid for every model family; a
   model-specific evaluator and objective must explicitly support it.
7. The grid implementation in `ui/static/ml_lab_tables.js` and
   `ui/static/ml_lab_transform_sheet.js` is part of the Data Lab UI; it is not
   a shared editable numerical matrix contract used by all model families.

## Research-code boundary: confirm before consolidation

- **GAN Stabilization Sandbox:** `ORIGIN_NOTES.md` documents adoption of
  architectural and stabilization ideas into stable GAN and experimental
  `experimental/gan_stabilization/` implementations. Do not reschedule a blanket
  transplant from the original sandbox.
- **EBGAN-Playground:** This source snapshot does not establish feature-by-feature
  EBGAN equivalence. Generic GAN and RBM support is not proof of a specific
  EBGAN architecture. Verify concrete missing methods only in the later review.
- **Tangent methods are different:** `experimental/gan_stabilization/tangent.py`
  contains discriminator-score tangent perturbations. The supplied
  `HSQA_DBN/data/augmentation.py::generate_tangent_samples` uses learned
  encoding/decoding for augmentation. Neither should be silently substituted
  for the other; HSQA_DBN keeps ownership of its current production/research
  augmentation pipeline.
- **pyPIQUE:** Reference for workflow and neutral matrix/table presentation,
  not an ML_Lab runtime dependency. Its downstream model ingestion must remain
  unchanged during this sprint collection.

## Acceptance checklist for S0.1

- [x] Identify public headless tasks, model families, current UI access, and
      existing data/artifact mechanisms.
- [x] Distinguish implemented features from missing experiment orchestration.
- [x] Record host-independent ownership and research-code boundaries.
- [x] Record contract migration requirements in the companion architecture doc.
- [ ] Execute repository-specific integration and UI acceptance tests in later
      implementation sprints; S0.1 is intentionally documentation-only.

**Next:** [`UNIFIED_EXPERIMENT_S0_1_CONTRACT.md`](UNIFIED_EXPERIMENT_S0_1_CONTRACT.md)
for the design decisions and nonbreaking migration rules.
