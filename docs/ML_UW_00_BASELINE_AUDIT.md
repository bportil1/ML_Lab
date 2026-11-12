# ML-UW-00 — Unified Research Workbench baseline audit

**Source audited:** supplied `ML_Lab-main(3).zip`, `ml-lab` version **0.32.0** (as declared in `pyproject.toml` and `src/ml_lab/__init__.py`). This audit covers the supplied source snapshot, **not** a claim about a separate installed or locally modified checkout. Reconcile `git status`, installed package origin, and host-mounted PAH chrome before modifying the actual working tree.

**Scope:** source inventory, ownership map, migration contract, and acceptance gates. **No production behavior, database schema, route, or UI layout is modified in ML-UW-00.**

## Findings: existing capabilities to preserve

| Concern | Current owner / evidence | Migration consequence |
| --- | --- | --- |
| Headless execution | `src/ml_lab/application.py`, `src/ml_lab/flask_adapter/`, `src/ml_lab/ui/__init__.py` | Keep CLI, task API, optional Flask UI, and PAH mount independently operable. |
| First-party UI routes | `src/ml_lab/ui/blueprint.py`, `templates/ml_lab_ui/`, `static/` | Move presentation incrementally, with compatibility routes while new tabs are introduced. |
| CSV/TSV intake | `src/ml_lab/data/intake.py`, `src/ml_lab/data/table.py` | Do not replace file inventory, bounded table reads, or source fingerprinting with UI-only alternatives. |
| Existing profiling | `src/ml_lab/data/profiling.py`, `data/types.py`, `ui/jobs.py` | A2 already has inferred type, missingness, cardinality, descriptive summaries, Pearson/Spearman and bounded MI. **Extend** it with variance, zero-rate surfaced uniformly, semantics overrides, additional graphics; do not duplicate the engine. |
| XML single-file flow | `data/xml_ingest.py`, `xml_selection.py`, `xml_rules.py`, `xml_preview.py`, `xml_materialize.py`, `xml_history.py` | Preserve namespace-aware field selection, repeated branches, guarded parsing, previews, materialization, and lineage. |
| XML collection flow | `data/xml_collection.py`, `xml_similarity.py`, `xml_collection_review.py`, `xml_collection_recipe.py`, `xml_collection_export.py`, `xml_collection_provenance.py` | Currently exposed as multiple independently posted pages. Unify the **user journey** without discarding validated collection operations. |
| Experiment dataset binding | `src/ml_lab/experiments/datasets.py` | Already pins project-local CSV/TSV source identity, columns and provenance; detects stale data. Reuse this as the binding foundation rather than introducing a competing dataset registry. |
| Experiment persistence | `src/ml_lab/experiments/store.py` | `ExperimentStore` keeps project-scoped `analysis/ml_lab/experiments.sqlite3`; retain isolation, experiment revisions, runs, and artifact lineage. |
| Existing representations | `src/ml_lab/representation/api.py`, `pca.py`, `training.py`, `transformer_training.py` | Stable PCA, configurable MLP AE, Transformer AE already exist. New sklearn methods and UI method registry should wrap stable APIs, not replace them. |
| Training / visualization | `src/ml_lab/ui/training_jobs.py`, `experiments/models.py`, `run_control.py`, `comparison.py`, `visualizations.py` | Preserve job lifecycles, cancellation, stored results, and exports. |

## Disconnected state and navigation: observed from source

1. `templates/ml_lab_ui/base.html` renders an independent ML_Lab-branded horizontal header and links for **Overview, Data, Experiments, Compare, Transform, Provenance**. Its styling is layered with `ml_lab.css`, `pah-module-theme.css`, and `ml_lab_pah_compat.css`.
2. The user also observes a **Workspace** link and a higher-level vertical tool pane in their running host. `Workspace` is **not present in this snapshot's `base.html`**; confirm whether PAH or a different installed revision inserts it before deleting or relocating chrome.
3. `create_ui_blueprint()` binds a project root for experiments (`experiment_project_root`, default CWD), but data intake, profiling, XML forms, comparison, and transform pages accept their own sources or paths. There is **no single authoritative active dataset context spanning Data → Feature Profiling → Representation → Comparison → General ML Tools** in this snapshot.
4. The XML collection flow has distinct routes for inventory, review, recipe, export, and provenance. Navigation passes selection via forms and URLs. User-reported redirects and losing collection context need reproduction in ML-UW-03; the audit does not claim they were reproduced in a running UI.
5. `ProfileJobManager` defaults to relative `analysis/data/profile_runs` while experiment records use `experiment_project_root / analysis/ml_lab/experiments.sqlite3`. Before binding them to one dataset, normalize ownership and scope carefully; **do not migrate or silently merge existing stores**.
6. Experiment-scoped dataset selection already exposes a useful stale-file safety boundary. The new instance context must never weaken its file containment, source hash, or provenance checks.

### UI route clusters (snapshot)

| Current routes | Future section |
| --- | --- |
| `/`, `/data`, `/data/source/<token>`, `/data/source/<token>/rows`, `/data/source/<token>/matrix`, `/data/transform`, `/data/provenance` | Data (with context/lineage drawer) |
| `/data/profile/start`, `/data/profile/job/<job_id>`, `/data/profile/<run_id>/<int:profile_index>` | Feature Profiling |
| `/data/xml/<token>` and related record-fields, preview, materialize, extraction endpoints | Data → XML single-file mode |
| `/data/xml/collection` and `/review`, `/recipe`, `/export`, `/provenance` | Data → XML collection mode |
| `/data/compare`, `/experiments/compare`, experiment chart/export endpoints | Comparison (distinguish **dataset comparison** from **experiment comparison**) |
| `/data/experiment/selection`, `/experiments/models`, `/experiments/models/run`, `/experiments/models/optimize`, `/experiments/synthetic` and run status/download endpoints | General ML Tools; Representation owns configured dimensionality-reduction workflows |
| `/task/<capability_id>`, `/api/capabilities` | Compatibility / advanced capability access |

The precise, current route inventory is reproducible with `python tools/ml_uw_00_inventory.py`. Routes must continue to resolve under standalone and PAH-mounted URL prefixes. Retain old endpoints as forwards/adapters until tests and consumers are migrated; do not redirect in circles.

## Agreed target: one dataset per ML_Lab instance

The default active instance has exactly one selected dataset identity at a time, shared by all five primary tabs:

**Data → Feature Profiling → Representation → Comparison → General ML Tools**

The word **instance** means an ML_Lab experimental workspace context, not a Flask global and not necessarily a distinct Python process. At minimum it must bind: project identity; selected dataset identity/hash/revision; feature and target column selections; source/provenance; preprocessing lineage; immutable references to profiles, representations, and runs. Context ownership must be explicit and server-side checked, not inferred from the last open tab or one process-global variable.

Changing datasets must invalidate *active references* to incompatible derived analyses while preserving archived immutable experiments and their lineage. Old/stale files are not auto-trusted. Future multiple simultaneous instances require distinct instance IDs; this sprint uses **one active dataset context** but must not embed a process-global singleton that would preclude it.

### Planned tab ownership

- **Data:** CSV/TSV intake, XML structural/collection extraction, transformations, source previews, lineage, dataset selection.
- **Feature Profiling:** existing A2 profile engine + new variance, zero-rate, inferred discrete/continuous semantics, editable overrides, distribution and type-aware pairwise plots.
- **Representation:** PCA plus other applicable sklearn reduction methods, configurable/preset/pretrained AE configurations, supported latent projections and reconstruction viewers.
- **Comparison:** fair cross-method evaluations, same source revision and preprocessing, faithful reconstruction metrics **only when available**, geometric/neighborhood metrics for embedding-only methods.
- **General ML Tools:** existing Models, EBM, Optimization, Generative tools behind nested options; preserve experimental discovery and legacy run flows.

Remove the duplicate horizontal ML Lab link/version/navigation bar after moving essential identity/status to the *existing host/top header*. Remove the observed “any order” helper text next to Tools where it actually originates; do not remove similarly named text by blind search. Compare standalone and PAH-mounted renderings first to avoid deleting the only navigation in standalone mode.

## Required representation capability safety

The planned method registry must declare independently: `fit`, `transform_new_samples`, `inverse_transform`, `approximate_decode`, `preprocessing_requirements`, `supports_sparse`, and whether out-of-sample embedding is supported.

- **t-SNE:** standard sklearn TSNE embedding does not expose a native transform or inverse. Never show/execute Reverse or imply out-of-sample projection without a separately modeled method.
- **Isomap:** supports transformation of new samples in sklearn, **not** a native inverse. Disable its inverse path and reject requests server-side.
- **PCA / TruncatedSVD / other decompositions:** inverse-like reconstruction is approximate when dimensions are discarded; availability is method-specific.
- **Autoencoder:** decoder is a learned **approximation**, not an exact mathematical inverse. Display reconstruction residuals and distinguish real observations from decoded edits.
- **KernelPCA:** inverse only when a supported inverse is fitted/configured; label approximate.

Capability enforcement must live in the application layer **and** in the UI. Embedding-only models remain eligible for non-reconstruction comparison metrics.

## Migration and acceptance gates

| Increment | Must be true before proceeding |
| --- | --- |
| ML-UW-00 (this patch) | Audit documents present; reproducible stdlib inventory passes; no runtime behavior changed; supplied zip baseline reconciled against actual checkout before applying later patches. |
| ML-UW-01 | One dataset context shared across tabs; dataset switch preserves archive but blocks stale references; no other module's SQLite database accessed; multi-instance-friendly IDs. |
| ML-UW-02 | Five tabs and relocated header; standalone and PAH-mounted views both usable; pre-existing URLs remain safe; no extra “any order” text. |
| ML-UW-03 | One flow for XML single-file/collection modes, preview, selection, materialization, provenance; no loops; collection selection survives navigation; parsing error states visible. |
| ML-UW-04/05 | Existing profile artifacts reused; correct zeros/variance/types, editable semantics and bounded pairwise plots; large tables stay paginated. |
| ML-UW-06/07 | sklearn methods and user-selected AE configuration/checkpoints fit through a capability-aware registry; no fabricated inverses; training lifecycle/cancel preserved. |
| ML-UW-08 | Same dataset/preprocessing/splits for comparisons; explicit omission of inapplicable metrics; synchronized selection tied to source observation ID. |
| ML-UW-09 | Existing API/CLI/headless, PAH mount, XML, experiment, storage-isolation tests and manual browser acceptance pass. |

## Before ML-UW-01

1. Confirm actual source revision on the user's machine (this attached ZIP could lag unshared local work); compare baseline route inventory and working-tree diffs.
2. Find the owner of the higher-level vertical pane and the extra Workspace link in PAH or a newer ML_Lab revision. Do **not** delete host chrome from ML_Lab code.
3. Determine whether existing `ExperimentStore` experiment IDs can be referenced directly from a single active-instance context without schema migration; prefer adapting over duplicating storage.
4. Preserve default standalone invocation, optional dependencies, and the explicit `project_root` boundary. Check that profile/derived outputs never leak across projects.
5. Confirm existing XML C4/C5/C6 call sequences and reproduce reported redirect/preview errors on the target checkout before rewriting controllers.
