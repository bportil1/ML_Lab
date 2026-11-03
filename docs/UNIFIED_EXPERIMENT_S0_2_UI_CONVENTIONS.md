# Unified Experiment Workflow — S0.2 UI and Interaction Conventions

**Status:** Design contract only (no UI, route, schema, or algorithm changes in this sprint).
**Predecessor:** [S0.1 experiment architecture](UNIFIED_EXPERIMENT_S0_1_CONTRACT.md).
**Companion:** [S0.2 scientific matrix specification](UNIFIED_EXPERIMENT_S0_2_MATRIX_EDITOR.md).
**Target:** The ML_Lab standalone interface, remaining compatible with optional PAH hosting.

## 1. Intent: shared application identity, independent modules

- ML_Lab, HSQA_DBN, and pyPIQUE should feel like members of one application family,
  without introducing a dependency on another module's frontend, database, or code.
- The PAH *application shell* is the reference for typography, restrained spacing,
  navigation, action hierarchy, focus states, feedback, and module-header styling.
- pyPIQUE and HSQA_DBN's collapsible workflow rails are **interaction references**;
  each module retains the stages appropriate to its algorithms.
- Matrix and spreadsheet areas are **generic scientific instruments**, not PAH-
  branded widgets. They follow the minimal matrix contract in the companion file.
- A module must render and function when opened standalone, with no PAH assets,
  endpoints, globals, or CSS reachable. The host can supply project context and
  mediate job/artifact links through documented interfaces, never internal DB reads.
- The shared style is a specification, not a new shared-runtime theme package.
  Each independent module may ship compatible local assets and theme variables.

## 2. Inspected local UI precedents (not promises of future functionality)

| Source | Relevant inspected files | Existing pattern to reference |
| --- | --- | --- |
| PAH | `pah/web/static/pah-identity.css`, `pah-labs.css`, `pah-workspace.css` | Branded application chrome, restrained technical controls, semantic status colors |
| pyPIQUE | `src/pique_py/static/visualizer.css`, `visualizer.js`, `templates/visualizer.html` | Collapsible workflow rail, stage LEDs, compact tables, interactive sort/filter/search/copy/export |
| HSQA_DBN | `hsqa_dbn/ui/assets/pah-module.css`, `hsqa-analysis-workspace.js` | PAH-compatible module theme, collapsible analysis/workflow navigation |
| ML_Lab | `src/ml_lab/ui/templates/ml_lab_ui/base.html`, `data.html`, `transform.html`; `ui/static/ml_lab.css`, `pah-module-theme.css`, `ml_lab_pah_compat.css` | Existing standalone module shell, dataset workflows and locally bundled PAH-compatible styling |
| ML_Lab tables | `ui/static/ml_lab_tables.js`, `ml_lab_transform_sheet.js`, `ml_lab_source_table.js` | Table search, filtering, sorting, pagination and transformation-column actions; **not** a complete editable matrix |

Paths in the first three rows are inside their respective **reference
repositories** supplied during the planning session, not dependencies to import
from ML_Lab. Their implementation details may change independently.

## 3. ML_Lab workspace information architecture

Top-level workspace: **Experiments** (experiment selection, recorded runs and
stage navigation), with **Data** and **Results** reachable through that
workspace. Preserve existing Data Lab, Compare, Transform, Provenance and task
URLs during the transition; new navigation must not orphan bookmarks or
existing forms. Avoid creating a static panel for every subsystem.

| Stage | Primary user task | Readiness depends on |
| --- | --- | --- |
| Data | Choose/import/inspect/transform a revisioned dataset | Resolvable input, compatible schema and provenance |
| Model | Select model family, typed parameters and target/feature mapping | Valid dataset/model combination, or a model-only operation |
| Training | Configure and execute the selected model where required | Resolved configuration and engine capabilities |
| Optimization (optional) | Configure a compatible search/evaluator and trials | An explicitly supported optimizer/evaluator/metric combination |
| Evaluation | Inspect family-specific outputs and compare comparable runs | Relevant completed artifacts or an inspectable failed run |
| Results | Review run history, references, exports and derived datasets | Independently discoverable artifacts / provenance |

Stages are navigation and workflow *readiness*, not a forced wizard. Model
families with no fitting step may skip Training; optimization is an optional
branch; a generative model's **sampling** may produce a new Data artifact for
another experiment. No stage is advertised as complete solely because the
corresponding tab was visited.

## 4. Navigation and layout contract

1. Use one collapsible workflow rail with compact, consistently ordered stages,
   a clearly identified active stage and optional status dots/squares. The rail
   can collapse to a narrow stage indicator; it must not become a permanent
   static side-pane that needlessly reduces the scientific workspace.
2. On narrower layouts, place stage navigation above content or into a usable
   collapsible control. Do not trap horizontal scrolling of scientific grids in
   the workflow rail; the scientific area owns its overflow.
3. Provide a recognizable **next meaningful action**, but never automatically
   execute or delete work merely because a stage changed. Support backtracking.
4. Keep experiment and dataset identity visible as concise context near the
   workspace header. Distinguish an unsaved draft from a persisted revision.
5. Keep destructive actions visibly separate from ordinary navigation,
   explicitly scoped to the affected run/dataset, and confirmation-gated.
6. Show both standalone and PAH-hosted modes with the same functional workflow;
   host mode may add links and status, not replace science controls.
7. Avoid decorating every matrix cell or table toolbar with branded PAH colors.
   Semantic highlighting is permitted when it communicates a documented value.

## 5. Two orthogonal status dimensions

Maintain the distinction introduced in S0.1 between **stage readiness** and
**job execution**. One display may show both, but never collapse them into an
uninterpretable generic 'done' badge.

| Status dimension | Values | User-facing intent |
| --- | --- | --- |
| Workflow stage | `unconfigured`, `blocked`, `ready`, `complete`, `stale` | Whether the current selection can proceed or reuse an existing output |
| Execution | `queued`, `running`, `succeeded`, `failed`, `cancelled` | What happened to an actual run, where supported |

A completed historical run remains successful even when its inputs are no
longer the active selection; the **current stage** is then `stale` rather than
silently reusing that result. Failure remains inspectable. Display text and
icons as well as status color (do not rely on color alone). Use animations only
for actual work in progress; do not imply progress data that the engine lacks.

## 6. Standard scientific control behavior

- **Forms:** Explicit field labels, typed controls and model-specific validation
  with actionable messages; preserve an advanced/raw configuration path for
  specialized parameters until parity is demonstrated.
- **Action placement:** Primary action close to the relevant form; secondary
  actions (reset, export, inspect provenance) visually subordinate. An action
  should name its scope (e.g., 'Run experiment' not 'Apply').
- **Job controls:** Show queued/running/final states and available progress,
  cancel and resume only when the selected engine implements those behaviors.
- **Tables and matrices:** Follow the [matrix contract](UNIFIED_EXPERIMENT_S0_2_MATRIX_EDITOR.md).
  Dense numerical data should use conventional grid navigation and formatting,
  not card-by-card rendering.
- **Plots/graphs:** Use legible axes, units, legends and data exports where
  available; never invent generic 'quality scores' across dissimilar models.
- **Provenance:** Accessible through lightweight context/details, never a
  mandatory intrusive overlay. Dataset/model/run IDs identify exactly which
  version produced a visible result.
- **Accessibility:** Keyboard navigation and visible focus, meaningful
  labels/status text, and proper focus restoration after dialogs and actions.

## 7. Theme ownership and isolation

- The local ML_Lab shell may adapt its already bundled
  `pah-module-theme.css` and `ml_lab_pah_compat.css` in implementation sprints.
- Avoid global `table`, `td`, `input`, `button` rules that unexpectedly restyle
  a scientific matrix. Scope scientific-grid rules to a dedicated root class;
  keep app chrome and matrix visual tokens logically separate.
- A matrix should remain readable with theme loading disabled or when viewed
  standalone. Do not couple mathematical value formatting to CSS or colors.
- Style equivalence means comparable information hierarchy and interactions,
  **not** copying source or defining a cross-repository runtime dependency.

## 8. Implementation allocation and test gates

| Sprint | Work enabled by this specification |
| --- | --- |
| 1 | Persist experiment/run/artifact metadata, stage readiness and revision references; no forced UI migration |
| 2 | Dataset selection and matrix implementation using the companion contract |
| 3 | Typed model-family configuration and unified stage navigation |
| 4 | Job status, optimizer trials and conditional progress controls |
| 5 | Consistent diagnostics, comparison tables, figures and exports |
| 6 | Generative-model sampling and synthetic dataset handoff |
| 7 | Standalone / optional PAH end-to-end parity, accessibility and regression checks |

**S0.2 acceptance (documentation patch):** The UI reference sources, layout,
stage labels, readiness/execution mapping, theme boundaries and matrix
interaction requirements are explicitly documented and linked. No runtime
features are claimed to have been implemented.

**Later implementation acceptance:** A user can select a dataset, configure a
valid model and inspect run results without manually passing JSON between
independent tabs; the same process works standalone and when launched through
PAH. Existing routes continue to operate until replacement parity is proven.
