# SU-10 — Interactive evidence overlays and pruning sandbox

**Status:** additive ML_Lab implementation; local Flask/contract tests complete. Real PAH_Shared SU-02/browser and hosted-PAH acceptance are intentionally deferred to SU-11.

## Ownership and entry point

Open **Data → Structured data** and explicitly select a previously exported, project-local JSON artifact. Click **Explore source graph** (existing SU-04 opt-in view) and select a source **node**. A new **Evidence and candidate sandbox** pane appears below the original shared viewer. Only the installed **PAH_Shared** `interactive_graph_view.mjs` draws the original and candidate structures. There is no forked graph renderer, CDN or cross-module SQLite query. The original five-section workbench and nested navigation are unchanged; the existing Feature Profiling and Comparison tools remain independent.

The browser can manually map a selected observed column to the selected **original source node** with **Bind this column**, accumulating a versioned SU-05 `ObservationBindingPlan`. It may also accept a complete plan pasted by the user, including multi-node `merged_sources` and repeated-value policies. A graph label or structural edge is **never** used to guess a statistical join. Group variables are declared separately; sample independence is not inferred.

The original graph is not an observation population. Display node/edge counts never become sampling counts, marginal probabilities, MI, variance contributions or causal claims.

## Explicit actions

**Inspect selected branch evidence** calls `POST /workbench/structured/sandbox` with `mode="inspect"` only after a deliberate click. It returns:

- selected node's verified SU-05 variable scope (subtree for rooted tree/forest, direct-only otherwise), aligned record count, original cell statuses, null/source-reason counts, and at most 12 original observation examples, with a truncation flag;
- optional SU-06 numeric targets, declared group-level variance and support/exclusions;
- optional SU-07 discrete entropy, MI/CMI, retained-variable redundancy, explicit numeric bins, estimator and sparsity warnings;
- explicit `unavailable` fields when no binding, no requested metric or no mapped scope exists.

**Preview candidate (no writes)** computes a deterministic SU-08 preserve / prune / summarize / aggregate / depth-reduce / flatten transformation. The returned manifest separates removed/collapsed structural identities from observed-column deletion. Source-aligned rows are never rewritten. `drop_exclusive` requires an actual validated SU-05 binding: structure-only `preserve` is the default. The candidate draws in an **additional instance of the same PAH_Shared renderer**, alongside the still-intact source viewer. A linked list of removed IDs and retained selection allows focusing the corresponding original source records. The preview is not an edit, write-back or trained model.

Optionally request a matched **SU-09** preserve-versus-candidate evaluation with explicit predictors, task kind, target, numeric variance/reconstruction columns, and either a grouping variable or a deliberate independent-row acknowledgment. The comparison returns original and candidate metrics, holdout split, complexity, retention/projection and estimator warnings. If feature columns are unchanged, empirical scores must be unchanged. No hidden training occurs on page navigation or node selection.

**Export explicit sandbox report (JSON)** makes a client-side downloadable copy of the latest returned result; the server does not create artifacts or write database rows.

## Contract and trust boundary

- UI request schema `ml-lab.structural-sandbox@1`, strictly bounded JSON body (96 KiB), only `POST` with an explicit `X-ML-Lab-Sandbox: explicit-analysis` header. No GET execution.
- The server reopens the selected **project-relative, nonsymlink, ≤2 MiB JSON** through existing `read_source` and requires its **original raw file SHA-256** from the httpOnly same-site selection cookie. It also requires the browser's exact `source_revision` and checks SU-05's **canonical full StructuredObject SHA-256** separately.
- Original source IDs stay unchanged. Candidate projection revalidates the shared viewer's ≤5,000-node / ≤15,000-edge and identity-length limits; no silent display sampling.
- SU-06/07/09 perform scientific calculations only from explicit **aligned rows**. The SU-10 interactive route refuses more than 10,000 rows; larger studies use existing headless APIs. No sampling assumption is manufactured.
- Local JS only uses DOM `textContent` for untrusted record values, never `innerHTML`; results are capped for on-screen text, with full JSON available only by user-triggered export. It never sends source code to any external service.
- An explicitly computed SU-06/SU-07 result produces a detached `pah-shared.graph-view@1` **node overlay**, installed with PAH_Shared's `viewer.update(..., {preserveSelection:true})` and displayed alongside the linked evidence pane. Measured entropy/MI/CMI and optionally one scoped variable's descriptive variance carry real support and units; unavailable metrics use `value:null`, never a fabricated zero. **No structural edge overlay is created.** A scoped observed-variable variance is not variance *attributed* to a source node or edge, and source links have no causal interpretation.
- Repeated observations are still **one sample each**, with the SU-05 policy determining whether repeated cell values can become scalar evidence. Ambiguous merged-source columns are identified explicitly, not allocated to individual nodes.
- Source selection changes invalidate the active scientific report; an existing preview can remain visibly pinned during inspector refocusing (with a stale-to-selection notice), until rerun or source closure.

## Validation

From repository root:

```sh
PYTHONPATH=src pytest -q tests/test_su10_structural_sandbox.py tests/test_su04_shared_graph_integration.py tests/test_su05_structural_observation_binding.py tests/test_su06_branch_uncertainty.py tests/test_su07_branch_information.py tests/test_su08_structural_candidates.py tests/test_su09_structural_retention.py
node --check src/ml_lab/ui/static/ml_lab_structured_graph.js
node --check src/ml_lab/ui/static/ml_lab_structured_sandbox.js
```

SU-11 remains responsible for browser-level graph interactions, actual PAH_Shared installation/asset serving, larger dataset and PAH host acceptance, cross-module fixtures, and any design tuning required after real-world use. Patch delivery does **not** mean installation in the user's live checkout.
