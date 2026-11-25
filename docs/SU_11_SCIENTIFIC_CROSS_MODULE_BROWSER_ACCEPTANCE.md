# SU-11 — Scientific, Cross-Module, and Browser Acceptance

**Status:** acceptance harness and offline fixture checks implemented. **Not a claim of full host acceptance.** This increment adds tests, an opt-in local artifact verifier, and an explicit deployment checklist; it does not change existing scientific estimators, shared graph rendering, Flask production routes, or source data.

## What is actually verified, and what is not

The new `tools/su11_acceptance.py` accepts **explicitly supplied** project-relative JSON exports or actual `.xml` files and reports source type, canonical source digest, raw file SHA-256, source topology, observed **record** count, adapter losses/warnings, exhaustive-view eligibility, and per-gate pass/fail/skip. No source data is uploaded, no repository or module SQLite is opened, no web requests are made, and nothing is written unless `--output` is explicitly supplied. Supplying `--viewer` invokes the **actual locally installed** `pah_shared/static/interactive_graph_view.mjs` through Node; it does not create a substitute validator.

`status: passed` means **all checks that were attempted for the supplied artifact passed**, not that omitted environment gates were completed. Consult each gate's `status` and `reason`. A missing source, invalid import, stale binding or failed real-viewer check yields a nonzero exit status. A graph that exceeds UI/display limits stays scientifically valid, but its viewer gate is marked **skipped/explicit_viewer_limit**, never displayed as sampled exhaustive evidence.

A raw XML input is processed via the existing safe `inspect_xml_structure()` → `XMLStructureAdapter` pipeline. It produces *canonical aggregate profiles*, not aligned per-record observations: occurrence counts are **not** interpreted as sample rows, and numeric uncertainty/MI is unavailable until actual record-level data are separately and explicitly aligned and bound. For JSON nested instances, AST handoffs, Code Analyzer exports, pyPIQUE concluded graph exports, graph@1 displays and network artifacts, the source adapter never invents observations.

## Local headless commands

From the **ML_Lab repository root**:

```sh
PYTHONPATH=src pytest -q tests/test_su11_acceptance.py tests/test_su11_live_browser.py \
  tests/test_su11_shared_renderer_browser.py \
  tests/test_su10_structural_sandbox.py tests/test_su09_structural_retention.py \
  tests/test_su08_structural_candidates.py tests/test_su07_branch_information.py \
  tests/test_su06_branch_uncertainty.py tests/test_su05_structural_observation_binding.py \
  tests/test_su04_shared_graph_integration.py
```

For actual project-local exports **you provide**:

```sh
PYTHONPATH=src python tools/su11_acceptance.py \
  --project-root /path/to/your/local/project \
  --artifact xml_raw:relative/path/to/multiple_records.xml \
  --artifact json:relative/path/to/nested_data.json \
  --artifact hierarchy_handoff:relative/path/to/ast_export.json \
  --artifact pypique_graph_directed:relative/path/to/concluded_graph.json \
  --output /path/to/local/su11_acceptance_report.json
```

Do not assume those files exist. A pyPIQUE/Code Analyzer handoff is a real-export gate **only after** the producer actually exports a supported artifact and it is supplied to the runner; generated synthetic fixtures do not count as live integration.

To verify the actual locally installed shared graph module:

```sh
PYTHONPATH=src python tools/su11_acceptance.py \
  --project-root tests/fixtures/structured_uncertainty \
  --artifact structured:xml_tree.json \
  --viewer /path/to/local/PAH_Shared/src/pah_shared/static/interactive_graph_view.mjs
```

For an observed StructuredObject, add **one** explicit versioned `--binding path/to/plan.json --node EXACT_SOURCE_NODE_ID`. SU-05's full canonical `source_digest` must match; rows and mapped source IDs are validated. Binding is deliberately not inferred from labels or XML path similarity.

## Optional *real* hosted/browser acceptance

The test `tests/test_su11_live_browser.py` requires (1) Flask and its dependencies installed in the active venv, (2) PAH_Shared SU-02's JS/CSS actually installed and registered as local Blueprint assets, (3) Playwright Python installed, and (4) a locally installed Chromium executable. The test only binds `127.0.0.1` and refuses nonlocal browser requests. Execute explicitly:

```sh
SU11_BROWSER=1 PYTHONPATH=src pytest -q tests/test_su11_live_browser.py -rs
```

It uses the **real** ML_Lab Blueprint, **real** PAH_Shared module asset, and browser actions (source selection, opt-in viewer, linked node, evidence action, transformation preview, JSON export and five-tab navigation). Source bytes are checked unchanged afterward. A skipped test is a **pending gate**, never browser acceptance. A synthetic test server is not the full PAH host; the integrated PAH instance must also be opened manually and checked in the target environment.

### Optional shared-engine Chromium component smoke

This is distinct from the real Flask-hosted application test. It imports the
**actual local PAH_Shared** `interactive_graph_view.mjs` in a browser page served
from `127.0.0.1`, checks SVG mounting, selection callbacks, zoom, fit and
context/overview, and rejects external browser requests. Set
`SU11_SHARED_VIEWER` to the locally installed module path and run:

```sh
SU11_BROWSER_COMPONENT=1 SU11_SHARED_VIEWER=/path/to/local/pah_shared/static/interactive_graph_view.mjs \
  PYTHONPATH=src pytest -q tests/test_su11_shared_renderer_browser.py -rs
```

If the machine's Chromium policy prohibits localhost navigation, the test
**skips with the explicit policy blocker** rather than claiming browser success.
The no-browser Node validator remains separately available through `--viewer`.

### Target-environment manual gates

| Gate | Proof required | Status if unavailable |
| --- | --- | --- |
| Real multi-record XML | Existing XML extraction, actual record alignment, grouping/null reason policy and SU-05 binding; no occurrence-as-row conversion | Pending; structural XML profile only |
| Nested JSON, AST and cyclic graph | Verified real local exports, original IDs/paths, source file integrity, exhaustive vs capped view | Pending without actual exports |
| pyPIQUE model | Existing concluded and learned backing viewer still works; signed associations and underlying probabilistic links are not merged or made causal | Pending without installed pyPIQUE and an original model archive |
| Code Analyzer AST | Real parser export with source spans and stable node IDs; independent module DB remains untouched | Pending without real ATLAS/AST export |
| Shared viewer | Both modules import same SU-02 JS implementation, not a copied renderer; pan/zoom/selection/overlay/export behavior | Node contract can pass headlessly; interactive behavior pending browser check |
| ML_Lab navigation | All 5 sections, Data subpanels, current dataset readiness, existing scientific operations | Headless static checks; hosted workflow pending |
| Session/project isolation | Separate mount roots and cookies, no cross-project reads, path traversal/symlink refusal, stale raw SHA rejection | Headless checks and optional hosted test |
| Saved artifact/cache | Reopen selected artifact, repeat cached computations only by explicit action; no stale SU-10 results after revision | Pending actual project/runtime |
| Large/deep/heterogeneous | Explicit 5,000-node / 15,000-edge limit; deep-tree tiers, DAG/cyclic edge treatment, finite-sample computational bounds | Synthetic limit tests; real performance pending |
| Exports and hygiene | JSON contents and digests, no source write-back, `git status --short`, generated outputs ignored but intentional assets retained | Pending live-project confirmation |

## Scientific estimands and interpretation limits

- **SU-06**: declared observed-variable occurrence, missingness, cardinality, numeric variance and nested-group descriptive decomposition; one aligned observation row remains one sample irrespective of repeated occurrences. Group and weight semantics are explicit, and degeneracy/unsupported operations remain unavailable.
- **SU-07**: empirical discrete plug-in entropy, MI and CMI with explicitly selected target, conditioning set, missingness rules and numeric bins. Sparse support and estimator bias diagnostics must be shown; MI/CMI is not a causal effect or independent guarantee of feature utility.
- **SU-08**: verified source-linked structural candidate transformations (preserve/prune/summarize/depth reduction/aggregate/flatten), with observation deletion separate and explicitly opt-in. Original graph is unchanged.
- **SU-09**: same-record holdout predictive tasks, retained variance/entropy, reconstruction proxy and complexity. Reference/candidate empirical scores remain identical when only topology changes; group leakage and train-only estimation controls remain in force.
- **SU-10**: user-triggered inspection and candidate preview; only **node** scientific overlays, no inferred source-edge statistics. Original source and all module-owned DBs remain untouched.

These are local observational estimands and bounded exploratory comparisons, **not** a validated causal pruning method or a claim that source connectivity itself carries probabilities. Any scientific calculation over source graph topology without a verified aligned-observation model must be labeled unavailable.

## Reporting convention

Record for each deployment: input/export SHA-256, exact installed revisions for ML_Lab/PAH_Shared/pyPIQUE/Code Analyzer, test command, passed/failed/skipped/timeout counts, browser and Flask versions, source mutation check, and remaining blockers. **Never** replace an absent test, timed-out test, synthetic fixture, or skipped browser with "passed." The full PAH-hosted application and other teams' real exports cannot be accepted from this archive alone.
