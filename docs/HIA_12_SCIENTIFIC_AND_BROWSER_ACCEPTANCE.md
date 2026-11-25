# HIA-12 — Whole-sprint scientific and browser acceptance

## Scope, ownership and execution

**HIA-12 is the final acceptance stage of the HIA-01–12 sprint.** This patch
adds reproducible checks and corrects two outdated structural-inspector labels.
It does **not** change any empirical estimator, reassign any source node, change
how the PAH_Shared graph engine renders, introduce hidden sampling, or write to
a user's scientific dataset, SQLite database or PAH embedded modules directory.

Run all source and test operations from the **standalone `ML_Lab` checkout**.
The browser acceptance uses the **standalone sibling `PAH_Shared` checkout**.
No public services, network requests, package downloads, or outside repositories
are needed. Source files and evidence never leave the local computer.

## Local acceptance command

```sh
cd ~/cluster/ML_Lab
python tools/hia12_acceptance.py
```

The runner executes:

1. All `test_hia*.py` and `test_su*.py` Python tests, plus graph-cache-revision tests.
2. The HIA-11 JavaScript overlay assertions through **Node's built-in test runner**.
3. Syntax checks for the HIA JavaScript controllers and scientific overlay module.

For the optional **local Chromium** acceptance, with Playwright and Chromium
already installed on this machine:

```sh
python tools/hia12_acceptance.py --browser \
  --shared-viewer ../PAH_Shared/src/pah_shared/static/interactive_graph_view.mjs
```

Optionally specify `--chromium /path/to/chromium`. This smoke fixture imports
**the actual local PAH_Shared renderer** into an offline browser page, mounts a
small original-source graph, and exercises HIA-11's real overlay module. It
checks selection/camera continuity, mapped-vs-unmapped colors, matched-cohort
level comparison, CSV/SVG/PNG downloads, invalidation, and source reset. It
intercepts/blocklists HTTP(S) requests. It does not claim to test the real
PAH-hosted page, installed ML_Lab process, or your personal source data.

A missing optional Flask or Playwright dependency is **not auto-installed**.
The Python suite skips Flask-dependent HTTP tests if Flask is absent. A
requested browser test fails with a clear local-dependency/path error instead
of silently reporting browser success.

## Scientific cross-stage evidence

| Stage | Acceptance invariant | Evidence |
|---|---|---|
| HIA-01/02/03 | Same exact source IDs, stable hierarchy/levels, no inferred dependency | Source inventory and scope selection tests |
| HIA-04 | Atomic H(X) is observed, bound, and traceable to the source digest | Finite balanced binary fixture |
| HIA-05/06 | Structural neighbors can be empirically independent; branch entropy/TC reflect observed variables, not edges | XOR fixture with independent connected node pairs |
| HIA-07 | Level H(L) and TC use explicit mapped columns and one cohort | Independent and redundant level fixtures |
| HIA-08 | H(L\|Z) and I(Y;L\|Z) use joint previous-level variables; target support reported separately | XOR and incomplete-target fixtures |
| HIA-09 | Explicit factorization can diverge from empirical joint; zero KL is descriptive, not proof of independence | Independent vs dependent X/W cases |
| HIA-10 | Sum of conditional level entropies equals joint entropy; target increments telescope over matched cohort | Complete and missing-target fixtures |
| HIA-11 | Colors apply only to measured original source IDs; unmeasured nodes neutral, measured zero distinct | Node overlay module and actual shared-renderer smoke |
| HIA-12 large structure | More than 5,000 source nodes and 15,000 edges remain exhaustive, IDs preserved | DAG with 5,101 nodes and 15,297 edges |

All scientific report JSON objects used in the cross-stage tests round-trip
with `allow_nan=False`, and no analysis mutates the original StructuredObject.
These fixture results do not establish the precision of entropy estimates on
arbitrary sparse/high-dimensional data. The plug-in estimates remain
finite-sample descriptive quantities; no source edge is causal evidence.

## Hosted deployment acceptance — still required on the user's machine

The following steps depend on the *actual* PAH / ML_Lab installation and
cannot be honestly declared complete from an offline fixture:

- [ ] Refresh the standalone `ML_Lab` and `PAH_Shared` installations using the
  existing PAH setup/update process, and restart existing ML_Lab instances.
- [ ] In **ML_Lab → Data → Structured Uncertainty**, open a verified structured
  file with aligned observations; use **Explore source graph**. Confirm the
  existing PAH_Shared camera/selection/pagination, fullscreen and detached
  window still work.
- [ ] Inspect an atomic node, direct neighbor(s), tree branch, level, and whole
  network. Provide **explicit node-to-column mappings** for each estimate;
  confirm the source digest and observation support in exported report JSON.
- [ ] Check HIA-07/08/09 on a third-level selection: H(L), H(L\|prior), and
  optional factorization must be distinct and correctly labeled. Confirm a
  large prior conditioning vector reports insufficient support instead of
  inventing an estimate.
- [ ] In HIA-10, verify cumulative target information and incremental level
  information are on the same complete-case cohort. Cross-level pairwise MI
  must remain a separate (non-additive) comparison.
- [ ] Use HIA-11 overlays: select entropy/support/level aggregate, inspect
  legends and contrast, navigate to another graph page, compare levels,
  export CSV + visible-page SVG/PNG, then edit a mapping and check that the
  stale colors and exports are invalidated.
- [ ] Change the selected source. Confirm pending requests cannot repaint the
  new graph and old source-revision reports cannot reappear.
- [ ] Inspect large-graph overview and a late original node ID; verify no
  5,000/15,000 size error, no silent truncation, and full-source JSON provenance.
- [ ] Confirm logs show no writes to scientific source artifacts/databases,
  and no changes in `Project_Assistant_Host/modules/`.

Record failed checkpoints with the relevant component version, URL, console
error and *non-sensitive* test fixture if possible. Do not transmit private
project data to outside services.

## Acceptance status semantics

**Locally verified**: cross-stage Python mathematics; large source graph
presentation contract; JavaScript overlay/exports; offline Chromium mounting
the real PAH_Shared engine when explicitly requested.

**Environment-dependent**: Flask route tests, where Flask is absent locally.

**Not yet verified**: end-to-end behavior of this patch in the user's own
PAH-hosted browser with their installed runtime and scientific project data.
The sprint's implementation can be considered delivered while those
machine-specific acceptance items remain open; do not mark them completed
without observing the relevant results.
