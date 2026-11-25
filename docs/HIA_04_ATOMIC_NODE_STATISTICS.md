# HIA-04 — Atomic node statistics (read-only, opt-in)

## Ownership and scope

This increment extends **ML_Lab**, not PAH's embedded `modules/` directory. It
builds on HIA-01 scope contracts, HIA-02 deterministic hierarchy indexing,
HIA-03's original-identity graph navigator and the existing SU-05 observation
binding API. PAH_Shared continues to own graph rendering, pan, zoom, paging,
and selection; HIA-03's fullscreen, detached viewer, and exports remain intact.

An original source node is a structural entity, **not** an independent sample.
An edge is a source relationship, **not** evidence of MI, CMI, causality,
conditional independence or probabilistic dependence. The user must first
select the source node, then choose an original observed column that represents
it. The mapping is declared in a single-node `ObservationBindingPlan` and
validated by SU-05. HIA-04 does not guess a mapping by identical labels or by
source adjacency, and does not store it or mutate any source data.

## Output schema

`ml-lab.hia-atomic-statistics@1` has immutable source and binding SHA-256
fingerprints, original `node_id` and declared `variable_id`, and the
HIA-02 node context. Reports include:

- Aligned observation-row count, scalar observed count, missing/unavailable
  count, status counts **and rates**, missing-source reasons, and original
  repeat/occurrence evidence before any optional SU-05 resolution.
- Scalar value frequency distribution; typed values (`1`, `true`, `'1'`)
  are distinct. For unresolved repeated cells, this distribution covers only
  scalar rows, explicitly marked partial. No repeated values are silently
  allocated to independent observations.
- Numeric count, mean, range and **sample variance** from the scalar numeric
  observations, when the observed scalar values are numeric and finite. It is
  unavailable for mixed categorical/numeric variables. Where repeated rows
  remain unresolved, the numeric subset is partial.
- Empirical discrete plug-in entropy in **bits**, state count and sample
  support. Integer/string/boolean states are treated categorically; floating
  values need user-declared `quantile` or `equal_width` bins, 2–128. Binning
  uses the existing SU-07 information-theory primitives. Missing/unknown/null
  rows are excluded under an explicit complete-case policy; empty strings are
  represented as their own observed state. This estimate is descriptive and
  generally biased at finite sample sizes.
- An explicit unavailable status for missing rows, insufficient support,
  unresolved repeated cells, non-scalar cells, floats without explicit bins
  and incompatible bin definitions. **No placeholder zero entropy or
  fabricated information** is returned.

The `mean_numeric` option performs an explicitly selected SU-05 repeated-cell
aggregation only if every occurrence is present, finite and numeric. The
original repeated-row/occurrence counts survive in the report's lineage.

## Interface and validation

On the existing **Data → Structured Uncertainty** page, select a project-local
source, click **Explore source graph**, then select an original node. The new
**Atomic node statistics** panel enumerates available aligned observation
columns. Selecting a column is always explicit. The user chooses the repeat
handling and optional discretization, and presses **Compute selected node
statistics**. No estimate is calculated by hovering, scrolling or selecting a
node. A compact summary and complete JSON report are then available, with a
read-only JSON export action.

The new `/workbench/structured/hierarchy/atomic` POST requires a same-origin
explicit-analysis header, strict JSON keys and a SHA-pinned source selection.
Every request rereads and validates the selected project-local source. A stale
revision or changed source is rejected, without writing to SQLite, triggering
jobs, or training a model. The response also includes the original source file
revision so the browser rejects stale asynchronous results.

## Boundaries and next steps

HIA-04 is **atomic** only. It does not infer node-to-node relationships or
estimate MI, CMI, correlations, joint entropies or structural-level aggregates.
HIA-05 introduces explicit neighbor associations, HIA-06 branches, and
HIA-07–09 empirical and optional factorized level information. The graph
viewer itself and the source hierarchy are unchanged.

Run regression tests locally:

```bash
PYTHONPATH=src python -m pytest -q tests/test_hia04_atomic_node_statistics.py
```
