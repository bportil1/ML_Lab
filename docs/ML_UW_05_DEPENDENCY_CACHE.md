# ML-UW-05 — Dependency matrix and bounded project-local profile cache

This is an additive ML-UW-04 follow-on for **Feature Profiling**. It preserves
existing A2 `profile_file` calculations and never changes model or training
input semantics. It does not introduce additional project databases, new
services, external communication, UI plotting libraries, or destructive
migrations.

## Implemented

- A symmetric dependence matrix built from **existing** Pearson, Spearman,
  discretized mutual information (MI), and normalized MI (NMI) outputs. Each
  metric/8–12–20-column selection only re-renders stored relationship results;
  it does not re-fit or re-estimate a statistical model. A matrix cell links to
  the existing type-aware pair explorer. Matrix diagonals and unavailable
  estimates remain **not estimated**, never silently set to 0 or 1.
- The source-inferred A2 eligibility cap remains 20 columns. The interactive
  report requests **all** valid pairs among those candidates instead of the
  former 150-pair display cap. Large datasets still profile at most 5,000
  deterministic reservoir-sampled rows.
- A bounded JSON **preview** cache in the existing per-project database:
  `analysis/ml_lab/experiments.sqlite3`. Cache keys include the active verified
  dataset identity, generation/revision, selected features, and visualization
  overrides. At most 24 previews per workbench instance are retained; each
  entry has a 2 MB serialized limit. No raw observations or dataset rows are
  persisted in cache. Role edits invalidate the workbench's cached previews.
- The caller revalidates the pinned dataset's bytes and provenance via
  `get_workbench_instance` before any cache read. Cache writes compare the
  current instance generation and pinned experiment revision in a database
  transaction. `refresh=1` (the **Recompute profile** button) bypasses the
  cached report, recomputes, and stores the new preview.
- Optional `workbench_feature_profile_cache` table leaves `PRAGMA user_version`
  unchanged at **3**, preserving compatibility with older v3 binaries if this
  patch is reversed. The table is re-created on initialization if absent.

## Non-goals / limitations

- Cache accelerates repeated *identical feature previews* and matrix metric
  switches. Changing the left/right or selected feature computes and caches
  that separate bounded preview; it does **not** cache all input CSV samples,
  avoid the verified dataset source checksum, or serve as a durable analysis run.
- The A2 `max_relationship_columns=20` sampling policy is still in force;
  the matrix cannot claim associations for additional unexamined columns.
- Matrix shading is relative to the maximum displayed magnitude, not a
  hypothesis test, causal claim, probability, or significance interval.
- This patch does not implement server-side pagination or new statistical
  estimators. Larger-scale matrix computation and PCA/autoencoder
  representation work remain separate increments.

## Manual acceptance

1. Apply on top of ML-UW-04, open Data and select a verified CSV/TSV dataset.
2. Enter Feature Profiling; click Inspect active dataset. Confirm profile
   statistics and matrix are computed once. Switch Pearson/MI/NMI and the
   8/12/20 column limit; confirm a cached response and unchanged A2 numbers.
3. Click a populated matrix cell; confirm the pair selector opens the exact
   left/right feature columns. Inspect an unset cell, which must show an em dash.
4. Edit a display role and confirm profile previews are invalidated without
   changing source files, experiment revisions, training inputs, or other
   workbench instances.
5. Modify the dataset on disk; verify the next request rejects the stale pinned
   selection rather than reading cache, then explicitly reselect the dataset.
