# ML-UW-13 — Bounded profiling cache and large-category safety (internal)

## Scope and baseline

This patch applies after ML-UW-12. It improves **Feature Profiling** without
changing the A2 profiling engine, the scientific metric formulas, representation
training, experiment history, dataset selection, or XML extraction.

## Observed bottlenecks

1. ML-UW-05 cached feature reports by the selected column and feature pair.
   Choosing a *new* distribution or pair triggered `profile_file()` again,
   including its expensive relationship computation. The scientific profile
   does not depend on these display selections.
2. Discrete/discrete pair previews computed a full dense `pandas.crosstab`
   for potentially thousands of unique labels, then displayed only the
   eight most frequent categories on each axis.
3. The existing per-instance 24-record cache could evict an expensive reusable
   baseline after many new feature/pair selections.

## Implementation

- An additional, distinct SQLite cache key identifies the **scientific
  baseline** by project-local workbench instance, generation, experiment
  revision, dataset identity/checksum, and feature-role overrides. The existing
  per-selection report key remains unchanged for backward compatibility.
- The first cache miss computes the existing full A2 profile and saves the
  bounded JSON report under both the selection key and the baseline key.
- A different selection uses the scientific baseline and refreshes only
  its source-bound distribution/pair preview from a deterministic reservoir.
  No sample rows or source tables are stored in SQLite. The existing column,
  summary, quality, and relationship values are reused without re-estimation.
- The source is fully checksum-verified by the normal workbench-context gate,
  by the preview's own `inspect_file`, and by the post-computation workbench
  check. A workbench generation/revision conflict still prevents writes.
- Explicit `refresh=1` always recomputes the full A2 profile, replacing the
  baseline. A malformed or missing baseline falls back to full computation.
- SQLite's optional table, existing 2 MiB/report cap and 24 entries per
  instance are retained. A cache eviction preference keeps the scientific
  baseline while counting it toward the same 24-entry limit. No migration and
  no additional DB are required; rollback remains v3-compatible.
- Discrete pair counts now use a sparse group-by rather than a dense matrix.
  Visible cell counts, categories, omission flag and global maximum remain
  identical for the same sample; the UI still displays a bounded 8x8 view.
- The UI distinguishes direct preview cache hits, science-only reuse, and
  full computation.

## Validation

Automated tests compare the cached-replay output with full fresh profiles
across multiple column and pair selections. They also assert no second A2
profile is run, source changes are rejected, the cache remains bounded, the
baseline survives eviction, SQLite stays version 3, and sparse category counts
match a dense reference for small inputs. High-cardinality tests prohibit
allocation of a dense crosstab entirely.

## Remaining performance limits / future work

- A new preview still performs one source inspection/checksum and reservoir
  scan. No unchecked mtime-only cache or stale dataset optimization was added.
- The initial cold report still follows A2's existing full profile path; its
  initial structural/source scanning was not rewritten in this sprint.
- Large XML collection comparison, held-out representation benchmarks,
  long-running job scheduling, and broader SQL profiling remain outside this
  targeted stabilization patch. A measured full-application benchmark is
  required before attributing percentage-level speedups.
- Flask-dependent mounted UI integration tests must be rerun in a configured
  application environment. Headless unit tests do not establish browser UX.

## Rollback

Revert ML-UW-13 source changes only. All extra cache entries are stored in
`workbench_feature_profile_cache`, which previous builds already tolerate.
Do not remove historical experiments, artifacts, or project-local databases.
