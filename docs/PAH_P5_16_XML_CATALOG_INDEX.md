# P5-16 — XML artifact catalog, first additive increment

This patch introduces an **opt-in, disposable, per-project** SQLite index at
`analysis/ml_lab/xml_catalog.sqlite3`. It is separate from ML_Lab's experiment
store and does not become authoritative for sources, recipes, datasets, or
artifacts. There is no migration of the existing C1–C7 XML storage paths.

`rebuild_xml_catalog(project_root)` verifies immutable P5-15 preview snapshots
and full typed-materialization manifests (including their JSONL bytes) before
inserting metadata in a replacement transaction. Invalid/tampered artifacts are
skipped and counted, never repaired or overwritten. `list_xml_catalog` provides
bounded, paged metadata queries optionally filtered by source SHA-256 and kind.

This is an artifact-reference index, **not yet the complete XML source and
collection catalog**. Recipe indexing, explicitly selected source inventory,
rebuild orchestration and user-facing catalog navigation remain for subsequent
P5-16 increments. Refresh is explicit: callers rebuild after publication or
file removal. Queries are metadata-only and never automatically revalidate
source files; `stale_until_rebuilt` is always true to avoid false freshness.

The index is project-local, fully rebuildable from producer-owned artifacts,
requires no networking, and never registers typed data as a numerical dataset.

## Acceptance boundaries

- Python-only tests cover empty projects, isolation, deterministic pagination,
  hash-verified artifacts, corrupt-artifact rejection and future-schema rejection.
- No Flask/browser behavior is claimed validated in this increment.
- Prior P5-15 hosted/browser acceptance gates remain open and independent.
