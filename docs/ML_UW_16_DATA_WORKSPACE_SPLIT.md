# ML-UW-16 — Separate Tabular Data, XML Data, and Postcollection Utilities

## UI ownership (internal)

The Tools sidebar retains five primary sections: Data, Feature Profiling, Representation, Comparison, and General ML Tools. Selecting Data reveals three muted secondary links below General ML Tools: Tabular Data, XML Data, and Post-Collection Utilities. All historical routes are preserved for bookmarked links and PAH integrations.

- **Tabular Data** is CSV/TSV-specific intake, inventory, and source-table access. Its UI routes pass an explicit accepted-suffix filter through the existing application task and profile-job runner; XML files inside a mixed directory are not parsed by that UI path. Explicit XML file inputs receive an explanatory form error (no implicit XML processing). The headless API and CLI retain their existing mixed-source defaults for backwards compatibility.
- **XML Data** owns single-source hierarchy, collection inventory (C1), structural comparison (C2), compatibility review (C3), shared recipe (C4), export (C5), provenance (C6), typed previews and catalog. Its legacy `/data/xml/...` endpoints are maintained, but the XML Data section is active throughout their routes.
- **Post-Collection Utilities** is a read-only launcher for existing non-destructive CSV/TSV cleaning/transformation, lineage tracing, dataset comparison and experiment-dataset setup. No automatic job execution, extraction, experiment creation or data mutation occurs on entry.

XML-generated CSV/TSV datasets can still be selected explicitly and analyzed in Feature Profiling, Representation and Comparison; UI sections do not imply separate experiment stores.

## XML-C2 source count

The previous 100-source UI default and 200-source hard ceiling are removed. By default, C2 compares **all successfully parsed sources contained in the XML-C1 inventory**. The optional `max_sources` argument is retained for advanced callers who want an explicit safety budget; it is no longer implicitly supplied by the XML-C2 UI or CLI. Inventory discovery retains its independently user-configurable `max_files`, size and parser-safety guards.

**Scale caveat:** C2 still materializes pairwise evidence (`n * (n-1) / 2`) and is synchronous. This patch does not provide streaming or paginated comparison artifacts. C3 compatibility review still has its independent 80-file limit. Future performance work should address these distinct limitations without silently discarding source files or claiming a full comparison when one was not performed.

## Acceptance / rollback

- Tabular Data accepts mixed directories but only processes `.csv`/`.tsv`; the core CLI and Python API still inspect XML unless given an explicit filter.
- Existing XML bookmarks resolve under their historical paths and activate XML Data navigation.
- The Post-Collection landing uses existing non-mutating GET routes and has no new persistence.
- XML-C2 handles more than 100 parsed sources while preserving the original source-fingerprint safety checks and advisory-only grouping results.
- No schema migration or new third-party dependency.

Revert ML-UW-16B to restore the previous seven-link sidebar; revert ML-UW-16 to restore the earlier XML-C2 source-count behavior. Previously created data and experiment records are unaffected.
