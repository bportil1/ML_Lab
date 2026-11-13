# ML-UW-20 — Null-tolerant shared recipes

Status: implemented as an additive C4/C5 extension, pending Flask/PAH runtime acceptance.

## Contract

- XML-C4 defaults to `allow_missing_fields=True`, with an explicit strict mode (`False`, or `--reject-missing-fields`). A source with a verified shared record root but no selected fields is retained and yields all-null rows.
- Every validated recipe includes `null_alignment` (`ml-lab.xml-null-alignment@1`) containing selected canonical field IDs, XML namespaces, source membership, field kind, canonical paths, and structural absence reasons. This is derived from the ML-UW-18 canonical inventory, not inferred from display labels.
- Missing paths and attributes become **Python `None`** in the C5 in-memory row representation; all main/child output rows have explicit keys for their displayed columns. Repeated-field strategies and shared-root validation remain mandatory.
- Collection extraction (not legacy single-file extraction) preserves empty elements as `""`, and explicit `xsi:nil="true"` as `None`. Absent attributes and paths are `None`.
- Bounded collection preview returns `null_observations_by_source`, with per-record, per-field condition codes and occurrence addresses from ML-UW-18. These observations are record-root based; they are **not** a fabricated one-to-one mapping onto rows expanded by repeated-branch strategies.
- Source hashes, structural review, merge restrictions, row limits, and explicit export confirmation remain enforced. No source XML is modified.

## Boundary / ML-UW-21

CSV and TSV have no inherent null type. This increment does **not** claim that a blank CSV cell distinguishes an empty string from null. ML-UW-21 must implement a defined round-trip null encoding and a reserved `Comments` metadata column using recorded field IDs, as well as full export-level per-cell lineage for transformed/exploded rows. The `Comments` column must be excluded from ML feature matrices by default.

The optional bounded `null_observations_by_source` is available in preview only, not yet serialized per row to the exported CSV/TSV; do not claim full export-level per-cell condition traceability from a source-wide recipe contract alone.

## Acceptance

Run `pytest -q tests/test_ml_uw_20_null_tolerant_recipes.py tests/test_xml_collection_recipe.py tests/test_xml_collection_export.py tests/test_ml_uw_18_xml_null_provenance.py` and then the non-Flask XML/workbench regressions. Flask-backed UI and actual PAH installation require separate testing.

## Rollback

Revert this patch with Git before applying any subsequent increment; no schema migration or source files are changed. Old validated C4 recipe files still revalidate with their saved explicit `allow_missing_fields` selection.
