# ML-UW-18 — XML field identity and null provenance (internal)

## Scope

This is an **opt-in foundation** for the later union/group/recipe/export sprints. It does **not** alter the existing C1–C7 similarity decisions, C3 grouping, C4 field approval, C5 preview, or CSV/TSV export. No UI is added, no source XML is written, and no SQLite schema is changed. The public entrypoints are:

- `ml_lab.data.catalog_xml_null_fields({source_id: inspected_artifact}, record_root_canonical_path)`
- `ml_lab.data.preview_xml_null_provenance(path, inspected_artifact, catalog, source_id, max_records=50)`

The field catalog represents a **union of canonical candidate scalar field identities**, not a decision that all structures are mutually compatible. Each entry has a collision-free `column_key == field_id`, a human-readable `display_path_hint`, a namespace-aware canonical path, and `present_in`, `missing_in`, `optional_in`, and `repeated_in` source lists. The final output column name belongs to the later shared-recipe stage, which must handle a source `Comments` field collision explicitly.

Selection includes element text and attributes exposed by existing XML field selection, **plus leaf elements that are entirely empty/nilled** in a source. This matters because an empty element is present, even when no nonempty text appeared in structural profiling. Repeated leaf elements are not coalesced into scalars.

## Per-record provenance

For each bounded record, `cells` contains one entry per union field, with:

- `field_id`, `column_key`, `status`, `occurrence_count`, and `observations`
- `observations[].status`, `value`, `null_reason`, `raw_text`, and `occurrence_address`
- `value` contains source text without implicit coercion; missing/nil produces Python `None`; an empty/whitespace-only element produces `""` with `status=empty_element` rather than `None` (original whitespace, if any, is preserved separately as `raw_text`)

| Condition | Status | Value | Null reason |
|---|---|---|---|
| Element absent in this record | `path_absent` | `None` | `path_absent` |
| Parent element present but attribute missing | `attribute_absent` | `None` | `attribute_absent` |
| Element with true `xsi:nil` | `explicit_xsi_nil` | `None` | `explicit_xsi_nil` |
| Present element with no non-whitespace text | `empty_element` | `""` | `None` |
| Present nonempty element text or attribute | `value` | source string | `None` |
| Multiple matching occurrences | cell status `multiple`; each observation retains its own condition/value | no synthesized scalar | as applicable per observation |
| Broken XML or forbidden DTD/entities | parsing exception | **no valid record** | not a null reason |

Element paths and XML namespace identities use the existing canonical escaped-name format. `occurrence_address` is stable within an inspected record and matches the existing typed-record lineage convention. A missing field has zero occurrences and one synthetic provenance observation with `path_absent`, rather than a fabricated XML element.

## Verification and invariants

- Always re-inspect the XML with the **existing safe parser** and reject changed source fingerprints. A second SHA-256 check after preview catches concurrent modifications.
- Catalog and preview are read-only. Neither is a materialized dataset; callers must not present these results as C4 recipe approval or C5 exported values.
- `max_records` bounds in-memory preview to 1–500 **records per call**. This is not a collection-size or XML-C3 selection cap.
- Repeated paths, empty values, explicit nil, absent attributes, and malformed input remain distinct. The later recipe must choose aggregation/child-table strategies for repeated values rather than assume one scalar.
- Planned ML-UW-19: use this same catalog to reason about optional paths and true structural conflicts. ML-UW-20–21: generate explicit normalized nulls and structured `Comments` provenance; preserve CSV/TSV round-trip semantics. ML-UW-22–23: render group and consolidated graph views from shared structure.

## Deployment / rollback

This additive patch only introduces a data module, exports its public names, adds tests, and this internal note. Reverting the patch removes those APIs without changing existing XML extraction behavior or stored files. The user must still verify any later Flask/PAH integration on the target machine; ML-UW-18 itself has no routes.
