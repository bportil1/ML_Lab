# P5-15 — Bounded typed record preview (integration increment)

## Purpose and authority

`ml_lab.data.xml_typed_records.preview_typed_xml_records` is an **explicit opt-in** record-level preview using the typed recipe from the preceding P5-15 foundation increment. It reuses `inspect_xml_structure` and its DTD/entity, source-byte, depth and structural limits, checks the inspected source SHA-256 fingerprint before reading values, and validates selected paths against the structural artifact. The original XML remains the authority. The C1–C7 tabular extraction, materialization, provenance and existing routes are untouched.

## Result semantics

- `ml-lab.xml-typed-preview@1` results expose `source_sha256`, `source_fingerprint`, record-root canonical path, per-record index and bounded record values. The supplied recipe remains under the caller's control; the result is **not** a registered dataset.
- Each selected path has explicit `present`, `count`, and `items`; an absent path (`present=false`, zero items) differs from an element whose text is absent or empty. Values are unconverted strings. `list` preserves repeated order; `object` preserves nested children; `content_tokens` preserves mixed text/children/tail order; `path_tokens` preserves canonical paths and occurrence indices.
- `linked_child_paths` produces separately addressable records with `parent_record_index` and sibling occurrence addresses. These are read-only previews; no source mutation, implicit joins or foreign keys are generated.
- `preview_record_count`, `source_record_count`, and `truncated` distinguish bounded preview from complete data. This is *not* yet an ML training matrix or a persisted extraction/materialization result.

## Limitations and follow-up

An in-memory preview of bounded records can still be expensive on very large records; callers should use small `max_records`, and `max_nodes_per_value` imposes a representation bound. The record index is a **preview ordinal**, not a durable document-wide byte-offset or identifier. Exact parent-child relationships for nested repeated record roots and persistent lineage require a separate reviewed materialization contract. No UI controls, SQLite schema, cross-module feature encoders or automatic flattening are introduced in this increment.

Rollback: remove `xml_typed_records.py`, its tests, and this document; the foundation P5-15 types can remain installed independently.
