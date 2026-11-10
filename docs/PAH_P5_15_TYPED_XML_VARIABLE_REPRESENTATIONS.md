# P5-15 — Opt-in typed XML variable representations

## Ownership and compatibility

`ml_lab.data.xml_typed_representation` defines `ml-lab.xml-typed-variables@1`
and a separately versioned `ml-lab.xml-typed-recipe@1` declaration. The existing
XML C1–C7 extraction, review, selection, preview, materialization, datasets and
provenance remain authoritative and unchanged. This is an optional programming
interface, not a replacement extractor or a new default setting.

## Scope

- `describe_xml_variables(structure_artifact)` describes potential scalar, object,
  list, content-token and path-token representations using the **existing**
  structural profile, without reading record instances.
- `build_typed_recipe(...)` records explicit representation choices, a record
  root and optional linked-child paths. It never converts them into a legacy
  extraction recipe or changes the extraction engine.
- `represent_xml_element(parsed_element)` makes an in-memory representation
  from an element already parsed by a trusted caller. It preserves child order,
  same-tag occurrence indices, expanded namespace names, attributes, mixed
  content text/tail tokens and null-vs-empty text. All scalar values remain strings.
  No file reading is performed and explicit depth/node bounds apply.

## Deferred integrations

No XML parsing policy is bypassed: callers must use existing safe ingestion
and source/provenance verification before passing record elements. No UI
changes, cross-module adapters, numeric encoders, SQLite tables or tabular
materialization are included. A future integration must add explicit record-level
source identity, cardinality and missing-field semantics; the present artifact
is **not** a ready-to-train dataset and does not claim full end-to-end lineage.

Rollback: delete this opt-in module, its tests and this document. No existing
schema or storage changes are required.
