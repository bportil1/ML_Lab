# ML-UW-21 — Null-aware XML collection exports and Comments

## Scope and contract

- XML-C5 **and** restart-safe XML-C7 create a reserved `Comments` metadata column in consolidated main tables and supplemental child tables. A non-null field has no entry in `Comments`; null fields are annotated with a JSON object mapping the **output column name** to a machine-readable reason.
- Reasons: `path_absent`, `attribute_absent`, `explicit_xsi_nil`, or `unresolved_null` when a one-to-one path/row relationship cannot be established (e.g., exploded/pivoted extraction). Never invent a source-specific reason from a mere blank cell. Missing paths alone are not structural incompatibilities.
- C5 in-memory preview uses native Python `None` for missing/null data and `""` for empty elements; its `Comments` value is a JSON text string. The export adopts a **versioned, lossless delimited encoding**, recorded in `main_dataset.null_encoding` and each child dataset record.
- CSV/TSV output uses `\N` for null and `\E` for the empty string. Any non-null literal beginning with `\` gains a leading backslash, so literal `\N`/`\E` strings round-trip unambiguously. The ordinary CSV library does not decode these tokens by itself.
- `read_null_aware_collection_table` verifies the exact output SHA-256 and its local manifest before decoding; the Data Lab provenance, comparison, transformation, data-table, and profiling paths recognize **manifested main collection outputs only**. Other delimited files retain their prior behavior. `Comments` remains metadata and is excluded from default experiment features; a user can explicitly opt in.
- Parsing failures are errors, not ordinary null rows. Source XML, checked recipe, and C6 lineage remain immutable. Current exporter paths still require explicit confirmation; no database migration is introduced.

## Source-to-column attribution

C4 canonical field IDs are matched to actual named output columns. Per-record XML null conditions are used only for one-output-row-per-record mappings; source-wide missing fields may be explained for any output row. Composite/repeated ambiguous nulls are marked `unresolved_null`. Empty values are not promoted to null. A source with no selected fields still contributes null-filled rows in both C5 and C7.

The C7 source-part checkpoint schema is now `ml-lab.xml-collection-source-part@2`; pre-existing in-flight C7 parts are rejected rather than silently interpreted under the newer contract. Completed historical exports retain their original manifests and byte content.

## Verification and limitations

Automated coverage includes XML `xsi:nil`, absent path/attribute, empty text, absent-all-fields, literal encoding tokens, C5/C7, CSV/TSV, tampered SHA, generic CSV non-regression, and dataset selection default-feature behavior. Browser tests still require an installed Flask environment.

**Large dataset behavior:** C5 remains bounded in-memory; C7 stages one source at a time, though per-source reason evidence adds work proportional to selected fields. Data Lab full-table scientific workflows intentionally load entire selected datasets; paging/profiling do not need to load all rows merely to verify the export manifest. Future performance work should profile large XML sources and selectively materialize null-provenance evidence.

## Next XML sprints

- **ML-UW-22**: graphical per-group XML tree with paths, counts, optionality, and null condition summaries.
- **ML-UW-23**: one interactive consolidated union-structure viewer.
- **ML-UW-24**: deployed Flask/PAH acceptance, export and reimport scenarios, source-safety checks, end-to-end documentation.
- Future P1: reproducible sampling of large pairwise comparisons, clearly labeled as partial evidence.
