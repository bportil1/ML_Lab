# P5-15 — Complete typed XML materialization (opt-in)

## Boundary and scope

`ml_lab.data.xml_typed_materialize.materialize_typed_xml_records` is a Python-only, explicit API for full source-record extraction into a project-local immutable JSONL stream. It does not register a dataset, flatten objects, numerically encode values, modify existing C1–C7 recipes, or change XML UI defaults.

Before streaming, the existing safe structural inspector validates the source, and the typed recipe is checked against the actual structural profiles. The source must match the inspected SHA-256. A second source SHA-256 check after streaming prevents publication following a detected source mutation. The writer compares extracted record counts with the authoritative structural record count. Record and output-byte limits fail closed; an exception leaves no manifest claiming completeness.

Published files are stored in `artifacts/xml_typed_records/`: one content-addressed `.jsonl` stream and a separately content-addressed `.json` manifest, both immutable. A repeated invocation reuses only files with matching SHA-256. `verify_typed_xml_materialization` checks the manifest and full stream hashes without loading the entire stream into memory. The source XML is not stored in either artifact.

The output includes ordered records, present/missing distinctions, variable path and occurrence address, linked child records, source fingerprint and recipe. Record IDs combine source SHA-256, record-root path and source-order ordinal; **ordinals are not stable across source changes**. `complete: true` means all matches for the selected record-root path in the verified source were written, not that every field in the XML was selected. The processing memory is bounded by the size of a single record subtree and individual rendered values; very large individual records remain subject to explicit node limits.

## Compatibility / remaining work

This increment intentionally offers no web Save/Materialize button, SQLite index, or scientific dataset registration. Cross-repository integration, full browser validation, lineage consumers and canonical project lifecycle controls remain separate acceptance steps. The legacy XML table pipeline is unchanged.
