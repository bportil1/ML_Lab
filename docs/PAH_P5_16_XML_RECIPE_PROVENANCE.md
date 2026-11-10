# P5-16 — XML collection recipe evidence and provenance index

This additive third increment introduces immutable, **opt-in** snapshots of
existing, structurally validated C4 XML collection recipe outputs. It does not
change the C4 recipe algorithm or extraction/CSV behavior. A snapshot preserves
the original recipe and its per-source SHA-256 fingerprints, and can explicitly
link to a previously saved, verified XML inventory snapshot. When linked, each
selected source fingerprint and the collection root must match the inventory.

`save_xml_recipe_snapshot` stores the complete, bounded recipe at
`artifacts/xml_collection_recipes/<sha256>.json` using content-addressed storage.
`read_xml_recipe_snapshot` validates its content hash, the constrained C4
provenance contract and the linked inventory (if any). A snapshot is **observed
evidence**, not proof that source XML is unchanged today; caller must re-inspect
sources before running extraction.

`rebuild_xml_recipe_catalog` rebuilds only `xml_recipes` and `xml_recipe_sources`
tables in the existing project-local `analysis/ml_lab/xml_catalog.sqlite3`.
The catalog is a disposable SQLite index; rebuilds of other catalog sections do
not discard recipes. `list_xml_recipes` supports bounded, paginated lookups by
inventory snapshot ID and/or source SHA-256. Index rebuild skips invalid
snapshots and does not edit authoritative artifacts.

No new UI, automatic recipe saves, dataset registration, SQLite data authority,
network access, or source-file mutation is introduced. Legacy XML C1–C7 paths
and numeric encoders remain unchanged. The future workspace catalog can reuse
these metadata APIs after hosted/browser acceptance.
