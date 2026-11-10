# P5-16 — XML source inventory and collection references

This is an **opt-in, additive** second increment. `save_xml_inventory_snapshot`
accepts an existing `inventory_xml_collection` output and saves a bounded,
immutable, content-hashed JSON snapshot at
`artifacts/xml_collection_inventories/<sha256>.json` in the owning ML_Lab
project. No original XML source is changed or copied. Source paths reflect the
local inventory observation and may become stale if XML files change.

`rebuild_xml_source_catalog` authenticates the snapshots and transactionally
rebuilds `xml_collections` and `xml_sources` inside the existing per-project
`analysis/ml_lab/xml_catalog.sqlite3` index; invalid snapshots are counted and
skipped. `list_xml_sources` offers filtered, bounded, paged source metadata.
This reuses the P5-16 foundation DB and does not create a second project
catalog. SQLite is **disposable** and never authoritative for original XML,
existing C1–C7 recipes, or typed materializations. Existing artifact rebuild
and catalog queries remain unchanged.

Snapshots represent **observed inventories**, not validated live current source
state. Consumers must re-run inventory before relying on freshness. Collection
recipe indexing and UI exposure remain separate future P5-16 increments. The
new operations are library APIs only; no automatic save, external sharing,
Flask modification, dataset registration, or source write occurs.
