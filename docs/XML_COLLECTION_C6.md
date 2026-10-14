# XML-C6 — Multi-file provenance, verification and cautious recovery

C6 follows XML-C1–C5. It does not change the reviewed C4 schema compatibility
rules or the existing single-file XML extraction implementation.

A successful consolidated C5 export now publishes four classes of artifacts:

- Consolidated CSV/TSV with `__source_file`/`__source_row` columns.
- Optional child tables with source identity and linkage.
- `.collection.json`: materialization manifest, committed last.
- `.provenance.json`: authoritative `ml-lab.xml-collection-provenance@1` event.

The event includes *each reviewed input* in order with its SHA-256, the exact
validated C4 recipe snapshot and hash, source-specific row counts and missing
fields, output hashes (including child tables), and a deterministic event ID.
The C4 recipe only lists selected inputs; it does not prove which unrelated
XML files were excluded, and C6 does not invent such exclusion history.

A `.collection.checkpoint.json` sidecar is written before a long extraction.
This is an **unauthoritative replay intent**, not successful lineage. It is
removed after a complete commit. It stays behind after exceptions/process
interruptions for inspection; no unfinished result is adopted automatically.
Recovery is allowed only when no matching output artifacts already exist and
sources still pass the C4 hashes and structural validation. Partial or
preexisting outputs block recovery for human review; no files are deleted.

## CLI (module-local, no edits to the central command parser)

```sh
python -m ml_lab.data.xml_collection_provenance verify --provenance analysis/data/combined.csv.provenance.json
python -m ml_lab.data.xml_collection_provenance inspect-recovery --output analysis/data/combined.csv
python -m ml_lab.data.xml_collection_provenance resume --output analysis/data/combined.csv --confirm
python -m ml_lab.data.xml_collection_provenance regenerate --provenance analysis/data/combined.csv.provenance.json --output analysis/data/regenerated.csv --confirm
```

The local UI provides the same operations at `/data/xml/collection/provenance`.
Regeneration is a fresh, fully revalidated extraction. Output checksums are
compared to the recorded outputs; a mismatch is reported as
`different_output`, **not** authoritative reproduction. Source changes or
missing inputs block regeneration. Re-generation requires a new output path
unless an explicit CLI overwrite is requested.

## Limitations / XML-C7 handoff

C6 deliberately replays the whole recipe instead of partially trusting
interrupted intermediate CSV files. It relies on C5's bounded in-memory
extraction and best-effort rollback on handled filesystem errors. A hard kill
mid-publication can still leave partial files; these are flagged/blocked for
manual remediation rather than quietly considered complete. Long-running
streaming, restartable per-source checkpoints, and transactional backup
recovery across hard kills belong to XML-C7.
