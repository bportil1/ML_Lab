# XML-C7 — bounded collection-scale extraction, monitoring and catalog handoff

XML-C7 extends the finished C1–C6 collection workflow **without replacing the
single-file extractor or the C5 export route**. It is an explicit opt-in Python
API and local CLI. No external tools, APIs, or network calls are used.

## Running it

1. Use C1–C4 to produce an explicitly reviewed, ready C4 recipe JSON.
2. Invoke `python -m ml_lab.data.xml_collection_scale export --recipe recipe.json --output analysis/data/combined.csv --confirm --progress-json`.
3. Inspect any interrupted progress with `python -m ml_lab.data.xml_collection_scale status --output analysis/data/combined.csv`.
4. After correcting a transient failure, run the same export command with `--resume`. Source, recipe, and staged-part SHA-256 checks are mandatory. Changing the recipe or source means beginning a new export at a new output location; old partial files are never silently adopted.
5. Verify the published `.provenance.json` with the C6 verifier. Regenerate using `python -m ml_lab.data.xml_collection_scale regenerate --provenance analysis/data/combined.csv.provenance.json --output analysis/data/reproduced.csv --confirm`.
6. Inspect a **logical** SQLite mapping proposal with `python -m ml_lab.data.xml_collection_scale sqlite-handoff --manifest analysis/data/combined.csv.collection.json`.

## What scales and what is bounded

- Instead of accumulating all rows from all sources, the runner processes one
  source at a time into a checksum-tracked, local part, then writes the unified
  CSV/TSV from those parts using one source at a time in memory.
- Resuming skips only already finished and checksum-verified source parts.
- Every selected source is revalidated against C4 data. Single-source extraction
  still relies on the existing XML preview engine's **10,000 main-output-row
  bound**. The C7 path does not support unlimited-size individual XML sources,
  arbitrary large nested expansion, or a true row-streaming XML parser yet.
- C1/C2/C4 discovery, structural pairwise checks, and recipe validation still
  run on the selected directory at the beginning (and again during resume).
  Large collections with thousands of files may still encounter those limits.
- Limits include source count/bytes, per-source root records, cumulative main
  and child rows, and child rows per source. Resource limits should be adapted
  deliberately rather than silently truncated.
- Structural incompatibility, unresolved repeated-branch rules, conflicting
  column meanings, source changes and truncated results block publication.
- Existing output files are never overwritten. Publication builds main/child
  data and C6 authoritative lineage in a staged transaction, publishes the
  manifest last, and rolls back on publication failures.

## Job observability and recoverability

The API supports an optional `progress(event)` observer, and the CLI supports
local stderr `--progress-json` events. Events give source counts and lifecycle
phases, not raw XML contents. Observer failures do not alter scientific outputs.
The local checkpoint is explicitly **non-authoritative** and exists only while
exporting or recovering. It is suitable for a read-only PAH Job Monitor adapter,
but XML-C7 does not claim to have completed PAH host-side integration.

A per-output local advisory lock serializes new/resumed exports to that output.
A small `.c7.lock` file persists as a safe lock identity. A failed run retains
its `.c7.state.json` plus hidden source parts. Source hash, recipe hash, and part
hash checks are required before reusing completed work. Recovery never adopts
partial result CSVs as successful datasets.

## SQLite readiness — strictly a boundary contract

`describe_sqlite_handoff()` returns a versioned, independently verifiable
manifest-derived structure containing main/child output paths, hashes, row
counts, source identities and the C6 event ID. A future **ML_Lab-owned** adapter
may map this into `runs`, `sources`, `datasets`, and artifact/source membership
relations. XML-C7 creates **no SQLite database**, runs no SQL migrations and
does not make PAH directly access ML_Lab private storage. Production-grade
SQLite support remains separate mandatory roadmap work (UT-4).

## Testing

`PYTHONPATH=src python -m pytest -q tests/test_xml_collection_scale.py tests/test_xml_collection_export.py tests/test_xml_collection_provenance.py`

Tests cover provenance verification, child tables, source-level resume, corrupted
part rejection, changed-source rejection, non-overwriting behavior, observer
failure isolation, reproducibility, and competing-export lock behavior.
