# Structured Uncertainty: direct single-file intake source repair

## Original defect

The ML-UW-36 selector accepted only a project-relative `.json` filename, defaulted
its reader to `StructuredObject.from_json`, and had no way to browse existing
project files. Consequently a standalone XML, CSV, TSV or ordinary JSON source
could not enter the Structure / Feature Profiling workflow through this page.
The UI displayed **Action unavailable: select a project-relative .json artifact**
without presenting a viable single-file route.

## New local workflow

1. Open **ML_Lab → Structured Uncertainty** (or Data → Structured data).
2. Choose **Browse project files** to list up to 120 project-local XML, CSV, TSV,
   and JSON files from a bounded scan, or type a path within the configured
   ML_Lab project. Absolute paths are accepted only when they resolve inside
   the same configured project directory. Directory or file symlinks are denied.
3. Press **Open file** (from the browser list) or **Open and verify file**.
   **Auto** detects supported formats and versioned JSON contracts.
4. The original file is read and SHA-256 pinned. All graph views and sandbox
   requests re-check the file and do not change it. For a revised file, select
   it again. No data is uploaded, copied, trained, or stored in a new database.

## Scientific scope

- `.xml`: feeds the existing hardened XML inspection pipeline and the
  `XMLStructureAdapter`. The resulting tree contains canonical *paths*,
  occurrence summaries, repeated/optional structure and inspected attributes.
  It **does not claim** individual occurrence-level topology, aligned samples,
  MI, variance, or statistical dependencies. For record-level information use
  the explicit XML tabularization / collection workflow first, then select
  the resulting CSV/TSV (with the applicable semantics).
- `.csv` / `.tsv`: explicit header fields become column nodes beneath a
  table root, with one aligned record per data row. Values are preserved as
  **strings**, including leading zeros; blank cells remain `empty`, not
  implicitly `missing`. Sampling independence, numeric typing, graph
  dependencies and source-to-node SU-05 bindings are not inferred.
  For numeric variance, prepare an explicitly typed StructuredObject or use
  a deliberate data-typing workflow before scientific estimation.
- `.json`: Auto uses `StructuredObject`, canonical XML profile, generic
  node-edge, or known shared/pyPIQUE graph contracts when declared; other
  JSON values use the existing `JSONAdapter`. Mixed-direction graphs still
  require an explicitly selected projection.

## Scope and security

The 2 MiB interactive source limit and 10,000-record limit remain in place.
The on-demand browser omits symlinks and common build/virtualenv folders;
the user can still enter a deeper supported path manually. No OS-wide file
picker, uploads, external services, cross-project database access, or module
ownership changes. Original five-tab navigation, JS viewer, SU-05–SU-11
scientific contracts and stored artifacts are unchanged.

## Validation

```bash
PYTHONPATH=src python -m pytest -q tests/test_structured_single_file_intake.py tests/test_ml_uw_36_structured_workbench.py tests/test_su10_structural_sandbox.py tests/test_su11_acceptance.py
```

Full PAH-hosted browser acceptance should use an actual XML and CSV from the
configured active project. Review the displayed source kind, support counts,
missingness, and warnings before interpreting numerical estimates.
