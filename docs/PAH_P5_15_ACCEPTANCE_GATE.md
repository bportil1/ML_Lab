# P5-15 — Typed XML acceptance gate (offline)

## Verified by the regression/acceptance tests

The P5-15 opt-in representation workflow supports versioned recipes, bounded
record previews, immutable content-addressed preview snapshots, complete JSONL
record materialization, SHA-256 integrity checks and source-qualified record
identity. No typed representation is automatically encoded or registered as a
scientific dataset. Legacy C1–C7 XML extraction remains authoritative and is
not replaced by this workflow.

`tests/test_xml_typed_acceptance_p515.py` exercises a real namespaced/mixed XML
source through structure inspection, preview, snapshot publication, full
materialization and verification. It checks repeated values, missing vs empty,
record parity between preview and complete output, deterministic publication,
project isolation, source-change rejection and failure without a complete
manifest. These checks run without a web server or network access.

## Remaining before sprint sign-off

- Run the actual Flask app's XML structure page and POST APIs for preview,
  save-preview, and full materialization in the installed environment.
- Verify browser controls, confirmation, error rendering, and actual artifact
  navigation manually (or through installed browser tests).
- Execute the legacy C1–C7 UI workflows on a representative source and compare
  prior dataset/CSV/recipe outputs; no parity claim is made from source-only tests.
- Apply patches to the target checkout with `git apply --check`, then rerun its
  installed test suite. The recovered archive is not a substitute for this.

Until these are verified, P5-15 is **implemented, acceptance pending**, and
P5-16 must not depend on unverified hosted behavior.
