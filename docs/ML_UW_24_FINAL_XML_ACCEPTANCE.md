# ML-UW-24 — Final XML collection acceptance and integration reconciliation

**Status:** headless XML collection acceptance verified against ML-UW-23; **hosted
Flask/PAH browser acceptance remains open** until run on the user's actual
installation. No external services or hosted data are required.

## End-to-end coverage

The acceptance fixture exercises the same source set through:

1. **C1 inventory:** read local XML inputs and fingerprint them (no writes).
2. **C2 comparison:** namespace-aware candidate groups and one observed
   consolidated union; incompatible expanded root QNames remain separate.
3. **C3 review:** variant acknowledgment is required; observations do not
   authorize merging.
4. **C4 recipe:** explicitly select a shared record root and canonical field IDs;
   absent fields use the null-tolerant default.
5. **C5 preview and export:** no files written during preview; per-record missing
   paths/attributes and explicit `xsi:nil` produce nulls with JSON `Comments`;
   a genuinely empty element remains `""`.
6. **C7 checkpointed export:** same table bytes as C5 for the same input, using
   per-source fragments; tested with both `.csv` and `.tsv`.
7. **C6 provenance:** published output/input hashes verify; changed or tampered
   files fail verification. An explicitly confirmed reimport retains literal
   `\N`/`\E` convention and metadata-only `Comments`.
8. **Workbench handoff:** the resulting dataset can be selected as the active
   project-local workbench dataset; `Comments` is not a default feature.
9. **ML-UW-22/23 visualizations:** both viewers consume exactly the same C2
   observed union and distinguish individual groups from the entire collection.

These checks use temporary files. They do **not** create analysis or export
artifacts inside the Git repository.

## Resolved C5/C7 defect: XML QName display-prefix mismatch

ML-UW-24 fixes an integration bug uncovered during the acceptance pass. C2/C4
correctly recognized `<a:name>` and `<b:name>` as the same field when the `a`
and `b` prefixes denoted the **same expanded namespace**, but the C5/C7
exporters incorrectly treated the preview labels `a:name` and `b:name` as
conflicting columns. Both exporters now register columns by their canonical
source field ID, branch strategy, and canonical element path, then remap the
source-specific display labels to the first registered label. The first-source
header is retained for compatibility with previous working collections.

**Safeguard:** two different canonical field identities that merely share a
human-readable name are *not* combined; this remains a validation error.
The same alignment is applied to C5/C7 supplemental child tables. Column
provenance and `Comments` refer to the final displayed output header.
Existing single-file XML preview naming is unchanged.

## Commands for local acceptance

From the ML_Lab repository root, in the configured local environment:

```bash
PYTHONPATH=src python -m pytest -q tests/test_ml_uw_24_final_xml_acceptance.py
PYTHONPATH=src python -m pytest -q tests/test_ml_uw_*.py tests/test_xml_collection_*.py tests/test_xml_typed_*.py
```

Use a working Flask installation for the optional mounted-route test. The
headless environment in which the patch was prepared did not have Flask.

## Required deployment / user acceptance (still open)

On the actual PAH-hosted installation, with *non-sensitive local test XML*:

- [ ] Open the five-tab Tools menu, choose Data → XML Data, and compare a
  directory with two structurally compatible files using different prefixes
  for the same namespace and an unrelated-root file.
- [ ] Confirm both individual group and consolidated union trees render,
  search/expand/inspect work, and source counts agree with the C2 JSON report.
- [ ] Select the compatible group, explicitly acknowledge variants, and
  validate the intended shared root and fields. Verify the incompatible root
  stays excluded.
- [ ] Check the combined preview and `Comments` map distinguish
  `path_absent`, `attribute_absent`, `explicit_xsi_nil` and empty strings.
- [ ] Export CSV and TSV (including C7 if using a large collection), verify
  manifests/provenance, import back, and confirm default features exclude
  `Comments`.
- [ ] Revisit Tabular Data, Feature Profiling, Representation and Comparison
  after selecting the export. Validate mount-aware URLs and project context.
- [ ] Check keyboard navigation and browser runtime interactions, not just
  template compilation. Record any console errors.
- [ ] Confirm the Git working tree stays clean following normal operation;
  generated XML artifacts under `analysis/data/derived/.xml-extractions/` are
  a known independent runtime-output / `.gitignore` audit priority.

## Known limitations / deferred work

- **Flask/PAH browser acceptance is not verified here.** The optional Flask
  route test is skipped if Flask is absent; JavaScript Node unit checks are not
  a substitute for a hosted browser session.
- **Full suite not a release gate in this patch:** run the broader installed
  project suite on the user's machine before treating the sprint as deployed.
- **No sampling yet:** C2 remains exhaustive pairwise `O(n²)`. The future P1
  sampling design must clearly distinguish partial evidence from exhaustive
  comparisons; sampled pairs cannot authorize a shared recipe.
- **No new streaming comparison:** C7 streams extraction across sources, not
  C2 pairwise comparison.
- **Structural safeguards remain:** ambiguous repetitions/mixed content need
  review; missing paths alone do not split compatible groups. No source XML
  rewrite, automatic recipe authorization, or unconfirmed export is permitted.
- **Runtime output hygiene:** Git exclusion/relocation of generated outputs is
  tracked separately in the active universal roadmap; avoid indiscriminate
  ignores for intentional datasets or research artifacts.

## Rollback

The patch adds shared collection-column alignment and tests/documentation.
It does not migrate SQLite or change the existing C2 union contract.
Reverting ML-UW-24 returns to ML-UW-23 behavior, including the prior display-
prefix mismatch on collections with equivalent namespace URIs.
