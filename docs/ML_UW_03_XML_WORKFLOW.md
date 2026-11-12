# ML-UW-03 — XML extraction to workbench dataset handoff

## Changes

- **One XML entry point:** `/data/xml/workbench` selects existing single-file XML structure exploration or collection compatibility review. All legacy XML routes stay in place. There are no new XML extraction algorithms in this increment.
- **Collection workflow:** inventory/review → shared recipe → preview → explicitly confirmed CSV/TSV export → local provenance verification → *explicit* active dataset selection. The recipe is handed directly from C4 to the C5 preview by the existing form; no manual JSON paste is required for this path. The XML stage indicator does not navigate to forms that would lose POST state.
- **Output destination:** the UI's default XML collection output is now under the configured experiment project root. Users can still select an alternative path; only an output *inside* the project can be bound to the active instance.
- **Dataset context:** the read-only GET `/workbench/dataset/select-file?path=…` examines a project-local CSV/TSV. The explicit POST verifies exact dataset ID and byte checksum, checks the optimistic workbench generation, selects feature and target columns and creates/pins an experiment revision with `select_workbench_dataset`. Older experiments remain intact. The dataset is not activated automatically when exported.
- **Provenance:** existing Data Lab provenance semantics are preserved. Selection also accepts the XML-C6 `ml-lab.xml-collection-provenance@1` schema only after validating the authoritative event, all source and output hashes, and matching the *consolidated main output* to the selected dataset. Stale sources and outputs invalidate the instance on subsequent reads.
- **Navigation:** the Data pane links to the XML entry point; back links avoid presenting empty recipe/export pages as resumable workflows. A completed export supplies its provenance/output fields to the verification form.

## Non-goals and remaining issues

- This patch does **not** remove the 80-file synchronous collection compatibility-review cap. The page states the limit. Efficient batched/streaming schema review remains separate work.
- The existing single-file JS extraction/materialization remains unchanged. Its output may be selected using existing dataset selection paths; a dedicated single-file one-click workbench handoff is still a subsequent integration item.
- The same instance is shared across tabs, but individual legacy tool forms may still need the active dataset automatically prefilled. Avoid implying every downstream tool has been wired before ML-UW-04+.
- No separate XML database or cross-module shared store is added. No auto-flattening, silent XML mutation, or implicit change to the active dataset occurs.
- HTTP/browser tests run only where `flask` is installed. Headless XML export/provenance and workbench tests run without Flask.

## Acceptance

1. Open the new XML workspace, choose a single XML file, and inspect it without writes.
2. Choose a directory, review compatible sources, build shared recipe, preview and explicitly export a consolidated table.
3. Open the generated table's 'Use consolidated dataset' form. Verify the fingerprint and choose features/targets.
4. Activate once; revisiting the workbench context returns the selected output and its provenance.
5. Reject stale workbench generation and tampered XML sources/outputs without changing the active pointer.
6. Verify collection provenance with prefilled paths from the completed export.
7. Legacy XML URLs, source explorer, recovery checkpoints, and previous experiments remain usable.
