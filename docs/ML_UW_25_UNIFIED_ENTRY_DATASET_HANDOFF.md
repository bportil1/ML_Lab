# ML-UW-25 — Unified entry and derived dataset handoff

## Problem

The legacy capability dashboard remained at the module's root URL, and the
old `/workspace` page was still advertised in the header. Independent browser
navigation storage could also carry an unrelated `experiment_id` onto tool
links. The five-tab workbench used a verified project-local SQLite dataset
binding, but producing a derived file did not update that binding. Consequently,
subsequent tools frequently asked users to select the same dataset again.

## Behavior

- `/` and `/workspace` redirect to `/data`, including under a PAH mount prefix.
  Legacy task endpoints, saved run details, and read-only `/workspace/context`
  continue to exist for compatibility; only the duplicate launchers disappear.
- The browser no longer executes the superseded navigation-localStorage script
  in the unified UI; the workbench context remains in the project-local
  `ExperimentStore` and its HttpOnly instance cookie.
- An **explicit successful write** of a project-local CSV/TSV automatically
  selects that output for the **current browser workbench instance**. Covered
  routes: XML-C5 collection export, XML-C6 resume/regenerate, single-file XML
  tabular materialization, data transformation Apply, editable-matrix draft
  save, and explicit experiment dataset selection.
- Selecting a dataset through the experiment selection form binds the **same
  saved revision** to the workbench (it does not generate a second experiment).
- Preview, inventory, read-only browsing, and failed writes never change the
  active workbench selection. Other browser instances remain isolated.
- A selection is verified using source identity/provenance. An output outside
  the selected project's folder is never automatically bound. If a new tab
  switches the active dataset while an export is running, optimistic generation
  checking prevents the older export from overwriting the newer selection.
- A write can succeed even if its optional workbench activation fails; the
  UI explains the failure and retains an explicit, verified selection link.
- Identical dataset identities can be reused without creating unnecessary
  experiment revisions; actual provenance changes require a new binding.
- XML collection `Comments` is metadata, not a default model feature. Both
  activation and manually presented selection forms respect that default.

## Acceptance requirements on the installed PAH host

1. Launch ML_Lab from PAH and verify the first visible page is **Data**, not
   the former `Machine Learning Lab` dashboard or the old Workspace launcher.
2. In XML Data, export a sample collection with null provenance. Confirm the
   success page reports automatic activation and that the header reflects the
   new active dataset without reselecting it in another tab.
3. Navigate through all five primary tabs. Feature Profiling, Representation,
   Comparison and General ML Tools should read the same source, experiment
   revision and workbench generation; the `Comments` column is not selected as
   a feature by default.
4. Apply a transformation and save a matrix-editor draft. Both should activate
   the resulting project-local output. Source data and historical experiments
   must remain unchanged.
5. Preview a collection without writing: the active dataset must not switch.
   Repeat with an export destination **outside** the project: the new output
   must not be selected. Confirm an informative error/link is shown.
6. Create a concurrent-tab context change during a slow export. Verify that
   the stale export does not override the newer active dataset.
7. Verify the legacy experiment run URLs and read-only metadata endpoints still
   function while the old landing pages are absent.

## Deployment notes

The source baseline for this patch is the sequentially reconstructed ML-UW-24
implementation, not a fresh snapshot of each independent host/cluster checkout.
Use `git apply --check` first and avoid force-applying on divergent source files.
Flask/browser integration requires testing in the installed environment;
headless regression alone does not prove the PAH mount behavior.
