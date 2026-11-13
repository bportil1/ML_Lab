# ML-UW-11 — XML workflow acceptance and cleanup

## Scope and verified changes

This increment preserves the existing XML-C1–C6 and single-document extraction engines. It does **not** add a second XML parser, modify extraction recipes, or automatically select a dataset, launch training, or write an output without the existing explicit confirmation.

- **Single file:** the successful materialization JSON now includes `workbench_select_url`, but only when the output is inside the owning experiment project and the unified workbench route is installed. The browser shows a **Use this dataset in workbench** link, which opens the existing verification/selection form; it never activates the dataset automatically. Outputs outside that project retain the normal Transform path but receive no cross-project activation link.
- **Collection:** C5 uses the same project-scoped handoff helper for the generated main CSV/TSV. Collection review's back link retains directory, recursion, visibility, include/exclude globs, file limit, and similarity threshold. A GET of inventory restores only the form, with no implicit filesystem scan; the user must explicitly inspect again.
- **Labels:** group actions are titled **Prepare full-group extraction recipe**, because the action validates and prepares the recipe but does not yet extract all rows or generate a CSV.

## Deliberate limits

- Originally C3 review was limited to 80 XML files. ML-UW-17 removes that ceiling and propagates the caller-selected inventory budget through C3 and C4; comparison remains synchronous and quadratic. The inventory's explicit count, size, and parser-safety limits remain applicable.
- Typed XML materialization remains an independent structured artifact, not automatically flattened into a numeric dataset.
- Back-navigation preserves filters but not a persisted review approval; returning to an earlier stage requires deliberate reinspection.
- Source files are never modified by the added UI handoffs. `workbench_select_file` remains responsible for verifying the generated dataset, provenance, schema, and generation before user-confirmed activation.

## Validation

`PYTHONPATH=src pytest -q tests/test_xml_* tests/test_ml_uw_*`

At patch creation: **180 passed, 20 skipped, 1 warning**. Flask was not installed in the local test environment. Those skipped browser tests require a separate runtime acceptance check on a machine where the optional Flask UI is installed. A fresh checkout based on ML-UW-10 passed `git apply --check` and a focused replay (**22 passed, 7 skipped**).

## Manual browser acceptance gate

1. In PAH's mounted ML_Lab, open Data → XML extraction workspace, inspect a local XML document, preview a record-root/field selection, confirm, and materialize a CSV inside the current experiment project. Confirm that **Use this dataset in workbench** opens a read-only verification form, not an automatic activation.
2. Repeat with an output outside the project and confirm that the activation link is omitted while Transform remains available.
3. Inspect an XML collection, choose filters, navigate review → inventory, and confirm inputs are restored but no scan has started.
4. Prepare a full-group recipe and confirm that no CSV is created until explicit preview and export confirmation. Verify C5's project-local dataset activation form and source provenance.
5. Verify the Data → Feature Profiling → Representation → Comparison dataset identity after explicit activation. Revisit prior experiments to confirm their records remain unchanged.

## Rollback

The increment contains only UI, route glue, tests, and these notes. No schema migration or persisted-format change was made. Revert this patch without modifying historical experiment databases or XML artifacts.
