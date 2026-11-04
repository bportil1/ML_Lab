# Sprint 2.2 — Minimal scientific matrix editor and non-destructive drafts

**Status:** Implemented in ML_Lab; Data Lab source browsing stays read-only.

## Usage

1. In **Data Lab → Open source**, choose **Edit a copy as matrix** for a fully
   parsed, headed CSV/TSV without malformed-width rows.
2. Use the grid as a conventional numerical matrix. Click or drag to select,
   Shift-click to extend, arrow/Tab to navigate, double-click/type/F2/Enter to
   edit, Enter to commit, and Escape to cancel. Ctrl/Cmd+C and Ctrl/Cmd+V copy
   and paste rectangular TSV selections. Undo/redo apply to the unsaved draft.
3. **Save derived dataset** creates a new CSV or TSV in ML_Lab's configured
   `derived_output_root` and writes the existing Data Lab provenance sidecar.
   The original is never overwritten. To edit the saved result, open its new
   dataset link and start another draft.

The grid uses intentionally neutral styling (pyPIQUE-style scientific tables)
independent of PAH chrome. Values are rendered and copied as exact CSV/TSV
strings, without numeric formatting/rounding. Signed integers, decimal/scientific
notation, missing (empty), and quoted tab/newline fields are supported.

## Boundaries and current limitations

- Editing is **only** allowed on a separate unsorted, unfiltered source page.
  Data Lab's original searchable/sortable source table remains read-only.
  Coordinates are 1-based **data rows** and 0-based columns, not visual sort
  positions. The viewport loads 25, 50, or 100 rows at a time and does not
  claim that unloaded rows are blank.
- Edits remain in a browser draft until explicit save. The server validates
  column types, coordinates, width, source fingerprint and edit count (5,000
  cell limit); attempts to save an unchanged draft are rejected. Server-side
  edits are always processed against the original source bytes.
- Saving does **not** automatically revise an experiment's selected dataset
  or invalidate historical runs. Sprint 2.3 adds an explicit **Use in
  experiment** action after saving, with a separate confirmation and mapping
  step; earlier runs remain pinned to their original revisions.
- The output directory is controlled by the application's configured
  `derived_output_root`, not a browser-supplied path. PAH should configure this
  under its selected project to preserve project-local storage boundaries.
- Formulae, column resizing, editing across filtered or sorted views, multi-
  page selection, automated retransformation of existing recipes, and full
  spreadsheet virtualization are **not** provided by this first editor.
- This patch does not alter the shared `interactive-table` or source-browser
  behavior used by other ML_Lab pages.

## Reproducibility

Each save records the original source SHA-256, exact coordinates and old/new
strings, a derived SHA-256 and a `ml-lab.transformation-event@1` provenance
sidecar. S2.1 `describe_selection()` accepts this derived table when its output
root lies within the current experiment project and the derived file is valid.
