# Unified Experiment Workflow — S0.2 Scientific Matrix Editor Contract

**Status:** Proposed behavior and appearance for Sprint 2 implementation; **not
implemented by S0.2**.
**Related:** [S0.2 UI conventions](UNIFIED_EXPERIMENT_S0_2_UI_CONVENTIONS.md),
[S0.1 experiment contract](UNIFIED_EXPERIMENT_S0_1_CONTRACT.md).

## 1. Purpose and primary design precedent

The scientific grid should feel like a **natural spreadsheet/numerical matrix
editor**, not a PAH-branded utility. Use pyPIQUE's restrained interactive table
presentation for the visual baseline: compact conventional cells, visible
headers, readable numbers, clear sort/filter behavior and simple CSV/copy
operations. Use the current ML_Lab data/transform views for source-data and
provenance semantics, **not** for adding elaborate styling to the grid.

**As-observed versus planned:** pyPIQUE's `visualizer.js` currently implements
interactive table sorting, per-column filters, global search, copy of visible
rows and CSV export on selected tables. This is a **table interaction precedent**,
not evidence that pyPIQUE already implements rectangular range-editing, paste
commit, formulae, undo or virtualized numerical matrices. All such features
below are requirements to implement and verify in ML_Lab, not transplantable
capabilities assumed to exist.

## 2. Supported display modes and semantics

| Mode | Example | Mutations allowed | Result identity |
| --- | --- | --- | --- |
| Read-only data inspection | Source matrix, feature/target preview, diagnostic values | None | Original dataset/artifact identity |
| Editable draft matrix | User-selected working dataset revision | Local uncommitted draft changes | Base revision + pending edit set |
| Transformed preview | Normalization, encoding, filtering preview | None in preview | Recipe/reference and preview execution |
| Analytical matrix | MI, weights, similarity or learned latent values | None by default | Producing model/run and metric definition |
| Exported/generated matrix | Synthetic samples, learned embeddings | None until explicitly imported as draft | Producing run, seed and artifact reference |

Never edit source arrays, model weights, test labels or immutable run outputs
indirectly through a presentation table. Explicitly distinguish a *matrix data
edit* from a *transformation recipe edit* and from a *model parameter edit*.

## 3. Appearance: minimal scientific surface

- Neutral canvas and cell backgrounds; subtle gridlines; compact row height;
  restrained typography with tabular numerals for numeric values.
- Clearly identifiable row indices and column names (or numeric column
  indices when names are unavailable), pinned headers where practical.
- Visible focus/selection outlines without obscuring numbers. Use color to
  convey selection, validation or an explicitly chosen scientific heatmap;
  never require color to interpret the data.
- No oversized cards, PAH brand gradients, status badges inside each cell,
  decorative charts behind the grid, or tool-specific terminology for standard
  spreadsheet operations.
- Keep actions close to the grid in a compact toolbar; provenance and detailed
  operations belong in optional menus or separate contextual panels.
- Allow horizontal/vertical scrolling in the grid area; keep experiment
  navigation outside the matrix viewport. A narrow window must not force all
  columns into unreadably thin cells.

## 4. Cell value and numerical fidelity

1. The displayed string is a **view of the stored value**, not its serialized
   numerical authority. Adjustable displayed precision must not round or
   overwrite underlying floating-point values.
2. Preserve scientific notation, zero, negative zero where meaningful, NaN,
   positive/negative infinity, null/missing values and text distinctly.
   Prevent confusing `0`, empty input, `NaN` and missing values.
3. Editing a typed numeric cell validates the full parsed value; invalid input
   remains visible as a draft error with a clear correction path. Do not silently
   convert invalid text to zero, or silently coerce string labels to numbers.
4. Preserve declared or inferred column types, headers and exact row ordering
   in the backing dataset revision; never trust a formatted cell's visible
   text as a lossless export representation.
5. An analytical matrix may use custom formatting (e.g., scientific notation,
   bounded percentages) but must label units and method and permit inspection
   or export of full-precision values where supported.

## 5. Expected interaction semantics

| Interaction | Proposed behavior | Guardrail |
| --- | --- | --- |
| Mouse selection | Click focus cell, drag to select rectangular range, Shift-extend | Selection doesn't modify values |
| Keyboard movement | Arrows move active cell; Tab / Shift+Tab traverse; Enter enters edit or confirms according to current mode | Keyboard always provides a visible active cell |
| Edit/commit | Double-click or type into editable cell; Enter commits validated value; Escape discards current draft cell | Read-only matrices never accept writes |
| Copy | Ctrl/Cmd+C emits tab-separated rectangular selection; single cell copies its complete value | Preserve data formatting identity on export |
| Paste | Ctrl/Cmd+V previews/parses TSV rectangular data into an editable draft | Reject mismatched shapes/types or explicitly offer safe mapping; never overwrite sources |
| Undo/redo | Undo/redo committed *draft edits* within current revision, if supported | Not a claim of historical dataset rollback |
| Sorting/filtering | Explicit commands with visible active filters and reset | Don't reorder or narrow the source dataset implicitly |
| Resize/scroll | Resize columns, scroll to selected cell, preserve row/column labels | No loss of selection on windowing |
| Import/export | CSV/TSV at minimum, with documented escaping, newline and missing-value semantics | Export full selected source vs visible/filtered rows by explicit choice |

**Copy/paste detail:** Clipboard TSV must preserve row and column boundaries;
properly handle embedded tabs, newlines and quoted values (CSV needs robust
quoting). Editing across a filtered or sorted view is ambiguous: either
resolve all coordinates back to stable underlying row/column IDs or disable
multi-cell paste with an explicit explanation until a safe mapping exists.

**Shortcuts:** Support standard keyboard behavior where browser support
permits, while allowing normal typing/search in focused toolbar fields.
Do not intercept global Ctrl/Cmd keys when focus is outside the grid.

## 6. Dataset revisions, provenance and stage staleness

An edit is first a **local draft** against an explicitly named base dataset
revision. Save/commit creates a new derived dataset revision with a stable
reference, validated schema, explicit changed cells or reproducible recipe,
and a link to its origin. Raw input is immutable by default. No automatic
saving into current training evidence and no unannounced in-place overwrite.

A committed dataset revision invalidates the current reuse claim of model and
training stages that depended on an older revision, without deleting previous
runs or re-labeling historical results. The UI should prompt the user to
choose whether the newly created revision becomes the experiment's active
input. Saving a draft must not silently submit/retrain the model.

Provenance is an ordinary inspectable detail attached to datasets and run
artifacts; it should not consume matrix-cell space. ML_Lab's existing Data Lab
fingerprints and transformation provenance remain authoritative where already
available (see S0.1).

## 7. Scale and accessibility

- Define a bounded initial viewport for potentially large datasets. Use
  page/window/virtual rendering or server-backed slices when necessary;
  rendering every cell is not a required or safe default.
- Current ML_Lab and pyPIQUE client-side table implementations must not be
  assumed to have virtualization. Determine thresholds through measured tests.
- Keep a persistent logical selection when cells scroll out of view and
  announce position, selected range, editability and validation to assistive
  technology. Keyboard-only workflows must remain usable.
- Never label an unloaded matrix tail as zero or missing; disclose the displayed
  row count and the total/unknown count accurately.
- Copy/export full data should operate from authoritative backing values, not
  only currently rendered DOM rows, except when explicitly labeled 'visible'.

## 8. Implementation and verification checklist for Sprint 2

- [ ] Read-only scientific matrix renders with minimal grid styling and
  accessible keyboard navigation in standalone ML_Lab.
- [ ] Column names, row positions, scientific-notation values, missing values
  and full-precision backing data remain distinguishable.
- [ ] Editable draft supports selection, validated commit/cancel and
  rectangular TSV copy/paste with explicit shape/type handling.
- [ ] Sorting/filtering/paging does not change underlying dataset order and
  cannot silently shift paste target coordinates.
- [ ] Saving changes produces a new dataset revision with provenance; an old
  completed run remains inspectable but is marked stale for active reuse.
- [ ] Read-only analytical values cannot be edited, even through pasted data.
- [ ] CSV/TSV exports explicitly distinguish full dataset, selected range and
  visible/filtered rows, preserving values correctly.
- [ ] Large matrices remain responsive at defined, measured scales without
  lying about omitted data or selection.
- [ ] Keyboard-only navigation, focus, labels and validation messages work.
- [ ] Same grid behavior works standalone and in optional PAH-hosted ML_Lab;
  no PAH/pyPIQUE runtime import is required.

**S0.2 acceptance (documentation patch only):** This specification makes the
visual reference, supported modes, interaction target, provenance semantics,
performance risks, and follow-up implementation gates explicit. It neither
adds a matrix editor nor claims any checklist item above has passed.
