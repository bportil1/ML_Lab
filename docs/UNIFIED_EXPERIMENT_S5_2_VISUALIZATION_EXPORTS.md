# Sprint 5.2 — Scientific visualizations and exports

## Scope

This read-only extension of the S5.1 comparison workspace plots measurements from
registered, SHA-256-verified project-local artifacts. The scientific engines,
run status, model configuration, underlying datasets, and artifact catalog are not
modified. It is available in standalone ML_Lab without PAH.

### Supported graph types

- **Comparable run metrics:** Neutral bar plots of a named finite scalar metric,
  only when the S5.1 comparison reports the selected runs as belonging to the
  same model family/method, dataset, selected features/targets, and execution type.
  A metric is plotted only when at least two model-training runs report it.
  Mixed cohorts and optimization outcomes never receive comparative bars;
  different optimization search plans/budgets need not be comparable.
- **Recorded epoch histories:** Native `history` arrays in saved model results,
  when they include actual numeric `epoch` or `iteration` coordinates and at least
  two finite measurements. This supports GAN, RBM, and neural representation
  histories where recorded. PCA and models without histories do not acquire
  fabricated training curves.
- **Firefly objective traces:** Uses the registered `optimization_history.csv`,
  plotting its observed `objective.<metric>` values in recorded candidate
  evaluation order. Failed and nonnumeric candidate values are omitted; fitness
  (which may reverse sign for maximization) is never passed off as the observed
  objective.

A maximum of 25 run selections, 24 charts, 500 displayed points per chart,
and 10,000 history entries are supported. Display downsampling retains the
first and last value; charts and CSV contain the same displayed points. Users
can still inspect the source result and optimization artifacts for full-fidelity
histories.

### Presentation and exports

- **SVG:** Server-generated, static, neutral graphs with numeric axes. No
  external plotting scripts, remote APIs, or embedded JavaScript.
- **CSV:** A conventional `x,label,y` table for exactly the displayed points.
  Text cells are spreadsheet-formula escaped.
- **JSON:** A portable read-only visualization report with run selection,
  plot data, source artifact IDs, and SHA-256 hashes. It is generated on demand;
  no new evaluation metrics or chart artifacts are written to the project.

Results are plotted only from hash-verified output artifacts within the owning
project. Existing comparison and result endpoints remain unchanged. The
experiment revision, dataset and method selection are those pinned at run time.
No metric normalization, winner selection, learned extrapolation, or forced
model training is performed by the visualization layer.

### Limitations and follow-up

- In-progress progress records are stage-only and not epoch/step training curves.
- Charts are bounded in size and omit unknown/non-finite metric fields.
- This sprint does not create scatterplots of latent coordinates, sample images,
  or model architectures: those may require separate validated artifact and
  dimensionality contracts.
- Flask-dependent browser routes require an environment with `ml-lab[ui]` for
  end-to-end testing. The chart contract and SVG/CSV rendering remain pure Python.
