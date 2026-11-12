# ML-UW-06 — Representation workbench (bounded preview)

## Scope

The **Representation** section now opens a functional, non-persistent
scikit-learn representation explorer. Input is the *verified active dataset*
selected in Data, limited to the explicitly selected feature columns. The
feature roles edited in Feature Profiling remain **display-only**, never an
implicit preprocessing or modeling instruction.

- Existing ML_Lab `run_pca` is reused for PCA. No second PCA implementation.
- Other estimators use scikit-learn: TruncatedSVD, FastICA, NMF, RBF KernelPCA,
  Isomap, and t-SNE. The headless `method_catalog()` declares whether the
  estimator supports projecting additional rows and inverse-transforming.
- Reconstruction is **opt-in** and is rejected by the execution function for
  unsupported methods (Isomap and t-SNE), even if a caller bypasses the UI.
  RBF KernelPCA's fitted inverse is approximate. Reconstruction scores are
  **in-sample**, not predictive generalization or model-comparison scores.
- T-SNE's fit-only embedding is *not* a reusable transformer. It has neither
  an inverse nor an out-of-sample transform. The rendering is an exploratory
  coordinate view, not a trained model artifact.
- Scaling is explicit: none, standard, minmax, or robust. NMF only accepts
  none/minmax, with an actual nonnegativity check.
- Results show embedding shape, estimator capabilities, first 12 coordinate
  rows, and a capped 240-point, deterministic 2D SVG scatter. No full input
  matrix is returned by a view endpoint.

## Input and safeguards

- A workbench instance must point to a valid, fingerprint-verified local
  CSV/TSV selection, with **no target columns**. Existing experiment and
  data validation are reused; the input is verified again after fitting.
- No silent imputation, categorical encoding, downsampling of *training*
  rows, or auto-exclusion of columns. Non-numeric, non-finite or missing input
  values are rejected. Users must prepare and reselect the intended dataset
  explicitly in Data before retrying.
- General limit: 5,000 rows, 64 selected features and 32 MiB source files.
  KernelPCA, Isomap and t-SNE limit rows to 400; t-SNE limits output to 3
  dimensions in Barnes-Hut mode. These are **preview resource limits**, not
  limits on the existing ML_Lab experiment or task engine.
- POST forms include an optimistic workbench generation and source SHA-256;
  the backend checks both against the active instance. Re-verification after
  fitting suppresses stale results when the source or pinned revision changes.
- No database migration, persistence, new dependency, external service, or
  change to source datasets, artifacts, training lifecycle or autoencoders.
  The original standalone PCA / neural task remains linked from this section.

## Limitations and follow-up

The interface is a **stateless preview**: estimator support for projecting
new observations does not imply that this view exposes a cross-request fitted
model. The future ML-UW-07 increment can integrate configurable autoencoder
experiments, and ML-UW-08 can handle matched evaluation and comparisons. This
patch does not silently save or publish a fitted estimator.

## Manual acceptance

1. Select a project-local numeric CSV/TSV without targets in Data.
2. Open Representation, fit PCA with 2 dimensions, and confirm the scatter,
   summary and optional reconstruction metrics. Existing `run_pca` results
   should match for the same input and scaling.
3. Switch to t-SNE or Isomap. Confirm inverse/reconstruction requests are
   rejected with an explicit explanation; verify t-SNE has no transform-new-
   data capability. For NMF, verify standard scaling is rejected and minmax
   works on data with negative original values.
4. Switch to 1 latent dimension; confirm no fictitious scatter appears.
5. Modify source bytes or change the selected experiment in another tab and
   submit the old form; confirm the stale request cannot produce a result.
6. Check the previous PCA / neural task and workbench sections still open.
