# ML-UW-08 — Representation Comparison Workbench

## Scope and ownership

The unified **Comparison** tab now offers a bounded, read-only representation comparison. The existing PCA and sklearn implementation in `representation.sklearn_catalog.explore_representation` is the sole fresh-fit path, and saved MLP/Transformer outputs come exclusively from the S4 experiment run/artifact registry. This is *not* a new model trainer, persistence service, dataset registry, or PAH-level cross-project database.

- Compare 2–5 unique sklearn methods, no implicit row sampling or imputation, with 8–400 source rows, up to 64 selected features, up to 8 latent dimensions (subject to each estimator's stricter limits), and files up to 32 MiB.
- Require identical input rows, column order, latent width, and feature scaling for fresh fits. Geometry metrics (sklearn trustworthiness, pairwise-distance Spearman) use the same scaled input-space reference; t-SNE/Isomap have no reconstructed-input score.
- Optionally inspect up to 4 *previously trained*, verified MLP/Transformer run outputs from the active experiment. Accept only exact dataset path/hash, feature order, row count, latent width, and training scaling. Match rows and columns before recomputing reconstruction errors directly from stored reconstructed arrays. Verified artifacts must pass registry integrity checks. Never import or load model-state PyTorch checkpoints from the browser.
- Export the displayed comparison numerics as CSV, a static SVG image, or an escaped LaTeX table. Export reruns the same bounded comparison to guarantee the result matches its submitted form; it does not use a server-held hidden report.
- Keep existing dataset comparison and model-run comparison accessible from the Comparison tab.

## Methodological safeguards

These numbers are **in-sample descriptive diagnostics**, not an independent or matched held-out benchmark. Saved neural models may have used an internal validation split and early stopping, whereas sklearn methods are fitted afresh on the full selected input. Both neural full-dataset outputs and sklearn reconstructed arrays contain observations used during fitting. Even when inputs/scaling/components match, differences in optimization budgets and fit histories remain. Do not rank methods as generalization winners from these metrics.

- Input-space MSE, RMSE, MAE are recomputed from reconstructed values in **original feature units**; absent for methods with no meaningful inverse. The comparison does not replace per-run training-space or validation metrics.
- Trustworthiness measures local neighborhood retention, and rank-Spearman compares pairwise input distances with latent distances. The latter is unavailable when one space has constant pairwise distances. All metrics are based on identical source observations; they are not uncertainty estimates.
- Each scatter uses its own axes; the SVG image export is of the numeric summary table, not a common-scale overlay. A relative trustworthiness bar chart is only an illustration of the numeric values.
- Unsupported method/scaling combinations and failed fits are reported explicitly, not fabricated as zero scores. Results and errors are bounded; no generated model is persisted.
- Every form checks workbench generation, experiment revision, data hash and selected targets. The route revalidates the instance and selected dataset after the comparison and discards stale results. Only the owning project-local SQLite store is read for run metadata.

## Verification and limitations

Headless unit tests cover geometry, unsupported inverses, configuration validation, source-matched stored runs, fingerprint tampering, and exports. Flask UI route tests are conditional on the optional Flask dependency. This increment does not add neural retraining, shared trial splits, model normalization, off-sample reconstruction quality, an estimator cache, cross-project artifact access, or further database migrations. Future work can create a scientifically controlled held-out cohort and delegate common export controls to PAH_Shared.
