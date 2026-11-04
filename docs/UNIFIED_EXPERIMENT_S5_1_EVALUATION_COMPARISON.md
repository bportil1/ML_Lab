# Sprint 5.1 — Evaluation and run comparison

## Scope

This increment introduces a **read-only** comparison of experiment-owned configured training and optimization runs. It does not introduce a new scientific metric engine, retrain models, rank heterogeneous results, update a completed run, or modify another repository.

- Python API: `ml_lab.experiments.compare_experiment_runs(store, experiment_id, run_ids=None)` returns the recorded run state, revision, model family/method, dataset SHA-256, selected columns, supported finite scalar metrics, and provenance status. `comparison_csv(report)` exports the same measurements to a conventional CSV table.
- UI: **Experiments → Model workspace → Recent runs → Compare experiment runs**, or **Run status → Compare runs**. Select up to 25 experiment-owned runs. Use **Export visible comparison (CSV)** to download the same selection.
- Training results are taken from `result.json`, in the *authoritative* recorded `result.metrics` or `results[].metrics` fields. This implementation exposes those metrics without inferring their meaning or changing their scale.
- Optimization results are taken from the registered `optimization_result.json`. The comparison displays the recorded `best_objective_terms[metric]`, the named goal, evaluation count, and best candidate settings. It does not present goal-adjusted Firefly fitness as a model-quality metric.
- Runs with no supported metric, incomplete/failed/cancelled runs, and missing, corrupted, or oversized result artifacts are shown with a reason and blank metric cells. A successful status by itself does not prove that a result artifact is usable.

## Comparability boundary

The comparison marks cohorts as **not directly comparable** when any run differs by execution type (model vs optimization), model family/method, dataset content hash, or selected feature/target columns. Different hyperparameters of the same method on exactly the same dataset can be compared, but interpretation remains the researcher's responsibility. No automatic best-run promotion or statistical significance claims are made. Different metrics may have different units and directions even inside one cohort.

## Integrity and independence

The existing project-local `ExperimentStore` owns run records and artifact identities. The artifact registry verifies SHA-256 content before the comparison reads result JSON; the maximum consumed result is 5 MiB. All file paths remain project-local and the read-only API performs no database migration beyond the existing store initialization. Cross-experiment and cross-project run selections are rejected. The CSV keeps unavailable values empty and escapes string cells that resemble spreadsheet formulas. There are no external network operations, PAH runtime requirements, or changes to the underlying scientific model implementations.

## Limitations and follow-up

This increment is a compact numeric comparison, not a fully general scientific results parser. GAN and RBM outputs that do not supply explicit `metrics` fields will show **No supported scalar evaluation metrics** rather than fabricated scores. No automatic data aggregation, metric normalization, plotting, or pairwise statistical testing is performed. Rich plots, diagnostic views, and broader exports belong to Sprint 5.2. Flask browser interaction requires testing in a local environment with ML_Lab's optional UI dependencies installed.
