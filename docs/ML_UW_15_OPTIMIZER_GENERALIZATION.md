# ML-UW-15 — Optimizer generalization (internal)

## Scope and architecture

ML-UW-15A adds an independent, seedable **Random Search** baseline alongside the preserved **Firefly** optimizer. Both consume the same task-agnostic `SearchSpace` and `CandidateEvaluator`, output a shared `OptimizationResult`, and use a common `optimization_history.csv` and `optimization_result.json` artifact contract. The Firefly-specific history files remain unchanged; Random Search emits its own `random_search_history.json` and `random_search_evaluations.json`.

ML-UW-15B extends experiment-plan validation to **1–4 jointly searched numeric controls**, with integer/continuous type checking, optional linear/logarithmic scales, and safe model-level validation **after all candidate settings are applied**. Non-search settings come from the immutable model revision. Search-space logarithmic coordinates affect Firefly *only* for explicitly log-scaled dimensions; original linear movement and output files retain their prior contract. Random Search independently samples in the parameter's declared scale. The generic optimizer API can already optimize arbitrary callable objectives, including categorical and fixed parameters; the experiment adapter does **not** expose arbitrary callables or category fields.

## Usage and budget

On a saved model configuration choose **Optimize saved configuration**, then select `Firefly` or `Random Search`, one required numeric control and up to three additional controls. Set lower/upper for each and optionally `Logarithmic` for positive values. A metric/goal is chosen from the **existing registered scientific objectives**. The search remains serial and is explicitly launched; it **does not train on page load**, **change the experiment model**, **write to the original data**, or **promote the best candidate**.

Experiment runs are capped at **32 attempted candidate evaluations**. Random Search uses `evaluations` directly. Firefly evaluates `population_size * (iterations + 1)` candidates (initial population included). The experiment runner allows Firefly population 2–8, iterations 0–5, and Random Search 1–32 evaluations. The generic optimization API has its own higher limit (100,000 random draws), independent from the bounded UI runner. Large search spaces can yield repeated integer/category combinations; evaluation calls count even when parameters repeat. **No candidate-result caching or cross-run reuse is claimed**.

Experiment plan shape (programmatic callers):

```json
{
  "algorithm": "random_search", "metric": "accuracy", "goal": "maximize", "seed": 11,
  "evaluations": 16,
  "parameters": [
    {"path": "config.cv_folds", "lower": 2, "upper": 4, "scale": "linear"},
    {"path": "task_options.test_size", "lower": 0.15, "upper": 0.30, "scale": "linear"}
  ]
}
```

One-control legacy plans (`parameter`, `lower`, `upper`, `population_size`, `iterations`, `seed`) remain accepted. Recorded one-dimensional results retain the old flat fields and the shared `schema` and executor identifiers. Raw model seed policies are not changed: fixed optimizer seed means reproducible *candidate proposals*, **not** necessarily bitwise-reproducible neural training.

## Scientific and operational boundaries

- Experiment objectives remain limited to registered numerical **representation, classification, regression, clustering** metrics. RBM/GAN adapters are **not** silently enabled; their objective definitions and comparable validation protocols require separate review.
- Both algorithms use the same observed metric and goal direction, recorded separately from goal-adjusted scalar fitness. Compare at a matched evaluation budget, source dataset/split, model revision and repeated search seeds.
- Experiment parameters are whitelisted by typed model registry. Random state/seed fields, filesystem paths, checkpoint controls, unsupported model suites, malformed bounds, and duplicate dimensions are rejected. Model-specific invalid combinations may still fail at a particular candidate; those failures are logged. A run fails if no candidate succeeds.
- Training and validation may occur on the same saved data split and **are not** automatically a held-out generalization benchmark.
- Cancellation is cooperative at evaluation boundaries; ongoing synchronous model calls cannot be forcibly terminated midway.
- No external service, new scientific model engine, new package dependency, project database migration, cross-project state, or process-level file transfer is introduced.

## Regression and release gates

Tests for the optimizer core, legacy Firefly reproducibility and output filenames, seeded Random Search, logarithmic movement, cancellation, candidate failures, single- and multi-parameter experiment runs, revision pinning, artifact verification and validation constraints are in `tests/test_ml_uw_15*.py` plus existing tests. Flask UI tests should be rerun in the actual installation; optional Flask is absent from the isolated offline validation environment. Complete PAH-mounted live acceptance and broader performance benchmarking remain separate gates.

## Rollback

Revert ML-UW-15B before ML-UW-15A if rolling back; do not remove project SQLite files or run artifacts. Existing completed runs retain their algorithm-specific result artifacts. The code uses the previous schema version and does not migrate SQLite.
