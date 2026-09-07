# Origin Notes

This initial ML_Lab base was created from the user-provided Classifier Generator repository as a design/code source.

Reused or adapted concepts include:

- declarative estimator specifications and search grids;
- sklearn Pipeline preprocessing for classification;
- training-only cross-validation followed by one holdout evaluation;
- classification metric reporting;
- JSON/CSV/joblib result artifacts;
- a thin CLI over a reusable Python engine.

Intentionally not carried forward:

- legacy Classifier Generator visualization/application code;
- legacy estimator-specific nested optimization loops;
- test-set-driven model selection;
- UI-specific state and plotting behavior.

New ML_Lab foundations added here include:

- task-neutral estimator specifications;
- first-class clustering registry/search/evaluation;
- clustering stability and optional external validation;
- public basic preprocessing operations;
- a package boundary designed for PAH and domain-specific adapters.
