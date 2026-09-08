# ML_Lab 0.1.0 Baseline Report

This baseline was constructed only from the user-provided Classifier Generator repository and local implementation/testing. No web search, connected application, or external repository was used.

## Included capabilities

- Headless Python API and CLI.
- Shared task-neutral estimator specification.
- Classification engine adapted from the cleaned Classifier Generator architecture.
- 20 registered classifier families.
- First-class clustering engine with 8 registered clustering families.
- Clustering internal metrics, repeated-run stability, and optional external-label evaluation.
- Shared scaling hints and reusable preprocessing helpers.
- Direct basic operations for scaling, imputation, and PCA.
- JSON/CSV/joblib result artifacts.
- Candidate-level search reports for classification and clustering.
- No legacy Classifier Generator visualization/application code.

## Registered clustering families

- Agglomerative Clustering
- BIRCH
- DBSCAN
- Gaussian Mixture
- K-Means
- MiniBatch K-Means
- OPTICS
- Spectral Clustering

## Verification

Repository tests:

```text
10 passed
```

Additional smoke verification:

- all 28 registered estimators instantiate;
- all 8 clustering families completed a one-candidate Iris run;
- classification CLI completed an LDA Iris workflow;
- clustering CLI completed a K-Means Iris workflow;
- classification and clustering output artifacts were generated successfully;
- the package built successfully as `ml_lab-0.1.0-py3-none-any.whl`.

## Intended next architectural steps

The package is ready to expand with task families such as regression, anomaly detection, dimensionality reduction, and feature selection. PAH and HSQA_DBN integrations should be thin adapters over this package rather than dependencies inside ML_Lab.


## Sprint 4 representation baseline

ML Lab 0.5.0 adds stable representation/compression support: PCA with reconstruction/provenance and a configurable MLP autoencoder with separate model/training contracts. PyTorch remains optional behind the `neural` extra. Representation is callable from Python, CLI, and the Flask-independent PAH service adapter.

## Sprint 5 neural infrastructure baseline

ML Lab 0.6.0 adds a PyTorch-lazy `ml_lab.neural` infrastructure package and refactors the stable MLP autoencoder to use it. Shared facilities cover device selection, deterministic seeding, validation splitting, history, early stopping, callbacks, best-model checkpointing/reload, and common trainer/result contracts. This sprint does not add a new neural architecture; it establishes the reusable training/runtime layer required by the upcoming Transformer autoencoder, GAN, and energy-based model work.


## Sprint 6 Transformer representation baseline

ML Lab 0.7.0 promotes a corrected generic Transformer autoencoder into stable `ml_lab.representation`. It supports 2-D feature matrices and 3-D pre-tokenized inputs, padding-aware vector tokenization, attention-pooled latent compression, learned decoder queries, shared neural training/checkpoint infrastructure, CLI/Flask-service execution, and headless latent/reconstruction artifacts. The Linux-binary experimental Transformer path now delegates to this stable implementation rather than maintaining a second training/model implementation.
