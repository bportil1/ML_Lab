# ML Lab Architecture

ML Lab is a headless machine-learning execution engine. It is designed to work in three modes without changing its core behavior:

1. standalone Python package / CLI;
2. called by PAH through a thin adapter;
3. called by a domain project such as HSQA_DBN.

## Responsibility boundary

ML Lab owns generic operations:

- estimator registries;
- preprocessing pipelines;
- model/candidate selection;
- generic classification evaluation;
- generic regression evaluation and model selection;
- generic clustering evaluation and stability testing;
- generic representation/compression operations and reconstruction evaluation;
- model/result serialization.

A caller owns domain semantics. For example, HSQA_DBN should continue to own MI/node feature construction, layer semantics, CWE labels, lineage, cluster interpretation, and visualization. It should pass a feature matrix into ML Lab rather than moving those responsibilities here.

## Task layout

`classification/`, `regression/`, `clustering/`, and `representation/` are stable peers at the API level. Regression uses supervised regression CV/holdout semantics, clustering is not routed through supervised CV semantics, and representation returns latent/reconstruction artifacts rather than estimator-ranking results.

`representation/` currently provides stable PCA, configurable MLP autoencoding, and configurable Transformer autoencoding. PyTorch is optional and lazily loaded only for neural representation paths. Neural models do not own their optimizers/training loops and reuse `ml_lab.neural`.

Future task families should follow the same headless pattern, for example:

- anomaly detection;
- feature selection.

## UI policy

There is intentionally no dedicated visualization application in this repository. Results are returned as Python objects and can be serialized as JSON/CSV/joblib artifacts. PAH or another caller can supply its own visualization surface.

## Experimental isolation

`ml_lab.experimental` is an incubator, not a dependency layer. Stable task families must never import from it. Experimental modules may import stable ML Lab facilities.

Experimental discovery uses `ExperimentalManifest` records containing a module import path and entrypoint. Listing and inspecting manifests must not import the implementation. `load_experiment()` / `run_experiment()` are the explicit boundary where experimental code is imported.

This creates a promotion path:

```text
legacy/research prototype
        ↓
ml_lab.experimental
        ↓ validation + API cleanup
stable ML Lab task/model/training subsystem
```

## Flask / PAH adapter boundary

`ml_lab.flask_adapter` is optional and deliberately outside the core engine. `import ml_lab` does not import Flask. `ml_lab.flask_adapter.service.execute_task()` also works without Flask and maps JSON-shaped payloads onto the stable Python task APIs.

When Flask is installed, `create_blueprint()` returns a Blueprint rather than a complete application. The caller owns route placement and can therefore mount ML Lab directly in PAH or several levels deep inside another module:

```text
PAH
└── service/module
    └── nested service
        └── ML Lab Blueprint
```

The adapter does not own PAH sessions, authentication, background execution, project roots, or visualization. Experimental routes are opt-in and disabled by default.

## Sprint 3 built-in experiment boundary

The first built-in experiments are `linux_binary_identification` and `max_clique_rl`. Their manifests live in the lightweight catalog, but their packages are not imported by discovery.

The Linux-binary experiment keeps domain adapters (byte-to-RGB and patch tokenization). Its generic Transformer autoencoder has now graduated into stable `ml_lab.representation`; the experimental compatibility path delegates to the stable model/trainer while binary-specific adapters remain experimental.

The max-clique experiment keeps graph/clique semantics and an optional graph-specific DQN. It does not create a stable reinforcement-learning API. The legacy partitioned-RBM and optimizer paths are explicitly outside this sprint.

Optional neural research dependencies are isolated behind the `experimental-neural` installation extra. Stable classification, regression, clustering, PCA representation, preprocessing, Flask discovery, and experimental manifest discovery do not require PyTorch. Stable MLP autoencoding uses the optional `neural` extra; experimental Transformer/DQN paths continue to use `experimental-neural`.

## Shared neural training boundary (0.6.0)

`ml_lab.neural` is a stable infrastructure layer rather than a model family. It owns task-neutral neural concerns: lazy PyTorch loading, runtime/device resolution, deterministic seeding, train/validation splitting, training histories, callbacks, early stopping, best-model checkpoints, and the common trainer/result contract.

Stable model families may depend on `ml_lab.neural`; `ml_lab.neural` must not depend on representation, GAN, energy-based, PAH, HSQA_DBN, or experimental modules.

```text
representation / future GAN / future energy_based
                    ↓
              ml_lab.neural
                    ↓
       runtime + history + selection
       callbacks + checkpoints + results
```

The MLP autoencoder is the first stable consumer of this layer. Its model remains separate from its task-specific trainer, while the trainer delegates generic runtime/model-selection behavior to `ml_lab.neural`. This establishes the convention that future Transformer/GAN/RBM implementations should reuse before they are promoted into stable ML Lab APIs.


## Stable Transformer representation boundary (0.7.0)

`ml_lab.representation` owns a generic Transformer autoencoder for vector or token-sequence compression. It receives tensors only; binary/image/patch adapters remain outside the model. Two-dimensional feature matrices may be tokenized through the stable adapter, while callers with existing token sequences can supply three-dimensional arrays directly. Padding is represented explicitly and excluded from pooling/loss.

The Transformer trainer is task-specific but delegates runtime, seeding, validation splitting, callbacks, model selection, and checkpointing to `ml_lab.neural`. This is the same boundary future GAN and energy-based trainers should follow.


## Experimental Transformer VAE

`ml_lab.experimental.transformer_vae` is an incubating variational counterpart to
the stable deterministic Transformer autoencoder. It uses a diagonal-Gaussian
posterior, reparameterization, beta-weighted KL regularization, and optional KL
warm-up. Stable ML Lab modules do not depend on it.

## Experimental decomposed Transformer VAE

`ml_lab.experimental.transformer_vae_decomposed` shares the baseline experimental
Transformer VAE architecture but owns a separate beta-TCVAE-style training objective.
It decomposes the minibatch density-ratio KL into index-code mutual information, total
correlation, and dimension-wise KL. Stable representation and neural infrastructure
never import this objective. The variant may import the baseline experimental VAE
model, keeping the architecture reusable while research loss semantics remain isolated.

The density estimator is explicitly identified as `minibatch_mixture`; it is not
presented as an exact full-dataset decomposition. Singleton training batches are
rejected/dropped because an aggregated-posterior density cannot be estimated from one
sample.


## Contractive Transformer VAE experiment

The experimental incubator includes `transformer_vae_contractive`, which reuses the Transformer VAE architecture and shared neural runtime while adding a configurable first-order Jacobian penalty on the posterior mean with respect to active input elements. Stable representation code does not depend on this experiment.
