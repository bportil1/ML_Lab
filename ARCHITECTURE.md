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
- generic clustering evaluation, repeated-run consensus/stability testing, and cross-algorithm agreement;
- generic representation/compression operations and reconstruction evaluation;
- model/result serialization.

A caller owns domain semantics. For example, HSQA_DBN should continue to own MI/node feature construction, layer semantics, CWE labels, lineage, cluster interpretation, and visualization. It should pass a feature matrix into ML Lab rather than moving those responsibilities here.

## Task layout

`classification/`, `regression/`, `clustering/`, `representation/`, `generative/`, and `energy_based/` are stable peers at the API/model level. Regression uses supervised regression CV/holdout semantics, clustering is not routed through supervised CV semantics, and representation returns latent/reconstruction artifacts rather than estimator-ranking results.

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


## Stable GAN boundary (0.8.0)

`ml_lab.generative.gan` owns generic adversarial generation for numeric feature tables. It supports independently configurable MLP or Transformer generator/discriminator architectures and delegates device selection, deterministic seeding, callbacks, history, and model-checkpoint state to `ml_lab.neural`. The stable trainer intentionally uses ordinary BCE-with-logits GAN training only.

Transformer architecture is stable functionality. Research-specific mechanisms from the legacy GAN Stabilization Sandbox—attention-conditioned generator/discriminator coupling, InfoNCE/shared critics, tangent-negative training, dynamic stabilization schedules, and unusual optimizer dynamics—remain outside the stable GAN API and are candidates for a future `ml_lab.experimental.gan_stabilization` module.

Stable GAN code must not import that future experimental module. Experimental GAN studies may import and reuse the stable generator/discriminator builders and training infrastructure.

## Experimental GAN stabilization (0.8.1)

`ml_lab.experimental.gan_stabilization` composes research training mechanisms around the stable `ml_lab.generative.gan` model builders; stable GAN code does not import this module. The experiment provides independently configurable self-supervised attention-similarity critic training, critic-context conditioning of latent noise, critic-embedding moment matching, discriminator-score tangent fake negatives, and an optional scheduled convex blend of MLP and Transformer discriminator branches.

The legacy GAN sandbox is treated as a source of research ideas, not as code to preserve verbatim. In particular, the similarity critic is dimension-generic rather than hard-coded to 49x16 tokens; contrastive positives are two augmented views of the same real samples rather than arbitrary generated/real pairs; and tangent negatives project random directions orthogonally to the local discriminator-score gradient. These perturbations are therefore tangent to a discriminator-score level set to first order, which is a narrower and more accurate claim than calling them guaranteed data-manifold tangents.


## Stable RBM model boundary (0.9.0)

`ml_lab.energy_based.rbm` owns generic Bernoulli, Gaussian, and Student-t Product-of-Experts RBM model definitions copied/adapted from the current user-provided HSQA_DBN implementation. HSQA_DBN is deliberately not changed in this phase; temporary duplication is an architectural safety measure until ML Lab training/parity tests are mature enough for a later dependency migration.

The RBM layer owns family configuration, construction, hidden/visible sampling, Gibbs transitions, energy evaluation, simple energy-gap diagnostics, and checkpoint round-tripping. It does **not** yet own CD-k training, DBN orchestration, HSQA experiment manifests, Firefly optimization, CWE semantics, or HSQA visualization. Those remain outside this model sprint.

The copied model code is adapted rather than preserved byte-for-byte. Public tensor inputs are normalized onto the model device/dtype, energy-gap sign is common across all families (`negative_energy - positive_energy`), and the Student-t model uses one `exp(log_sigma)` scale parametrization throughout instead of mixing exponential and softplus transforms.

`ml_lab.energy_based` and its registry/configuration modules remain PyTorch-free at import time. PyTorch is required only when an RBM is constructed, sampled, evaluated, or checkpointed.

## Stable energy-based training boundary (0.10.0)

`ml_lab.energy_based.training` owns generic single-RBM training schemes. The first stable scheme is CD-k. It is independent of HSQA experiment planning, DBN layer orchestration, CWE semantics, Firefly search, and visualization. HSQA_DBN remains untouched; code duplication is intentional until a later parity/migration phase.

The copied HSQA training design is adapted to ML Lab's model/runtime contracts rather than preserved verbatim. The key correction is gradient ownership: stable ML Lab CD-k differentiates the actual model-family energy difference `E_data - E_negative` with the negative chain detached. This prevents handwritten Gaussian/Student-t gradient formulas from drifting away from the energy functions they are intended to optimize.

Training owns batching, vectorized CD chains, persistent-chain state, optimizer state, parameter constraints, relative best-model selection, optional adaptive momentum, callbacks, history, and generic diagnostics. Persistent states are retained separately by effective batch size so a short tail batch does not destroy the full-size chain state.

Monitoring is not an optimization objective unless a caller explicitly builds such a scheme later. Epoch-wise `mean(-E(v))` is unnormalized. Normalized likelihood checkpoints use explicit partition metadata. Bernoulli models below a configurable visible-dimension threshold can enumerate the visible state space exactly; otherwise ML Lab uses proposal-based importance sampling and records proposal identity plus effective sample size. Continuous-family normalization is therefore treated as an estimate, not an exact partition function.

The built-in Sinkhorn diagnostic is an entropic-regularized balanced transport approximation and is labeled accordingly rather than being reported as exact Earth Mover distance. Monitoring sampling preserves CPU/CUDA RNG state so turning diagnostics on does not perturb subsequent CD updates.

## Experimental partitioned RBM boundary (0.10.1)

`partitioned_rbm_training` is intentionally experimental. Stable RBM models and CD-k training may be imported by the experiment; stable energy-based code must not import the experiment. The strategy keeps every sample intact, partitions only feature/hidden dimensions, warm-starts merged blocks from prior learned parameters, and leaves newly exposed cross-block weights zero (or optionally random) before merged refinement.

## Clustering stability boundary

Stable clustering keeps model selection, repeat stability, and algorithm agreement as separate concepts. Repeated fits of the selected candidate may produce co-assignment consensus matrices and ARI/NMI summaries. Cross-algorithm agreement compares final selected labelings and is reporting/analysis metadata; it is not silently folded into estimator selection. Noise label `-1` is excluded from pairwise agreement by default, and co-assignment denominators count only runs where both samples received non-noise assignments.


## Optimization wing

`ml_lab.optimization` is a stable task-neutral search layer. `ParameterSpec`, `SearchSpace`, `CandidateEvaluator`, `CandidateEvaluation`, `OptimizationResult`, and the optimizer registry do not depend on RBMs or other ML_Lab task families. The initial Firefly algorithm is adapted from the current provided HSQA_DBN implementation while leaving HSQA_DBN unchanged. ML_Lab corrects the source seed-ordering defect so a Firefly seed controls the initial population as well as movement randomness.
