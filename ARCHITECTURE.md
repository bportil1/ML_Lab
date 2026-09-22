# ML Lab Architecture

ML Lab is a reusable machine-learning execution engine whose **core remains headless**. It is designed to work in four modes without changing its core behavior:

1. standalone Python package / CLI;
2. standalone optional first-party UI;
3. mounted by PAH through thin REST/UI adapters;
4. called by a domain project such as HSQA_DBN.

## Responsibility boundary

ML Lab owns generic operations:

- unknown-data intake and structural inventory;
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

`data/`, `classification/`, `regression/`, `clustering/`, `representation/`, `generative/`, and `energy_based/` are stable peers at the API/model level. `data/` is intentionally model-agnostic and provides the intake boundary used before task-specific modeling. Regression uses supervised regression CV/holdout semantics, clustering is not routed through supervised CV semantics, and representation returns latent/reconstruction artifacts rather than estimator-ranking results.

`representation/` currently provides stable PCA, configurable MLP autoencoding, and configurable Transformer autoencoding. PyTorch is optional and lazily loaded only for neural representation paths. Neural models do not own their optimizers/training loops and reuse `ml_lab.neural`.

Future task families should follow the same headless pattern, for example:

- anomaly detection;
- feature selection.

## UI policy

The ML Lab **core remains headless**, but the repository now owns an optional first-party presentation layer in `ml_lab.ui`. The UI is not an alternative implementation of ML functionality: it calls the same host-neutral application service used by the REST adapter and ultimately the same stable task APIs.

The UI must remain optional. `import ml_lab` and `import ml_lab.ui` do not import Flask or PyTorch. A base installation therefore stays suitable for CLI, Python, batch, and embedded callers. Installing `ml-lab[ui]` enables both a standalone application (`ml-lab ui`) and a mountable Blueprint (`create_ui_blueprint()`).

PAH should mount ML Lab's Blueprint rather than reimplementing ML Lab forms/task semantics. PAH may own navigation, authentication, project/workspace context, and long-running job orchestration around the mounted module. ML Lab remains runnable when PAH does not exist.

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

## Application service and Flask / PAH adapter boundary

`ml_lab.application.execute_task()` is the host-neutral JSON-shaped application service shared by the optional REST adapter and first-party UI. It imports no Flask code. `ml_lab.flask_adapter.service` remains as a backwards-compatible re-export for existing hosts.

`ml_lab.flask_adapter` is optional and deliberately outside the core engine. `import ml_lab` does not import Flask.

When Flask is installed, `create_blueprint()` returns a Blueprint rather than a complete application. The caller owns route placement and can therefore mount ML Lab directly in PAH or several levels deep inside another module:

```text
PAH
└── service/module
    └── nested service
        └── ML Lab Blueprint
```

The REST adapter does not own PAH sessions, authentication, background execution, or project roots. Experimental routes are opt-in and disabled by default. The optional `ml_lab.ui` Blueprint owns ML Lab's presentation surface while still leaving host-level navigation/authentication/workspace concerns to PAH.

## Optional first-party UI boundary (0.13.0)

`ml_lab.ui` provides two entry points over the same UI implementation:

```text
ml-lab ui
    ↓
standalone Flask app

PAH / another Flask host
    ↓
create_ui_blueprint()
```

The UI began as a thin capability-aware shell with a generic JSON workbench over `ml_lab.application.execute_task()`. Data Lab A1/A2/A2.1/ML-2 provide the first typed workspace while still invoking the same host-neutral inventory/profile/table/transformation contracts. Future typed workspaces should follow that pattern: core capability first, application contract second, CLI/REST/UI presentation last. HTML forms are never the source of ML semantics.

The UI has no CDN, external asset, or network-service requirement. Its static assets are packaged with ML Lab. Experimental UI is hidden unless explicitly enabled.


## Data Lab A1 boundary (0.14.0)

`ml_lab.data` is a stable, model-neutral intake subsystem. A1 performs read-only discovery and structural inventory of CSV/TSV sources. It does not clean, coerce, normalize, impute, or otherwise mutate data.

```text
file / directory paths
        ↓
recursive discovery
        ↓
format handler (CSV / TSV in A1)
        ↓
encoding + delimiter + header inference
shape + malformed-width detection
likely exported-index detection
source hash
        ↓
ml-lab.data-inventory@1
```

Unsupported files are inventoried as unsupported instead of disappearing. The format boundary is intentionally extensible so later JSON/JSONL, Parquet, Excel, and other handlers can emit the same inventory records. A2 statistical profiling is layered on this inventory boundary; cleaning/transformation belongs to A3; broader provenance semantics remain an A5 concern.

The operation is exposed consistently as `data.inspect` through the application service, `ml-lab data inspect` through CLI, the optional Data Lab UI, and the generic Flask adapter. A host such as PAH may register the returned inventory as an artifact, but ML Lab does not depend on PAH to create or inspect it.


## Data Lab A2 boundary (0.15.0)

A2 adds a second stable read-only artifact: `ml-lab.data-profile-collection@1`, containing per-file `ml-lab.data-profile@1` records. The profiler never silently upgrades statistical dependence into causal semantics. It records inferred types and univariate data-quality signals, then computes bounded pairwise dependence summaries only for eligible non-constant, non-identifier columns.

```text
ml-lab.data-inventory@1
        ↓
rectangular valid rows
        ↓
optional deterministic reservoir sample
        ↓
column inference + missingness + cardinality
duplicates + descriptive summaries + IQR outliers
        ↓
bounded pairwise relationships
  numeric ↔ numeric: Pearson + Spearman
  eligible pairs: discretized MI + normalized MI
        ↓
ml-lab.data-profile@1
        ↓
ml-lab.data-profile-collection@1
```

Profiling limits are part of the contract rather than hidden implementation details. `profiled_row_count`, `source_row_count`, `sampled`, `sample_strategy`, and `relationship_row_count` make it explicit whether a statistic was computed over all valid rows or a deterministic subset. Large/wide datasets therefore remain safe to inspect without pretending sampled metrics describe an unbounded full population.

`data.profile` is exposed through the Python API, application/REST service, `ml-lab data profile`, and the same mountable Data Lab UI. Hosts such as PAH may register these artifacts, but ML Lab remains independently runnable and does not import PAH.


## Data Lab A2.1 interaction boundary (0.16.0)

A2.1 is a presentation/artifact sprint, not a change to A2 statistical meaning. The core profiler remains synchronous and read-only. The optional UI wraps profile calls in a small in-process job manager so standalone ML Lab and a PAH-mounted Blueprint can expose explicit `starting`, `running`, `completed`, and `failed` states without requiring Redis, Celery, or another external service.

UI-triggered runs persist locally as:

```text
ml_lab_results/data/profile_runs/<run-id>/
├── manifest.json
├── profile_collection.json
└── profiles/
    └── <dataset-profile>.json
```

These artifacts can be reopened after returning to the Data Lab page. Profile viewers expose quality signals, columns, and relationships through local interactive tables with search, per-column filters, sorting, highlighting, reset controls, and user-selected pagination.

Source browsing is backed by the stable `data.table` application/core contract. The browser defaults to 50 rows per page and offers 25/50/100/250/500/All. Pagination is only a rendering choice: there is no hidden first-N-row cap. Search and per-column filters traverse the full valid source, sorting operates across all matching rows, and users may explicitly request all rows. For ordinary unsorted pages the implementation streams through the delimited source rather than materializing the entire table in the browser.

The source UI uses short-lived signed local path tokens generated by the mounted Blueprint so arbitrary host file paths are not exposed as unauthenticated route parameters. PAH continues to own any outer authentication/access policy.

## ML-1 real-data checkpoint (0.16.1)

ML-1 intentionally adds no cleaning or transformation semantics. It validates the existing A1/A2/A2.1 contracts against real project data before the controlled-transformation layer is designed. Nine local CSV datasets were profiled over their complete valid-row populations (`max_rows=0`) without modifying the sources. The detailed checkpoint is stored at `docs/checkpoints/ML-1-real-data-checkpoint.md`.

The checkpoint confirmed the current intake/profile/table boundaries and exposed requirements that must remain explicit in ML-2: user-declared type overrides when value domains are ambiguous (for example numeric 0/1 fields inferred as boolean), non-destructive handling of all-missing/constant columns, configurable missing/sentinel treatment, opt-in outlier handling, string/path normalization, and derived-column parsing for mixed experiment metadata. Identifier inference remains advisory and must never become an automatic deletion rule.

The checkpoint also reinforces the transformation boundary:

```text
raw source
    ↓
read-only inventory/profile
    ↓
explicit transformation recipe
    ↓
preview + diagnostics
    ↓
user-approved derived dataset
```

Raw source files remain immutable by default. Every later transformation must be representable as a persistent recipe with observable row/column/value changes and coercion failures.

## ML-2 controlled transformation boundary (0.17.0)

ML-2 adds source-changing behavior only through a separate derived-data contract. Inventory, profiling, and source browsing remain read-only. `ml_lab.data.transform` owns ordered recipe execution; the application service, CLI, and typed UI all delegate to it.

```text
raw CSV/TSV + source SHA-256
        ↓
ml-lab.transformation-recipe@1
        ↓
preview_transformation()
        ↓
operation diagnostics + bounded display preview
        ↓ explicit apply
new derived CSV/TSV
 + recipe JSON
 + ml-lab.derived-dataset@1 manifest
```

The source and output path may never resolve to the same file. Existing derived output artifacts are replaced only when the caller explicitly opts into overwrite. Writes use a temporary file followed by an atomic replace. Malformed input rows are rejected by default; skipping them must be stated by the recipe. These rules prevent a cleaning action from silently turning a read-only profiling workflow into destructive mutation.

Each operation reports rows/columns before and after, columns added/dropped, missing-cell counts, coercion failures, and changed cells when the before/after shapes are comparable. The derived manifest records both source and derived SHA-256 values. This is intentionally enough lineage for ML-2 artifacts to be auditable, but the generalized lineage graph and cross-dataset provenance model remain ML-4 work.

The stable recipe vocabulary covers column selection/removal/reordering/renaming, quality-based constant/all-missing removal, type coercion, missing/sentinel handling, duplicate and row filtering, string cleanup, constrained derived columns (no arbitrary Python evaluation), categorical encoding, scaling, opt-in IQR treatment, joins, melt, and pivot. The UI provides typed controls for the common single-source cleanup path; advanced operations still execute through the exact same recipe engine via Python/application/CLI callers.

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

## ML-3 dataset comparison boundary

Dataset discovery is implemented in `ml_lab.data.comparison` and is callable from Python, the application task `data.compare`, CLI, and the optional Data Lab UI. Comparisons use file fingerprints, ordered/unordered schema comparisons, multiset row overlap, likely identifier overlap, and conservative filename/schema heuristics. Inferred labels such as train/test or version/derivative are descriptive hypotheses only. ML-4 remains responsible for formal lineage and provenance.
