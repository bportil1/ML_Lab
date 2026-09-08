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


## Sprint 1 regression expansion

The regression family list was adapted from the user-provided legacy experimental repositories, especially the regression/model-mapping work in the GAN Stabilization Sandbox. The old hand-written loops and unusual metric implementations were not copied. ML_Lab re-expresses those estimator families through its declarative registry, sklearn Pipeline/GridSearchCV selection, holdout evaluation, and task-neutral artifact contracts.

## Sprint 2 experimental and service boundaries

Sprint 2 adds an experimental incubator as new ML_Lab infrastructure rather than copying unfinished legacy algorithms into the stable engine. Experimental discovery is manifest-based and lazy so research dependencies do not leak into normal ML_Lab imports.

Sprint 2 also adds an optional Flask Blueprint adapter for PAH/nested-service integration. Flask remains an optional extra and the adapter delegates directly to the existing task APIs; no legacy visualization/application layer is introduced.

## Sprint 3 legacy experimental imports

The `linux_binary_identification` experiment was adapted from the user-provided `Linux_Binary_Identification` repository. Reused concepts include byte-stream-to-RGB representation, patch tokenization, attention-pooled Transformer encoding, and latent compression. The original visualization code, file-coupled model training, legacy clustering, and model-owned optimizer pattern were intentionally not copied.

The `max_clique_rl` experiment was adapted from the user-provided `Max_Clique_ML_Solver` repository. Reused concepts include adjacency-matrix clique state, clique-valid actions, planted-clique graph generation, and the prototype DQN search. The old partitioned-RBM and optimizer implementations were intentionally excluded for later redesign.


## Binary Image Analysis Project

ML_Lab's `binary_fractal_conversion` experiment preserves the legacy project's byte-driven Mandelbrot and IFS representation ideas in a cleaned, headless form. Plotting, RBM, optimizer, and classifier code were intentionally not imported.


## Sprint 6 stable Transformer autoencoder

The stable Transformer autoencoder was promoted from ideas in the user-provided `Linux_Binary_Identification` repository and its cleaned ML_Lab experimental successor. The legacy implementation was not copied verbatim. The stable rewrite removes file/image loading from the model, removes model-owned optimizer/training overrides, drops unrelated VAE/TC loss machinery, fixes decoder memory/query semantics, adds explicit padding-aware reconstruction, and reuses `ml_lab.neural` for training/runtime behavior. The binary-specific conversion pipeline remains under `ml_lab.experimental`.


- The experimental Transformer VAE preserves the variational latent-model direction
  seen in older research code, but was rebuilt around a standard diagonal-Gaussian
  posterior instead of carrying forward mixed ICMI/TC/DWKL objectives by default.

- The decomposed Transformer VAE revisits the ICMI/TC/DWKL research direction from
  the supplied legacy neural experiments, but does not copy their ad-hoc RBM/Bernoulli
  approximations. It uses a diagonal-Gaussian beta-TCVAE-style minibatch density
  decomposition so ICMI, TC, and DWKL have explicit probabilistic definitions and sum
  to the same minibatch density-ratio KL estimate.


- The contractive Transformer VAE is a clean research implementation of first-order Jacobian regularization inspired by contractive/manifold-tangent ideas; it does not copy legacy training code.

## Stable GAN wing

The stable GAN architecture was informed by the user-provided `GAN_Stabilization_Sandbox`, especially its use of Transformer generator/discriminator layers. The legacy implementation was not copied verbatim: image/MNIST assumptions, import-time application behavior, hard-coded 28x28 reshaping, attention-conditioned training, InfoNCE/shared-critic coupling, tangent-negative generation, transformer-strength scheduling, and legacy optimizer experiments were intentionally excluded. ML Lab keeps ordinary MLP/Transformer GAN architecture in the stable API while reserving those research-specific training mechanisms for a future experimental stabilization module.

## Experimental GAN stabilization

The `gan_stabilization` experiment is adapted from ideas in the user-provided `GAN_Stabilization_Sandbox`. ML Lab does not copy its hard-coded MNIST shapes, inverted CUDA logic, unconditional discriminator noise, fixed 49-token similarity critic, or legacy optimizer code. The experimental rewrite reuses the stable GAN model builders and isolates research mechanisms behind explicit configuration switches.


## Stable RBM model wing

The stable Bernoulli, Gaussian, and Student-t Product-of-Experts RBMs were copied/adapted from the current user-provided `HSQA_DBN-refactor(2)` repository. That HSQA_DBN repository remains untouched and continues to own its working copies. ML Lab intentionally duplicates the generic model layer for now so extraction can be validated independently before any later deduplication.

The adaptation preserves the active family controls and sampling concepts while cleaning library-level behavior: model inputs are device/dtype safe, family names/settings have a standalone registry/config contract, checkpoints carry complete model provenance, the energy-gap sign is standardized across families, and the Student-t scale parameter uses a single exponential log-scale transform throughout. CD-k, DBN orchestration, HSQA-specific diagnostics/experiments, and Firefly optimization are not part of this copy step.

## Stable CD-k training wing

The CD-k architecture was copied/adapted from the current user-provided HSQA_DBN training stack (`training/base.py`, `cdk.py`, phase/control/model-selection helpers, and training diagnostics). HSQA_DBN was not modified. ML Lab intentionally owns an independent copy while the generic API is validated.

Several implementation choices were changed during extraction instead of copying legacy behavior. Family-specific handwritten CD gradients were replaced by autograd of the model's own energy difference with detached negative samples; persistent chains are retained per effective batch size; diagnostic RNG use is isolated from the training trajectory; normalized likelihood uses exact Bernoulli enumeration when feasible and explicit importance sampling otherwise; and Student-t defaults initialize `nu` inside its valid range so clamping does not zero its initial shape gradient. The historical optimizer/Firefly code was not moved in this sprint.

## Partitioned RBM provenance

The partitioned-RBM idea was recovered from the uploaded Max Clique ML Solver and Binary Image Analysis Project. Their implementations were treated as research prototypes only. ML_Lab preserves the local-block→merge→global-refinement concept while correcting sample-axis slicing, remainder loss, visible/hidden geometry assumptions, direct parameter mutation/normalization, and ambiguous per-loop epoch decay.


## Generic optimization / Firefly

The stable `ml_lab.optimization` Firefly implementation is adapted from `optimization/firefly.py`, `optimization/base.py`, `optimization/evaluator.py`, `optimization/result.py`, `optimization/space.py`, `optimization/fitness.py`, and `optimization/registry.py` in the user-provided current HSQA_DBN archive. The HSQA_DBN source repository is not changed. HSQA-specific RBM objective/evaluator code is not copied into the generic optimizer layer.
