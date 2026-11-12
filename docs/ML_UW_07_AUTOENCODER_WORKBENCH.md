# ML-UW-07 — Configurable Autoencoder Workbench

## Intent and existing ownership

The Representation tab now links to an autoencoder workbench for the stable `representation.autoencode` and `representation.transformer_autoencode` engines. No new trainer, dataset registry, or independent execution mechanism is introduced. Configurations are saved through `configure_experiment_model`, which validates typed settings and creates an immutable experiment revision. Training is launched explicitly through the existing configured experiment run queue. Saving does **not** train.

## Interface

- MLP: encoder hidden widths, activation, symmetric decoder, dropout, latent width, output activation.
- Transformer: token width, model width, attention heads, encoder/decoder depth, feedforward width, normalization, dropout, max tokens, latent width.
- Common: training epochs, batch size, learning rate, weight decay, validation fraction, early stopping, minimum improvement, random state, device, scaling, deterministic execution.
- Best-vs-last training-state selector uses the existing `restore_best` training setting. The *best* model checkpoint is persisted in the project-owned run directory whenever autoencoder training reaches a checkpoint. A last-state checkpoint is not persisted; choosing "last" affects which weights are used for reported results.
- Saved runs remain pinned to the experiment revision; a workbench rebind to the new revision occurs after a successful save.

## Safety and limitations

- The form checks dataset generation, revision, experiment ID and file SHA before writing; saved experiment config is validated by the existing typed model API. No dynamic file path or arbitrary JSON is accepted from the neural form.
- Source features must be numeric, finite, and free of missing values; training remains governed by the existing engine. No automatic imputation or hidden row sampling is introduced.
- Checkpoints are registered as project-local output artifacts. They are *not* downloaded or opened automatically: PyTorch files should only be loaded when their origin is trusted.
- Reconstruction table compares the first 12 rows of the first selected feature for verified small results (up to 500 source rows, 32 features, and 3 MiB result JSON); larger results retain the existing run results page and aggregate diagnostics where the result artifact fits.
- Training-history curves and reconstructed inputs are in-sample diagnostics, **not held-out model-validation claims**. Training jobs have existing stage-level monitoring; no epoch-level live progress or resume is introduced.
- Flask is an optional extra. Headless configuration tests run without Flask; mounted HTTP route tests require an environment with Flask installed.
- No database schema version change or additional package dependency.

## Rollback

Reverting ML-UW-07 leaves existing configured experiment revisions and run data intact. The original model runner remains authoritative; checkpoint files are ordinary run artifacts indexed by the existing artifact registry.
