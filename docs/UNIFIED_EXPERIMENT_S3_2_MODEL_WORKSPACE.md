# Sprint 3.2 — Unified model workspace

This sprint **exposes the stable Sprint 3.1 model registry** as a single,
standalone-capable configuration UI. It does not alter the model engines or
introduce training/optimization execution.

## Workflow

1. In Data Lab, inspect a project-local CSV/TSV dataset, use **Use in
   experiment**, and select feature/target columns (Sprint 2.3).
2. Open **Experiments** in ML_Lab's top navigation, or follow **Configure
   model** after dataset selection.
3. Choose the selected experiment, model family and method. Supported stable
   families are classification, regression, clustering, representation, RBM,
   and GAN; the method list derives from the existing scientific registries.
4. Expand typed configuration groups. Enter only overrides; blank controls
   defer to authoritative model engine defaults. Advanced model-specific
   nested options are collapsible. Multi-estimator selection is available for
   classification, regression and clustering.
5. Click **Save model configuration**. The UI uses `expected_revision` to
   publish an immutable experiment model revision. No task is executed.

The UI lives at `GET/POST /experiments/models` inside ML_Lab's existing
mountable Blueprint. PAH can mount the same Blueprint, passing the existing
`experiment_project_root`; no PAH-owned database or service is required.
Original `/task/<capability_id>` JSON workbenches remain available during the
migration, and all CLI/headless APIs remain unchanged.

## Input/output and safety

- Input controls are generated from `describe_model_method()` and parsed by
  `ml_lab.ui.model_forms`. The form advertises fields and types from the model
  registry, not a second hard-coded model configuration schema.
- Empty fields are omitted, preserving scientific defaults. Boolean fields
  use explicit true/false selects, numeric fields are typed, arrays accept
  comma-separated values or JSON arrays, and nested options have natural
  grouped controls. Only PCA's union-typed scalar asks for JSON notation.
- The registry's **task options** are presented together but written as
  top-level settings (`test_size`, `hidden_dim`, `estimators`), matching the
  existing legacy payload shape. Model and training dataclasses remain
  nested in `config`, `model_config`, and `training_config` as appropriate.
- Model checkpoint paths are intentionally **not** editable: Sprint 4 will
  allocate output paths inside the experiment's project. The UI must not
  accept unadvertised field paths as hidden controls.
- The server revalidates all typed settings, model-target compatibility,
  source content and Data Lab provenance; it rejects stale revisions instead
  of overwriting concurrent work. No raw model payload or arbitrary output
  paths are accepted from the form.
- The unified workflow chrome remains PAH-compatible. Scientific matrix
  tables and the existing Sprint 2.2 matrix editor remain neutral and
  unchanged. The model workspace does not import pyPIQUE or HSQA_DBN.

## Known limitations / scheduled work

- The model workspace **configures but does not run** training. Training job
  lifecycle, optimization and checkpoint allocation remain Sprint 4.
- It does not automatically migrate historical legacy task runs into
  dataset-selected experiments. Data Lab is the authoritative dataset picker.
- Full numeric dataset validation is performed when preparing a model handoff;
  the UI provides feature/target and dataset-integrity checks before saving.
- The UI lists the latest 100 experiment selections. Additional experiment
  history/navigation capabilities are deferred to the broader workflow UI.
- Browser integration tests require the optional Flask UI dependency;
  parser and Jinja template tests can run without Flask or PAH.

## Local checks

```bash
PYTHONPATH=src python -m pytest -q tests/test_experiment_model_form.py \
    tests/test_experiment_models.py tests/test_experiment_model_ui.py
PYTHONPATH=src python -m compileall -q src/ml_lab/ui
```

**Next:** Sprint 4.1 — training lifecycle/monitoring, following the roadmap's
model-workspace milestone; no training is coupled into Sprint 3.2.
