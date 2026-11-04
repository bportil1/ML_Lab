# Unified Experiment — Sprint 3.1 Model Registry and Configuration

**Status:** Opt-in headless API. This patch does not change the existing
`ml_lab.application.execute_task`, command-line commands, Flask JSON workbenches,
PAH integration, model implementations, or training behavior. The visible model
selection workflow is Sprint 3.2.

## Scope and ownership

The registry lives in `ml_lab.experiments.models` and describes the **stable,
currently runnable** scientific task families without copying model factories,
reimplementing estimator grids, or importing torch. The authoritative sklearn
estimators come from the existing classification, regression, and clustering
registries. PCA/autoencoder, GAN, and RBM controls use their existing scientific
configuration classes and RBM family registry.

| Model family | Selection methods | Required selected targets | Underlying legacy task |
|---|---|---|---|
| Classification | Existing estimator IDs; explicit multi-estimator `suite` | Exactly one | `classification` |
| Regression | Existing estimator IDs; explicit multi-estimator `suite` | Exactly one | `regression` |
| Clustering | Existing estimator IDs; explicit multi-estimator `suite` | Zero or one optional reference label | `clustering` |
| Representation | `pca`, `mlp_autoencoder`, `transformer_autoencoder` | None | `representation` |
| Energy-based | `bernoulli`, `gaussian`, `student_t_poe` RBMs | None | `rbm` |
| Generative | `gan` with separately configurable generator/discriminator architectures | None | `gan` |

Experimental architectures and the EBGAN-Playground migration question remain
outside this stable registry. GAN *model training* and synthetic-data *sampling*
remain distinct operations. The authoritative HSQA_DBN tangent-manifold
augmentation is unchanged and is not copied into this layer.

## Public API

```python
from ml_lab.experiments import (
    ExperimentStore, list_model_families, list_model_methods,
    describe_model_method, configure_experiment_model,
    prepare_configured_model_handoff,
)

store = ExperimentStore("/path/to/project")
# Start with a dataset-selected experiment from Sprint 2.3.
experiment = store.get_experiment(experiment_id)

families = list_model_families()
methods = list_model_methods("representation")
controls = describe_model_method("representation", "pca")
configured = configure_experiment_model(
    store, experiment_id,
    family="representation", method="pca",
    settings={"config": {"n_components": 2}},
    expected_revision=experiment["revision"],
)
handoff = prepare_configured_model_handoff(store, experiment_id)
# handoff['task'] and handoff['payload'] can be passed to execute_task
# in a future execution stage. Sprint 3.1 does NOT invoke a model or create a run.
```

For supervised suites, use `method="suite"` with explicit settings such as
`{"estimators": ["ridge", "rf"], "config": {"cv_folds": 3}}`.
Estimator IDs must exist in that family's authoritative registry. Single-method
selection emits the current legacy `estimators: [method]` shape. None of this
restricts the direct, preexisting `execute_task()` multi-estimator API.

## Configuration and revision contract

- `ml-lab.model-configuration@1` stores selected **family**, **method**, legacy
  **task**, and a validated **task-settings** object in the experiment's existing
  `model_spec_json`. The project-local SQLite schema is unchanged.
- Controls for existing scientific dataclass configurations are described using
  field names, JSON-typed values, valid enum options, and defaults. Unknown
  options, wrong types, invalid family/method IDs, and invalid ranges are rejected
  before any experiment revision is published. Nested CD-k options and RBM
  family-specific settings are validated. Advanced controls remain available;
  a UI need not expose all of them at once.
- Explicitly configured values are stored as strict JSON. Unspecified defaults
  are resolved by the **existing scientific engine**; they are not duplicated
  into a second set of model defaults. Environment/package-version recording and
  a fully resolved executable configuration snapshot remain S4 lifecycle work.
  In particular, legacy classifier/regressor `error_score=NaN` is not written
  into the strict-JSON SQLite store; omitting it preserves engine behavior.
  User-provided `training_config.checkpoint_path` is rejected in this opt-in
  interface until S4 assigns project-local checkpoint destinations; direct
  legacy task APIs are unchanged.
- A model cannot be configured without an S2.1/S2.3 pinned dataset. Column and
  target compatibility are checked, and the selected source/provenance hashes
  are revalidated. Feature numeric validation runs when the full handoff is
  prepared, as in Sprint 2.1.
- `expected_revision` is required. Publication uses the existing SQLite atomic
  compare-and-swap; stale browser/client state cannot replace a newer model
  revision. An identical validated configuration is idempotent.
- Reconfiguration creates a new immutable revision, retains the selected dataset
  and clears the *new* evaluation plan. Historical experiment revisions and
  their completed run links remain intact. An S2.3 dataset change clears the
  active model configuration, requiring explicit reconfiguration.
- Legacy-recorded task experiments have a different input schema and cannot be
  silently converted into dataset-selected model experiments.
- `prepare_configured_model_handoff()` revalidates the persisted model settings,
  dataset content and Data Lab lineage, then wraps the S2.1 task payload with
  `ml-lab.configured-model-handoff@1` and the model specification. It does
  **not** execute a run, register a checkpoint, or claim that torch is present.

## Boundaries

This patch does **not** introduce an EBM/optimizer equivalence contract,
training queue, GPU execution, new synthetic-data methods, or a full model UI.
Optimization remains an optional later stage. PAH remains an optional mediator
and never reads the ML_Lab private database. HSQA_DBN and pyPIQUE remain
untouched. The scientific matrix editor retains the separate minimalist
pyPIQUE-inspired presentation specified in Sprint 0.2.

## Local verification

```bash
PYTHONPATH=src python -m pytest -q tests/test_experiment_models.py \
    tests/test_experiment_dataset_selection.py \
    tests/test_experiment_dataset_handoff.py \
    tests/test_experiment_store.py tests/test_experiment_artifacts.py \
    tests/test_experiment_legacy_adapter.py
```

**Next:** Sprint 3.2 builds the visible unified model-configuration workspace
on this API without removing the existing task-specific workbenches prematurely.
