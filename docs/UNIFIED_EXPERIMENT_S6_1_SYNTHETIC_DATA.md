# Sprint 6.1 — Generative sampling and synthetic datasets

This sprint introduces an **opt-in** synthetic data workflow using the existing
project-local experiment store, Data Lab dataset selections, run controls, and
provenance sidecars. It does **not** replace or modify scientific engines or
move tangent-manifold generation out of HSQA_DBN.

## Usage

After Data Lab → **Use in experiment**, open **Experiments → Model** and select
**Generate synthetic dataset**. Choose a sampling method, number of rows, and
seed. For Student-t, provide degrees of freedom greater than 2. The new run
retains the exact experiment revision and source dataset hash; the result page
provides a CSV download and an explicit **Use generated data in an experiment**
link. Creating a dataset never changes the source or automatically switches the
current experiment.

For standalone Python use:

```python
from ml_lab.experiments import ExperimentStore, generate_synthetic_dataset

store = ExperimentStore('/local/project')
result = generate_synthetic_dataset(
    store, '<experiment-id>',
    {'method': 'gaussian_correlated', 'count': 250, 'seed': 42},
    expected_revision=1,
)
print(result['dataset']['path'])
```

Other methods: `bootstrap` (row bootstrap), `gaussian_independent` (per-feature
mean and sample standard deviation), `gaussian_correlated` (empirical covariance),
`student_t` (correlated multivariate Student-t with matching finite covariance
for `degrees_of_freedom > 2`), and `gan`/`rbm` only when a compatible model
configuration has been saved in the selected experiment. The Gaussian and
Student-t generators estimate moments from the selected numeric feature columns.
They **do not** preserve domain constraints such as positive-only counts or
integer-only observations. If this matters, use explicit downstream validation
and transformation before using the generated data as training input.

## Model-based sampling boundary

GAN and RBM generation use ML_Lab's existing generative GAN and CD-k RBM
implementations, with the selected experiment's feature matrix and model
settings. Each request **trains a fresh model** using the requested seed and
then samples the trained in-memory model: GAN via `generative.gan.sample`, RBM
via `energy_based.rbm.run_chain` with ten Gibbs steps from randomly selected
source rows. Saved training-run JSON records are **not trained-weight
checkpoints** and cannot be resampled. Model checkpoint persistence/reuse is a
separate future feature. There is no implicit hyperparameter optimization.
The training engine may have its own additional randomness and numerical
conditions; sample generation is not a guarantee of distributional fidelity.

The legacy experimental `gan_stabilization.tangent` helper remains isolated.
The authoritative HSQA_DBN tangent-manifold augmenter is not migrated or
aliased into this generic workspace.

## Persistence and safety

- Plans allow only a known method, a bounded count of 2–5,000, a 32-bit seed,
  and optional finite Student-t degrees of freedom above 2.
- No user-controlled output path. Files are stored under the run's
  `analysis/ml_lab/training_runs/<run_id>/` directory in the selected project.
- The source's byte hash and Data Lab lineage are checked at queueing and
  execution boundaries. The source is never overwritten.
- `synthetic.csv`, its native Data Lab provenance sidecar, an input manifest,
  and a result summary are registered as SHA-256 artifact records with run
  ancestry. A failed or cancelled run is not presented as successful.
- Generation is local and uses the existing in-process worker; it cannot
  forcibly cancel a model during a scientific kernel. An application restart
  can interrupt such a job. A completed run can be explicitly selected as a
  new experiment dataset through the existing Data Lab form.
- Statistical methods require two or more complete numeric feature rows.
  Missing/nonfinite values are rejected, not silently imputed.

## Non-goals and follow-up

This is not a replacement for the legacy GAN training workbench. Model
checkpoint reuse, experimental EBGAN verification, HSQA_DBN tangent-manifold
code ownership, richer synthetic-quality diagnostics, and data validity
constraints remain future work. Synthetic runs are **not** included as
training/optimization metric comparisons because they are dataset creation
operations, not measured model evaluations.
