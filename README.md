# ML_Lab

`ML_Lab` is a small, headless machine-learning engine intended for standalone use and later integration into PAH. It provides first-class **classification**, **regression**, **clustering**, and **representation/compression** task families and is structured so additional basic ML capabilities can be added without carrying forward the legacy Classifier Generator visualization layer.

The classification engine reuses the strongest design ideas from the supplied Classifier Generator refactor: declarative estimator registries, preprocessing inside sklearn pipelines, cross-validation on training data only, one holdout evaluation after selection, and explicit machine-readable reporting.

Regression is a parallel supervised subsystem with regression-specific metrics and single- or multi-output targets. Clustering remains a separate first-class subsystem with clustering-specific selection/evaluation semantics.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
```

## Discover estimators

```bash
ml-lab list-estimators
ml-lab list-estimators --task classification
ml-lab list-estimators --task clustering
ml-lab list-estimators --task regression
```

## Classification

```bash
ml-lab classify data.csv \
  --label label \
  --algorithms rf lda ridge \
  --cv 5 \
  --output results/classification
```

Python:

```python
from ml_lab import classification

results = classification.run(
    X,
    y,
    estimators=["rf", "lda"],
    config=classification.ClassificationSearchConfig(cv_folds=5),
)
```

Classification artifacts include `result.json`, `metrics.csv`, `candidates.csv`, and fitted models under `models/`.

## Regression

```bash
ml-lab regress data.csv \
  --target target \
  --algorithms ridge random_forest extra_trees svr knn \
  --cv 5 \
  --output results/regression
```

Multiple numeric targets are supported:

```bash
ml-lab regress data.csv --target target_a target_b
```

Python:

```python
from ml_lab import regression

results = regression.run(
    X,
    y,
    estimators=["ridge", "random_forest"],
    config=regression.RegressionSearchConfig(cv_folds=5),
)
```

Regression evaluation includes RMSE, MAE, R², median absolute error, and explained variance. Search is performed on the training split only; the selected estimator is evaluated once on the holdout split. Multi-output targets use a shared search contract and preserve target names in prediction artifacts.

Regression artifacts include `result.json`, `metrics.csv`, `candidates.csv`, `predictions.csv`, and fitted models under `models/`.

## Clustering

```bash
ml-lab cluster data.csv \
  --algorithms kmeans agglomerative spectral birch gmm \
  --selection-metric silhouette \
  --output results/clustering
```

An optional label column may be supplied for **external evaluation only**:

```bash
ml-lab cluster data.csv --label known_group
```

Python:

```python
from ml_lab import clustering

results = clustering.run(
    X,
    estimators=["kmeans", "spectral", "dbscan"],
    config=clustering.ClusteringSearchConfig(
        selection_metric="silhouette",
        repeats=3,
    ),
)
```

Initial clustering families:

- K-Means
- MiniBatch K-Means
- Agglomerative
- Spectral
- BIRCH
- DBSCAN
- OPTICS
- Gaussian Mixture

Internal evaluation includes silhouette, Calinski-Harabasz, Davies-Bouldin, cluster counts/sizes, cluster-size entropy, noise fraction, and repeated-run ARI stability. Optional external labels add ARI, NMI, homogeneity, completeness, and V-measure.

Clustering artifacts include `result.json`, `metrics.csv`, `candidates.csv`, `cluster_assignments.csv`, and fitted models.

## Representation / compression

Stable representation operations are available through `ml_lab.representation`. PCA works with the base install; MLP and Transformer autoencoders require the optional neural dependency.

```bash
ml-lab represent data.csv --method pca --components 3 --output results/pca

pip install -e '.[neural]'
ml-lab represent data.csv \
  --method mlp_autoencoder \
  --hidden-dims 64 32 \
  --latent-dim 8 \
  --epochs 100 \
  --output results/autoencoder

ml-lab represent data.csv \
  --method transformer_autoencoder \
  --token-width 4 \
  --model-dim 64 \
  --nhead 4 \
  --latent-dim 16 \
  --epochs 100 \
  --output results/transformer-autoencoder
```

Python:

```python
from ml_lab import representation

pca_result = representation.pca(
    X,
    config=representation.PCARepresentationConfig(n_components=3),
)

ae_result = representation.autoencode(
    X,
    model_config=representation.MLPAutoencoderConfig(
        hidden_dims=(64, 32),
        latent_dim=8,
    ),
    training_config=representation.AutoencoderTrainingConfig(epochs=100),
)

transformer_result = representation.transformer_autoencode(
    X,
    model_config=representation.TransformerAutoencoderConfig(
        token_width=4,
        model_dim=64,
        nhead=4,
        latent_dim=16,
    ),
    training_config=representation.AutoencoderTrainingConfig(epochs=100),
)
```

The stable MLP and Transformer autoencoders expose separate `encode()`, `decode()`, and `forward()` model methods. Training is kept outside the model object. Results contain latent vectors, reconstructions in the original feature scale, reconstruction metrics, training history where applicable, and model/transformation provenance.

Representation artifacts include `result.json`, `latent.csv`, `reconstruction.csv`, plus `transformer.joblib` for PCA or `model.pt`, optional `scaler.joblib`, and `training_history.csv` for neural runs.

PyTorch remains optional: importing `ml_lab` or using PCA does not import PyTorch.

## Basic preprocessing operations

ML Lab also exposes small reusable operations directly:

```python
from ml_lab import preprocessing

scaled = preprocessing.scale(X, "standard")
filled = preprocessing.impute(X_with_missing, strategy="median")
embedding = preprocessing.pca(X, n_components=3)
```

These return both transformed data and the fitted sklearn transformer so callers can preserve provenance or apply the same transformation later.

## PAH / HSQA direction

ML Lab should remain independent: it does not import PAH or HSQA_DBN. A PAH or HSQA adapter should call the public Python API and consume the returned records/artifacts.

For HSQA_DBN specifically, generic clustering/search/evaluation can eventually move here while HSQA keeps domain-specific feature construction, node semantics, lineage, interpretation, and visualization.

See `ARCHITECTURE.md` for the intended boundary.

## Experimental incubator

Research prototypes live behind `ml_lab.experimental` rather than being promoted directly into the stable API. The stable package never imports experimental modules. Discovery reads lightweight manifests, and a module implementation is imported only when explicitly loaded or run.

```bash
ml-lab experimental list
ml-lab experimental info <experiment_id>
ml-lab experimental run <experiment_id> --kwargs-json '{"example": 1}'
```

Built-in experimental modules include `linux_binary_identification`, `binary_fractal_conversion`, `max_clique_rl`, `transformer_vae`, and `transformer_vae_decomposed`.

Programmatic hosts can register an experiment without importing its implementation:

```python
from ml_lab.experimental import ExperimentalManifest, register_experiment

register_experiment(
    ExperimentalManifest(
        id="my_experiment",
        name="My Experiment",
        module_path="my_package.experimental_module",
        capabilities=("representation",),
    )
)
```

The implementation is loaded only by `load_experiment()` or `run_experiment()`. Future legacy/research imports such as Linux Binary Identification, Max Clique RL, and partitioned RBM training can therefore incubate here without becoming dependencies of classification, regression, clustering, or other stable task families.

## Optional Flask / PAH adapter

Flask is an optional integration dependency, not part of the ML engine:

```bash
pip install -e '.[flask]'
```

A host can mount ML Lab at any route depth:

```python
from flask import Flask
from ml_lab.flask_adapter import create_blueprint

app = Flask(__name__)
app.register_blueprint(
    create_blueprint(enable_experimental=False),
    url_prefix="/pah/services/ml-lab",
)
```

Initial endpoints are:

```text
GET  /health
GET  /estimators?task=classification|regression|clustering|all
POST /run/classification
POST /run/regression
POST /run/clustering
POST /run/representation
```

When a host explicitly creates the Blueprint with `enable_experimental=True`, it additionally exposes experimental manifest/list/run routes. Experimental HTTP execution is disabled by default.

The run endpoints accept JSON-shaped arrays (`X`, `y`/`y_true`, optional estimator lists, and task config dictionaries) and return serializable ML Lab result records. Representation payloads use `method="pca"` with `config`, or `method="mlp_autoencoder"` / `method="transformer_autoencoder"` with `model_config` and `training_config`. Transformer payloads may contain either 2-D feature matrices or pre-tokenized 3-D arrays. The adapter is synchronous and intentionally thin; PAH remains responsible for workspace state, long-running job orchestration, authentication, and visualization.

## Built-in experimental modules

Sprint 3 adds two isolated research experiments recovered from user-provided legacy repositories. They remain outside ML Lab's stable API and are loaded only when explicitly selected.

```bash
ml-lab experimental list
ml-lab experimental info linux_binary_identification
ml-lab experimental info max_clique_rl
```

### Linux Binary Identification

`linux_binary_identification` incubates the legacy byte-stream representation and Transformer-autoencoder work without carrying forward the old visualization application or clustering implementation. Base-install modes include byte-to-RGB and RGB-patch tokenization. The Transformer path is optional and requires PyTorch.

```bash
ml-lab experimental run linux_binary_identification \
  --kwargs-json '{"mode":"patch_tokens","data":[0,1,2,3],"image_size":2,"patch_size":1}'
```

The binary-specific experiment now delegates generic Transformer model/training behavior to the stable `ml_lab.representation` Transformer autoencoder. Byte/RGB/patch conversion remains experimental and domain-specific.

### Max Clique RL

`max_clique_rl` isolates graph-specific maximum-clique search ideas from the legacy Max Clique ML Solver. A deterministic greedy baseline works with the base installation. The optional DQN path requires PyTorch.

```bash
ml-lab experimental run max_clique_rl \
  --kwargs-json '{"mode":"greedy","generated":{"num_nodes":12,"clique_size":4,"background_edge_probability":0.05,"seed":42}}'
```

Legacy partitioned-RBM and optimizer implementations are intentionally excluded from this import. Those concepts will be reconsidered separately against ML Lab's future energy-based/training interfaces.

Install the optional neural dependency when using Transformer/DQN experiment paths:

```bash
pip install -e '.[experimental-neural]'
```

## Shared neural infrastructure

ML Lab 0.7.0 includes `ml_lab.neural`, a task-neutral foundation for stable neural models. Importing it does not import PyTorch. The shared layer provides:

- common neural training configuration;
- deterministic Python/NumPy/PyTorch seeding;
- CPU/CUDA/MPS device resolution;
- deterministic train/validation index splitting;
- training-history records;
- early-stopping/model-selection state;
- in-memory and optional on-disk best-model checkpoints;
- checkpoint reload helpers;
- lightweight training callbacks;
- a common `NeuralTrainer` / `NeuralTrainingResult` contract.

The stable MLP autoencoder now uses this shared infrastructure rather than maintaining a private copy of those behaviors. Existing representation calls remain valid. A neural run may additionally request a best-model checkpoint:

```python
from ml_lab import representation

result = representation.autoencode(
    X,
    training_config=representation.AutoencoderTrainingConfig(
        epochs=100,
        checkpoint_path="results/best-autoencoder.pt",
    ),
)
```

or from the CLI:

```bash
ml-lab represent data.csv \
  --method mlp_autoencoder \
  --checkpoint-path results/best-autoencoder.pt
```

Callbacks are available to programmatic callers through the `callbacks=` argument on `representation.autoencode()` / `representation.run()`. This infrastructure is intended to be reused by the stable Transformer autoencoder, GAN, and later energy-based model trainers rather than allowing each neural family to invent its own runtime/checkpoint/history conventions.


## Stable Transformer autoencoder (0.7.0)

The Transformer autoencoder promoted from the Linux-binary experiment is now a generic stable representation model. It accepts either ordinary 2-D feature matrices or pre-tokenized 3-D tensors. Two-dimensional matrices are split into adjacent fixed-width tokens, with zero padding masked out of attention pooling and reconstruction loss. Three-dimensional inputs are treated as already tokenized sequences.

The stable implementation fixes several prototype hazards from the legacy lineage: file/image preprocessing is outside the model; the model does not own an optimizer or override `nn.Module.train()`; VAE-specific losses are not mixed into a deterministic autoencoder; decoder queries and latent memory have explicit shapes; padded elements do not affect reconstruction loss; and arbitrary-rank reconstruction metrics work for sequence tensors.


### Experimental Transformer VAE

ML Lab also incubates a Transformer variational autoencoder under
`ml_lab.experimental.transformer_vae`. It uses the stable Transformer token adapter
and shared neural runtime, but keeps variational sampling and KL objectives outside
the stable representation API while they mature.

### Experimental decomposed Transformer VAE (0.7.2)

`ml_lab.experimental.transformer_vae_decomposed` reuses the same Transformer VAE
architecture but replaces the single beta-weighted KL term with a research objective
that separates the latent regularizer into **ICMI**, **total correlation (TC)**, and
**dimension-wise KL (DWKL)**. The implementation uses a differentiable minibatch
mixture-density estimator, records every component separately, supports independent
weights and warm-up, and reports the ordinary analytic Gaussian KL as a diagnostic.

The objective is:

```text
reconstruction
+ alpha * ICMI
+ beta  * TC
+ gamma * DWKL
```

Because `q(z)` and `q(z_j)` are estimated from each minibatch, these terms are
research estimates rather than exact full-dataset quantities. The estimator is also
quadratic in minibatch size, so this variant remains explicitly experimental.


### Experimental contractive Transformer VAE

`transformer_vae_contractive` adds a research variant of the Transformer VAE with a first-order contractive penalty on the encoder posterior mean, `||d mu(x) / d x||_F^2`. Exact and Hutchinson estimators are available; this objective remains experimental rather than part of the stable representation API.
