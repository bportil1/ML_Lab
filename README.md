# ML_Lab

`ML_Lab` is a small, headless machine-learning engine intended for standalone use and later integration into PAH. It starts with **classification** and **clustering** and is structured so other basic sklearn-backed task families can be added without carrying forward the legacy Classifier Generator visualization layer.

The classification engine reuses the strongest design ideas from the supplied Classifier Generator refactor: declarative estimator registries, preprocessing inside sklearn pipelines, cross-validation on training data only, one holdout evaluation after selection, and explicit machine-readable reporting.

Clustering is a separate first-class subsystem with clustering-specific selection/evaluation semantics.

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
