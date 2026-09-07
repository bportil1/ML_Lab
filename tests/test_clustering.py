from pathlib import Path

from sklearn.datasets import make_blobs

from ml_lab.clustering import ClusteringSearchConfig, run_cluster_selection
from ml_lab.clustering.evaluation import external_metrics, internal_metrics, stability_score
from ml_lab.clustering.reporting import save_results
from ml_lab.core.specs import EstimatorSpec
from sklearn.cluster import KMeans


def tiny_kmeans_spec():
    return EstimatorSpec(
        id="tiny_kmeans",
        name="Tiny K-Means",
        task="clustering",
        factory=lambda seed: KMeans(random_state=seed, n_init="auto"),
        param_grid=[{"n_clusters": [2, 3, 4]}],
        preprocess="standard",
    )


def test_internal_and_external_metrics():
    X, y = make_blobs(n_samples=90, centers=3, random_state=7)
    labels = KMeans(n_clusters=3, random_state=7, n_init="auto").fit_predict(X)
    internal = internal_metrics(X, labels)
    external = external_metrics(y, labels)
    assert internal["cluster_count"] == 3
    assert internal["silhouette"] is not None
    assert external["adjusted_rand"] > 0.9
    assert stability_score([labels, labels.copy()]) == 1.0


def test_cluster_selection_and_stability():
    X, y = make_blobs(n_samples=120, centers=[(-8, -8), (0, 8), (8, -8)], cluster_std=0.45, random_state=11)
    result = run_cluster_selection(
        tiny_kmeans_spec(),
        X,
        ClusteringSearchConfig(selection_metric="silhouette", repeats=2, random_state=11),
        y_true=y,
    )
    assert result.task == "clustering"
    assert result.best_params["n_clusters"] == 3
    assert result.metrics["cluster_count"] == 3
    assert result.external_metrics is not None
    assert result.external_metrics["adjusted_rand"] > 0.9
    assert result.stability is not None
    assert len(result.candidates) == 3


def test_clustering_reporting(tmp_path: Path):
    X, y = make_blobs(n_samples=90, centers=3, random_state=5)
    result = run_cluster_selection(
        tiny_kmeans_spec(),
        X,
        ClusteringSearchConfig(repeats=2, random_state=5),
        y_true=y,
    )
    save_results([result], tmp_path)
    assert (tmp_path / "result.json").exists()
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "candidates.csv").exists()
    assert (tmp_path / "cluster_assignments.csv").exists()
    assert (tmp_path / "models" / "tiny_kmeans.joblib").exists()
