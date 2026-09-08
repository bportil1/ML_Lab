from pathlib import Path

import numpy as np
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


def test_consensus_matrix_and_noise_eligibility():
    from ml_lab.clustering import coassignment_matrix, consensus_consistency

    runs = [
        [0, 0, 1, -1],
        [0, 1, 1, -1],
    ]
    consensus, eligible = coassignment_matrix(runs, ignore_noise=True)
    assert consensus.shape == (4, 4)
    assert consensus[0, 1] == 0.5
    assert consensus[0, 2] == 0.0
    assert eligible[0, 1] == 2
    assert eligible[0, 3] == 0
    assert np.isnan(consensus[0, 3])
    assert 0.5 <= consensus_consistency(consensus) <= 1.0


def test_repeat_stability_is_label_permutation_invariant():
    from ml_lab.clustering import repeat_stability_report

    report = repeat_stability_report(
        [
            np.array([0, 0, 1, 1]),
            np.array([7, 7, 4, 4]),
            np.array([2, 2, 9, 9]),
        ],
        repeats_requested=3,
        seeds=[10, 11, 12],
    )
    assert report.adjusted_rand_mean == 1.0
    assert report.normalized_mutual_info_mean == 1.0
    assert report.consensus_consistency == 1.0
    assert report.repeat_labels.shape == (3, 4)
    assert len(report.pairwise_records) == 3


def test_cross_algorithm_agreement_ignores_label_ids():
    from ml_lab.clustering import algorithm_agreement_report

    report = algorithm_agreement_report(
        ["a", "b", "c"],
        [
            np.array([0, 0, 1, 1]),
            np.array([5, 5, 9, 9]),
            np.array([0, 1, 0, 1]),
        ],
    )
    assert report.adjusted_rand_matrix[0, 1] == 1.0
    assert report.normalized_mutual_info_matrix[0, 1] == 1.0
    assert report.adjusted_rand_matrix[0, 2] < 1.0
    assert report.summary_record()["pair_count"] == 3


def test_selected_clustering_has_repeat_stability_artifacts(tmp_path: Path):
    X, y = make_blobs(n_samples=80, centers=3, random_state=17)
    result = run_cluster_selection(
        tiny_kmeans_spec(),
        X,
        ClusteringSearchConfig(repeats=3, random_state=17),
        y_true=y,
    )
    assert result.stability_analysis is not None
    assert result.stability_analysis.repeat_labels.shape == (3, len(X))
    assert result.stability_analysis.consensus_matrix.shape == (len(X), len(X))
    save_results([result], tmp_path)
    assert (tmp_path / "algorithm_agreement.json").exists()
    assert (tmp_path / "algorithm_agreement_adjusted_rand.csv").exists()
    assert (tmp_path / "stability" / "summary.csv").exists()
    assert (tmp_path / "stability" / "tiny_kmeans_consensus.csv").exists()
    assert (tmp_path / "stability" / "tiny_kmeans_repeat_assignments.csv").exists()


def test_noise_is_not_treated_as_shared_cluster_when_ignored():
    labels_a = np.array([-1, -1, 0, 0])
    labels_b = np.array([-1, -1, 1, 1])
    # Historical behavior sees the two noise points as a shared label.
    assert stability_score([labels_a, labels_b], ignore_noise=False) == 1.0
    # Noise-aware mode compares only assigned samples; their cluster IDs may differ
    # but the partition is still identical, so agreement remains one on 50% coverage.
    from ml_lab.clustering import pairwise_label_agreement
    agreement = pairwise_label_agreement(labels_a, labels_b, ignore_noise=True)
    assert agreement["coverage_fraction"] == 0.5
    assert agreement["adjusted_rand"] == 1.0
