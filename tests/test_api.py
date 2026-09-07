from sklearn.datasets import load_iris

from ml_lab import classification, clustering
from ml_lab.core.specs import EstimatorSpec
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans


def test_public_modules_are_callable():
    raw = load_iris(as_frame=True)
    class_spec = EstimatorSpec(
        "lr", "LR", "classification",
        lambda seed: LogisticRegression(random_state=seed, max_iter=500),
        [{"C": [1.0]}], "standard"
    )
    dataset = classification.split_dataset(raw.data, raw.target, random_state=2)
    c = classification.run_model_selection(class_spec, dataset, classification.ClassificationSearchConfig(cv_folds=2, n_jobs=1))
    assert c.task == "classification"

    cluster_spec = EstimatorSpec(
        "km", "KM", "clustering",
        lambda seed: KMeans(random_state=seed, n_init="auto"),
        [{"n_clusters": [3]}], "standard"
    )
    k = clustering.run_cluster_selection(cluster_spec, raw.data, clustering.ClusteringSearchConfig(repeats=2))
    assert k.task == "clustering"
