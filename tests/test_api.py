from sklearn.datasets import load_iris

from ml_lab import classification, clustering, regression
from ml_lab.core.specs import EstimatorSpec
from sklearn.linear_model import LogisticRegression, Ridge
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


def test_regression_public_module_is_callable():
    raw = load_iris(as_frame=True)
    spec = EstimatorSpec(
        "rr", "Ridge Regression", "regression",
        lambda seed: Ridge(),
        [{"alpha": [1.0]}], "standard"
    )
    target = raw.data["sepal length (cm)"]
    features = raw.data.drop(columns=["sepal length (cm)"])
    dataset = regression.split_dataset(features, target, random_state=2)
    result = regression.run_model_selection(
        spec, dataset, regression.RegressionSearchConfig(cv_folds=2, n_jobs=1)
    )
    assert result.task == "regression"
    assert "rmse" in result.metrics
