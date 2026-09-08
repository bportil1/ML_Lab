from ml_lab.classification.registry import get_estimator_spec as get_classifier, list_estimators as list_classifiers
from ml_lab.clustering.registry import get_estimator_spec as get_clusterer, list_estimators as list_clusterers
from ml_lab.regression.registry import get_estimator_spec as get_regressor, list_estimators as list_regressors
from ml_lab.registry import list_estimators


def test_classifier_registry_preserves_legacy_families():
    ids = {spec.id for spec in list_classifiers()}
    expected = {"knn", "svc", "gp", "rf", "lsvc", "hgbc", "ada", "qda", "lda", "mlp", "ridge", "pa", "sgd", "etc", "gnb", "mnb", "compnb", "bnb"}
    assert expected <= ids
    assert get_classifier("lda").factory(42).__class__.__name__ == "LinearDiscriminantAnalysis"


def test_cluster_registry_has_first_class_families():
    ids = {spec.id for spec in list_clusterers()}
    assert {"kmeans", "minibatch_kmeans", "agglomerative", "spectral", "birch", "dbscan", "optics", "gmm"} <= ids
    assert get_clusterer("spectral").task == "clustering"


def test_regression_registry_has_first_class_families():
    ids = {spec.id for spec in list_regressors()}
    assert {
        "linear", "ridge", "sgd", "decision_tree", "lasso", "elasticnet",
        "random_forest", "extra_trees", "kernel_ridge", "omp", "knn",
        "svr", "linear_svr", "huber", "quantile",
    } <= ids
    assert get_regressor("ridge").task == "regression"


def test_top_level_registry_combines_tasks():
    tasks = {spec.task for spec in list_estimators()}
    assert tasks == {"classification", "clustering", "regression"}
