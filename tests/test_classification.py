from pathlib import Path

from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.linear_model import LogisticRegression

from ml_lab.classification import ClassificationSearchConfig, run_model_selection, split_dataset
from ml_lab.classification.reporting import save_results
from ml_lab.core.specs import EstimatorSpec


def tiny_spec():
    return EstimatorSpec(
        id="tiny_logreg",
        name="Tiny Logistic Regression",
        task="classification",
        factory=lambda seed: LogisticRegression(random_state=seed, max_iter=500),
        param_grid=[{"C": [0.1, 1.0]}],
        preprocess="standard",
    )


def test_binary_and_multiclass_selection():
    cancer = load_breast_cancer(as_frame=True)
    bundle = split_dataset(cancer.data, cancer.target, test_size=0.2, random_state=3)
    result = run_model_selection(tiny_spec(), bundle, ClassificationSearchConfig(cv_folds=2, n_jobs=1, random_state=3))
    assert result.task == "classification"
    assert result.best_params["C"] in {0.1, 1.0}
    assert 0.0 <= result.metrics["accuracy"] <= 1.0

    iris = load_iris(as_frame=True)
    bundle = split_dataset(iris.data, iris.target, test_size=0.2, random_state=3)
    result = run_model_selection(tiny_spec(), bundle, ClassificationSearchConfig(cv_folds=2, n_jobs=1, random_state=3))
    assert result.metrics["f1"] is not None
    assert result.metrics["roc_auc"] is not None


def test_classification_reporting(tmp_path: Path):
    raw = load_breast_cancer(as_frame=True)
    bundle = split_dataset(raw.data, raw.target, test_size=0.2, random_state=4)
    result = run_model_selection(tiny_spec(), bundle, ClassificationSearchConfig(cv_folds=2, n_jobs=1))
    save_results([result], tmp_path)
    assert (tmp_path / "result.json").exists()
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "candidates.csv").exists()
    assert (tmp_path / "models" / "tiny_logreg.joblib").exists()
