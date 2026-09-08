from pathlib import Path

import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge

from ml_lab.core.specs import EstimatorSpec
from ml_lab.regression import RegressionSearchConfig, run_model_selection, split_dataset
from ml_lab.regression.reporting import save_results


def tiny_spec() -> EstimatorSpec:
    return EstimatorSpec(
        id="tiny_ridge",
        name="Tiny Ridge",
        task="regression",
        factory=lambda seed: Ridge(),
        param_grid=[{"alpha": [0.1, 1.0]}],
        preprocess="standard",
    )


def test_single_output_regression_selection():
    raw = load_diabetes(as_frame=True)
    bundle = split_dataset(raw.data, raw.target, test_size=0.2, random_state=3)
    result = run_model_selection(
        tiny_spec(),
        bundle,
        RegressionSearchConfig(cv_folds=3, n_jobs=1, random_state=3),
    )
    assert result.task == "regression"
    assert result.best_params["alpha"] in {0.1, 1.0}
    assert result.multi_output is False
    assert result.metrics["rmse"] > 0
    assert result.metrics["mae"] > 0
    assert result.target_names == ["target"]


def test_multi_output_regression_uses_shared_search_contract():
    raw = load_diabetes(as_frame=True)
    X = raw.data.iloc[:160].reset_index(drop=True)
    y = pd.DataFrame({
        "disease_progression": raw.target.iloc[:160].reset_index(drop=True),
        "synthetic_second_target": raw.target.iloc[:160].reset_index(drop=True) * 0.5 + X["bmi"] * 20.0,
    })
    bundle = split_dataset(X, y, test_size=0.2, random_state=4)
    result = run_model_selection(
        tiny_spec(),
        bundle,
        RegressionSearchConfig(cv_folds=2, n_jobs=1, random_state=4),
    )
    assert result.multi_output is True
    assert result.y_pred.ndim == 2
    assert result.y_pred.shape[1] == 2
    assert result.target_names == ["disease_progression", "synthetic_second_target"]


def test_regression_reporting(tmp_path: Path):
    raw = load_diabetes(as_frame=True)
    bundle = split_dataset(raw.data, raw.target, test_size=0.2, random_state=5)
    result = run_model_selection(
        tiny_spec(),
        bundle,
        RegressionSearchConfig(cv_folds=2, n_jobs=1, random_state=5),
    )
    save_results([result], tmp_path)
    assert (tmp_path / "result.json").exists()
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "candidates.csv").exists()
    assert (tmp_path / "predictions.csv").exists()
    assert (tmp_path / "models" / "tiny_ridge.joblib").exists()
