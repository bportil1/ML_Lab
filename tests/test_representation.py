from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml_lab import representation
from ml_lab.flask_adapter import execute_task
from ml_lab.representation.reporting import save_result


def test_pca_representation_reconstructs_and_preserves_feature_names(tmp_path: Path):
    rng = np.random.default_rng(4)
    X = pd.DataFrame(rng.normal(size=(40, 5)), columns=list("abcde"))
    result = representation.pca(
        X,
        config=representation.PCARepresentationConfig(n_components=3, scaling="standard", random_state=4),
    )
    assert result.method == "pca"
    assert result.latent.shape == (40, 3)
    assert result.reconstruction.shape == X.shape
    assert result.feature_names == list(X.columns)
    assert 0.0 < result.metrics["explained_variance_ratio_sum"] <= 1.0
    assert result.metrics["rmse"] >= 0.0

    save_result(result, tmp_path)
    assert (tmp_path / "result.json").exists()
    assert (tmp_path / "latent.csv").exists()
    assert (tmp_path / "reconstruction.csv").exists()
    assert (tmp_path / "transformer.joblib").exists()


def test_representation_service_supports_headless_pca_payload():
    X = np.arange(60, dtype=float).reshape(20, 3)
    response = execute_task(
        "representation",
        {
            "X": X.tolist(),
            "feature_names": ["x", "y", "z"],
            "method": "pca",
            "config": {"n_components": 2, "scaling": "standard", "random_state": 7},
        },
    )
    assert response["task"] == "representation"
    result = response["result"]
    assert result["method"] == "pca"
    assert result["latent_shape"] == [20, 2]
    assert len(result["latent"]) == 20
    assert len(result["reconstruction"]) == 20


def test_csv_loader_rejects_non_numeric_input(tmp_path: Path):
    path = tmp_path / "mixed.csv"
    pd.DataFrame({"x": [1, 2], "label": ["a", "b"]}).to_csv(path, index=False)
    try:
        representation.load_csv_features([path])
    except ValueError as exc:
        assert "non-numeric" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected non-numeric representation input to be rejected")
