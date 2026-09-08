from __future__ import annotations

import numpy as np
import pytest

from ml_lab.flask_adapter import PayloadError, execute_task


def test_clustering_payload_executes_without_flask_dependency():
    X = np.vstack([
        np.zeros((6, 2)),
        np.full((6, 2), 5.0),
    ])
    response = execute_task(
        "clustering",
        {
            "X": X.tolist(),
            "feature_names": ["x", "y"],
            "estimators": ["kmeans"],
            "config": {"repeats": 1, "random_state": 7},
        },
    )
    assert response["task"] == "clustering"
    assert response["results"][0]["estimator_id"] == "kmeans"
    assert len(response["results"][0]["labels"]) == len(X)
    assert "algorithm_agreement" in response
    assert response["results"][0]["stability_analysis"]["repeats_succeeded"] == 1


def test_payload_validation_is_host_friendly():
    with pytest.raises(PayloadError, match="two-dimensional"):
        execute_task("clustering", {"X": [1, 2, 3]})
    with pytest.raises(PayloadError, match="unsupported"):
        execute_task("not-a-task", {"X": [[1], [2]]})


def test_gan_payload_executes_without_flask_dependency():
    pytest.importorskip("torch")
    X = np.random.default_rng(9).normal(size=(18, 4))
    response = execute_task(
        "gan",
        {
            "X": X.tolist(),
            "feature_names": ["a", "b", "c", "d"],
            "model_config": {
                "latent_dim": 3,
                "generator_hidden_dims": [8],
                "discriminator_hidden_dims": [8],
            },
            "training_config": {
                "epochs": 1,
                "batch_size": 6,
                "generated_sample_count": 5,
                "device": "cpu",
            },
        },
    )
    assert response["task"] == "gan"
    assert len(response["result"]["generated_samples"]) == 5
    assert response["result"]["metadata"]["generator_type"] == "mlp"
