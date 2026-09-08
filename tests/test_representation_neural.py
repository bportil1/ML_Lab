from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from ml_lab import representation
from ml_lab.representation.reporting import save_result


def test_mlp_autoencoder_has_clean_encode_decode_contract():
    config = representation.MLPAutoencoderConfig(hidden_dims=(10, 6), latent_dim=3, activation="gelu")
    model = representation.build_mlp_autoencoder(8, config)
    x = torch.rand(5, 8)
    latent = model.encode(x)
    reconstruction = model.decode(latent)
    assert tuple(latent.shape) == (5, 3)
    assert tuple(reconstruction.shape) == (5, 8)
    assert tuple(model(x).shape) == (5, 8)


def test_autoencoder_training_is_reproducible_and_reports_artifacts(tmp_path: Path):
    rng = np.random.default_rng(12)
    basis = rng.normal(size=(48, 2))
    projection = rng.normal(size=(2, 6))
    X = basis @ projection
    model_config = representation.MLPAutoencoderConfig(hidden_dims=(8, 4), latent_dim=2, activation="relu")
    training_config = representation.AutoencoderTrainingConfig(
        epochs=12,
        batch_size=16,
        learning_rate=5e-3,
        validation_fraction=0.2,
        patience=None,
        scaling="standard",
        random_state=9,
        device="cpu",
    )
    first = representation.autoencode(X, model_config=model_config, training_config=training_config)
    second = representation.autoencode(X, model_config=model_config, training_config=training_config)

    assert first.method == "mlp_autoencoder"
    assert first.latent.shape == (48, 2)
    assert first.reconstruction.shape == X.shape
    assert len(first.history) == 12
    np.testing.assert_allclose(first.latent, second.latent, atol=1e-6)
    assert first.metrics["rmse"] >= 0.0

    save_result(first, tmp_path)
    assert (tmp_path / "result.json").exists()
    assert (tmp_path / "latent.csv").exists()
    assert (tmp_path / "reconstruction.csv").exists()
    assert (tmp_path / "training_history.csv").exists()
    assert (tmp_path / "model.pt").exists()
    assert (tmp_path / "scaler.joblib").exists()
