from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from ml_lab import representation
from ml_lab.flask_adapter import execute_task
from ml_lab.representation.reporting import save_result


def _small_model(**overrides):
    values = {
        "token_width": 2,
        "model_dim": 8,
        "latent_dim": 3,
        "nhead": 2,
        "encoder_layers": 1,
        "decoder_layers": 1,
        "feedforward_dim": 16,
        "dropout": 0.0,
        "max_tokens": 8,
    }
    values.update(overrides)
    return representation.TransformerAutoencoderConfig(**values)


def _small_training(**overrides):
    values = {
        "epochs": 4,
        "batch_size": 8,
        "learning_rate": 3e-3,
        "validation_fraction": 0.2,
        "patience": None,
        "scaling": "standard",
        "random_state": 11,
        "device": "cpu",
    }
    values.update(overrides)
    return representation.AutoencoderTrainingConfig(**values)


def test_transformer_model_contract_supports_masked_tokens():
    config = _small_model(token_width=1, max_tokens=5)
    model = representation.build_transformer_autoencoder(4, config)
    x = torch.rand(3, 5, 4)
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[:, -1] = False
    latent = model.encode(x, element_mask=mask)
    reconstruction = model(x, element_mask=mask)
    decoded = model.decode(latent, token_count=5, element_mask=mask)
    assert tuple(latent.shape) == (3, 3)
    assert tuple(reconstruction.shape) == (3, 5, 4)
    assert tuple(decoded.shape) == (3, 5, 4)


def test_transformer_autoencoder_handles_padded_vector_tokens_and_is_reproducible(tmp_path: Path):
    rng = np.random.default_rng(8)
    X = rng.normal(size=(24, 5))
    config = _small_model(token_width=2, max_tokens=4)
    training = _small_training()

    first = representation.transformer_autoencode(X, model_config=config, training_config=training)
    second = representation.transformer_autoencode(X, model_config=config, training_config=training)

    assert first.method == "transformer_autoencoder"
    assert first.latent.shape == (24, 3)
    assert first.reconstruction.shape == X.shape
    assert first.metadata["token_count"] == 3
    assert first.metadata["token_dim"] == 2
    assert len(first.history) == 4
    np.testing.assert_allclose(first.latent, second.latent, atol=1e-6)
    assert np.isfinite(first.metrics["rmse"])

    save_result(first, tmp_path)
    assert (tmp_path / "result.json").exists()
    assert (tmp_path / "latent.csv").exists()
    assert (tmp_path / "reconstruction.csv").exists()
    assert (tmp_path / "training_history.csv").exists()
    assert (tmp_path / "model.pt").exists()
    assert (tmp_path / "scaler.joblib").exists()


def test_transformer_autoencoder_supports_pre_tokenized_3d_input():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(18, 4, 3))
    result = representation.transformer_autoencode(
        X,
        model_config=_small_model(token_width=1, max_tokens=4),
        training_config=_small_training(epochs=2, scaling="standard"),
    )
    assert result.latent.shape == (18, 3)
    assert result.reconstruction.shape == X.shape
    assert result.metadata["input_shape"] == [18, 4, 3]
    assert result.metadata["token_count"] == 4
    assert result.metadata["token_dim"] == 3
    assert np.isfinite(result.metrics["mae"])


def test_representation_service_runs_transformer_payload():
    X = np.arange(60, dtype=float).reshape(12, 5)
    response = execute_task(
        "representation",
        {
            "X": X.tolist(),
            "method": "transformer_autoencoder",
            "model_config": {
                "token_width": 2,
                "model_dim": 8,
                "latent_dim": 2,
                "nhead": 2,
                "encoder_layers": 1,
                "decoder_layers": 1,
                "feedforward_dim": 16,
                "max_tokens": 4,
            },
            "training_config": {
                "epochs": 1,
                "batch_size": 6,
                "validation_fraction": 0.0,
                "patience": None,
                "scaling": "standard",
                "random_state": 2,
                "device": "cpu",
            },
        },
    )
    result = response["result"]
    assert result["method"] == "transformer_autoencoder"
    assert result["latent_shape"] == [12, 2]
    assert result["reconstruction_shape"] == [12, 5]
