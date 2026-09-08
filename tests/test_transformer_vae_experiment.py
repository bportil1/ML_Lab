from __future__ import annotations

import sys

import numpy as np
import pytest

from ml_lab.experimental import get_manifest, list_experiments, run_experiment


def test_transformer_vae_manifest_is_lazy():
    module_name = "ml_lab.experimental.transformer_vae"
    sys.modules.pop(module_name, None)
    ids = [manifest.id for manifest in list_experiments()]
    assert "transformer_vae" in ids
    assert module_name not in sys.modules
    manifest = get_manifest("transformer_vae")
    assert "variational_autoencoder" in manifest.capabilities
    assert module_name not in sys.modules


def test_transformer_vae_description_does_not_require_training():
    description = run_experiment("transformer_vae")
    assert description["experiment"] == "transformer_vae"
    assert "fit_transform" in description["available_modes"]


def test_transformer_vae_config_validation():
    pytest.importorskip("torch")
    from ml_lab.experimental.transformer_vae import (
        TransformerVAEConfig,
        TransformerVAETrainingConfig,
    )

    with pytest.raises(ValueError):
        TransformerVAEConfig(model_dim=7, nhead=2)
    with pytest.raises(ValueError):
        TransformerVAETrainingConfig(beta=-1.0)


def test_transformer_vae_train_and_prior_samples():
    torch = pytest.importorskip("torch")
    from ml_lab.experimental.transformer_vae import (
        TransformerVAEConfig,
        TransformerVAETrainingConfig,
        build_transformer_vae,
        train_transformer_vae,
    )

    model = build_transformer_vae(
        2,
        TransformerVAEConfig(
            model_dim=8,
            latent_dim=3,
            nhead=2,
            encoder_layers=1,
            decoder_layers=1,
            feedforward_dim=16,
            max_tokens=4,
        ),
    )
    tokens = torch.rand(2, 3, 2)
    reconstruction, mu, logvar, sampled = model(tokens, sample=False)
    assert tuple(reconstruction.shape) == (2, 3, 2)
    assert tuple(mu.shape) == (2, 3)
    assert tuple(logvar.shape) == (2, 3)
    assert torch.allclose(sampled, mu)

    rng = np.random.default_rng(4)
    X = rng.normal(size=(10, 6)).astype(np.float32)
    result = train_transformer_vae(
        X,
        model_config=TransformerVAEConfig(
            token_width=2,
            model_dim=8,
            latent_dim=3,
            nhead=2,
            encoder_layers=1,
            decoder_layers=1,
            feedforward_dim=16,
            max_tokens=4,
        ),
        training_config=TransformerVAETrainingConfig(
            epochs=2,
            batch_size=5,
            validation_fraction=0.2,
            patience=None,
            beta=0.1,
            kl_warmup_epochs=2,
            random_state=9,
            device="cpu",
        ),
        prior_sample_count=3,
    )
    assert result.posterior_mean.shape == (10, 3)
    assert result.posterior_logvar.shape == (10, 3)
    assert result.reconstruction.shape == X.shape
    assert result.prior_samples is not None
    assert result.prior_samples.shape == (3, 6)
    assert np.isfinite(result.metrics["mean_posterior_kl"])
    assert len(result.history) == 2
    assert result.history[0]["beta"] == pytest.approx(0.05)
    assert result.history[1]["beta"] == pytest.approx(0.1)
