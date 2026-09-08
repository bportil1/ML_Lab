from __future__ import annotations

import sys

import numpy as np
import pytest

from ml_lab.experimental import get_manifest, list_experiments, run_experiment


def test_contractive_transformer_vae_manifest_is_lazy():
    module_name = "ml_lab.experimental.transformer_vae_contractive"
    sys.modules.pop(module_name, None)
    ids = [manifest.id for manifest in list_experiments()]
    assert "transformer_vae_contractive" in ids
    assert module_name not in sys.modules
    manifest = get_manifest("transformer_vae_contractive")
    assert "jacobian_penalty" in manifest.capabilities
    assert "tangent_regularization" in manifest.capabilities
    assert module_name not in sys.modules


def test_contractive_description_is_explicit_about_first_order_target():
    description = run_experiment("transformer_vae_contractive")
    assert description["experiment"] == "transformer_vae_contractive"
    assert "d mu(x)/d x" in description["objective"]
    assert description["contractive_target"] == "posterior_mean_jacobian_wrt_input"
    assert set(description["available_estimators"]) == {"exact", "hutchinson"}


def test_exact_contractive_penalty_matches_linear_jacobian():
    torch = pytest.importorskip("torch")
    from ml_lab.experimental.transformer_vae_contractive import contractive_jacobian_penalty

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    weights = torch.tensor([[2.0, -1.0], [0.5, 3.0]], requires_grad=True)
    mu = x @ weights
    penalty = contractive_jacobian_penalty(
        torch,
        mu,
        x,
        estimator="exact",
        normalize_by_active_inputs=False,
        create_graph=True,
    )
    expected = weights.pow(2).sum()
    assert torch.allclose(penalty, expected, atol=1e-6)
    penalty.backward()
    assert weights.grad is not None
    assert torch.isfinite(weights.grad).all()


def test_contractive_config_validates_estimator_controls():
    pytest.importorskip("torch")
    from ml_lab.experimental.transformer_vae_contractive import (
        ContractiveTransformerVAETrainingConfig,
    )

    with pytest.raises(ValueError):
        ContractiveTransformerVAETrainingConfig(contractive_weight=-1.0)
    with pytest.raises(ValueError):
        ContractiveTransformerVAETrainingConfig(contractive_estimator="unknown")
    with pytest.raises(ValueError):
        ContractiveTransformerVAETrainingConfig(hutchinson_samples=0)


def test_contractive_transformer_vae_trains_and_records_first_order_penalty():
    pytest.importorskip("torch")
    from ml_lab.experimental.transformer_vae_contractive import (
        ContractiveTransformerVAETrainingConfig,
        TransformerVAEConfig,
        train_contractive_transformer_vae,
    )

    rng = np.random.default_rng(17)
    X = rng.normal(size=(12, 6)).astype(np.float32)
    result = train_contractive_transformer_vae(
        X,
        model_config=TransformerVAEConfig(
            token_width=2,
            model_dim=8,
            latent_dim=2,
            nhead=2,
            encoder_layers=1,
            decoder_layers=1,
            feedforward_dim=16,
            max_tokens=4,
        ),
        training_config=ContractiveTransformerVAETrainingConfig(
            epochs=2,
            batch_size=4,
            validation_fraction=0.25,
            patience=None,
            beta=0.5,
            contractive_weight=0.1,
            contractive_warmup_epochs=2,
            contractive_estimator="exact",
            random_state=13,
            device="cpu",
        ),
        prior_sample_count=2,
    )
    assert result.posterior_mean.shape == (12, 2)
    assert result.reconstruction.shape == X.shape
    assert result.prior_samples is not None and result.prior_samples.shape == (2, 6)
    assert np.isfinite(result.metrics["mean_contractive_penalty"])
    assert result.metrics["mean_contractive_penalty"] >= 0.0
    assert result.history[0]["contractive_factor"] == pytest.approx(0.5)
    assert result.history[1]["contractive_factor"] == pytest.approx(1.0)
    assert "train_contractive_penalty" in result.history[-1]
    assert np.isfinite(result.history[-1]["train_contractive_penalty"])
    assert result.metadata["contractive_target"] == "posterior_mean_jacobian_wrt_input"
    assert result.metadata["contractive_estimator"] == "exact"
