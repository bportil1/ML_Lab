from __future__ import annotations

import sys

import numpy as np
import pytest

from ml_lab.experimental import get_manifest, list_experiments, run_experiment


def test_decomposed_transformer_vae_manifest_is_lazy():
    module_name = "ml_lab.experimental.transformer_vae_decomposed"
    sys.modules.pop(module_name, None)
    ids = [manifest.id for manifest in list_experiments()]
    assert "transformer_vae_decomposed" in ids
    assert module_name not in sys.modules
    manifest = get_manifest("transformer_vae_decomposed")
    assert "total_correlation" in manifest.capabilities
    assert "dimension_wise_kl" in manifest.capabilities
    assert module_name not in sys.modules


def test_decomposed_description_is_explicit_about_estimator():
    description = run_experiment("transformer_vae_decomposed")
    assert description["experiment"] == "transformer_vae_decomposed"
    assert description["density_estimator"] == "minibatch_mixture"
    assert "ICMI" in description["objective"]


def test_decomposed_config_rejects_singleton_batches_and_negative_weights():
    pytest.importorskip("torch")
    from ml_lab.experimental.transformer_vae_decomposed import (
        DecomposedTransformerVAETrainingConfig,
    )

    with pytest.raises(ValueError):
        DecomposedTransformerVAETrainingConfig(batch_size=1)
    with pytest.raises(ValueError):
        DecomposedTransformerVAETrainingConfig(tc_weight=-1.0)


def test_decomposition_terms_are_finite_differentiable_and_additive():
    torch = pytest.importorskip("torch")
    from ml_lab.experimental.transformer_vae_decomposed import estimate_decomposed_kl

    torch.manual_seed(4)
    mu = (torch.randn(6, 3) * 0.2).requires_grad_()
    logvar = (torch.randn(6, 3) * 0.1).requires_grad_()
    epsilon = torch.randn(6, 3)
    z = mu + torch.exp(0.5 * logvar) * epsilon
    estimate = estimate_decomposed_kl(torch, z, mu, logvar)

    for value in (
        estimate.icmi,
        estimate.total_correlation,
        estimate.dimension_wise_kl,
        estimate.estimated_kl,
        estimate.analytic_kl,
    ):
        assert torch.isfinite(value)
    assert torch.allclose(
        estimate.estimated_kl,
        estimate.icmi + estimate.total_correlation + estimate.dimension_wise_kl,
        atol=1e-6,
    )
    estimate.weighted(icmi_weight=1.0, tc_weight=4.0, dwkl_weight=1.0).backward()
    assert mu.grad is not None
    assert logvar.grad is not None


def test_decomposed_transformer_vae_trains_and_records_terms():
    pytest.importorskip("torch")
    from ml_lab.experimental.transformer_vae_decomposed import (
        DecomposedTransformerVAETrainingConfig,
        TransformerVAEConfig,
        train_decomposed_transformer_vae,
    )

    rng = np.random.default_rng(7)
    X = rng.normal(size=(12, 6)).astype(np.float32)
    result = train_decomposed_transformer_vae(
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
        training_config=DecomposedTransformerVAETrainingConfig(
            epochs=2,
            batch_size=4,
            validation_fraction=0.25,
            patience=None,
            reconstruction_weight=1.0,
            icmi_weight=1.0,
            tc_weight=2.0,
            dwkl_weight=1.0,
            decomposition_warmup_epochs=2,
            random_state=11,
            device="cpu",
        ),
        prior_sample_count=2,
    )
    assert result.posterior_mean.shape == (12, 3)
    assert result.reconstruction.shape == X.shape
    assert result.prior_samples is not None and result.prior_samples.shape == (2, 6)
    for metric in (
        "icmi_estimate",
        "total_correlation_estimate",
        "dimension_wise_kl_estimate",
        "decomposed_kl_estimate",
        "analytic_kl",
    ):
        assert np.isfinite(result.metrics[metric])
    assert len(result.history) == 2
    assert result.history[0]["warmup_factor"] == pytest.approx(0.5)
    assert result.history[1]["warmup_factor"] == pytest.approx(1.0)
    for key in ("train_icmi", "train_tc", "train_dwkl", "train_decomposed_kl"):
        assert key in result.history[-1]
        assert np.isfinite(result.history[-1][key])
    assert result.metadata["density_estimator"] == "minibatch_mixture"
