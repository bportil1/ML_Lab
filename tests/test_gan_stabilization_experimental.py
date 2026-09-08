from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from ml_lab.experimental import get_manifest, run_experiment


def _data(rows: int = 18, features: int = 5):
    return np.random.default_rng(7).normal(size=(rows, features)).astype(np.float32)


def test_gan_stabilization_manifest_is_lazy():
    manifest = get_manifest("gan_stabilization")
    assert "tangent_negatives" in manifest.capabilities
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src")
    code = "import sys; from ml_lab.experimental import get_manifest; get_manifest('gan_stabilization'); assert 'ml_lab.experimental.gan_stabilization' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_describe_documents_corrected_mechanisms():
    record = run_experiment("gan_stabilization", mode="describe")
    assert "score_tangent_fake_negatives" in record["mechanisms"]
    assert any("not guaranteed data-manifold" in warning for warning in record["warnings"])


def test_score_tangent_negatives_are_first_order_orthogonal():
    torch = pytest.importorskip("torch")
    from ml_lab.experimental.gan_stabilization.tangent import score_tangent_negatives

    class LinearScore(torch.nn.Module):
        def forward(self, x):
            return 2.0 * x[:, 0] - x[:, 1] + 0.5 * x[:, 2]

    x = torch.randn(4, 3)
    tangents = score_tangent_negatives(LinearScore(), x, epsilon=0.1, count=1)
    delta = tangents - x
    normal = torch.tensor([2.0, -1.0, 0.5])
    directional = delta @ normal
    assert torch.max(torch.abs(directional)).item() < 1e-5


def test_experimental_stabilized_gan_trains_with_all_mechanisms():
    pytest.importorskip("torch")
    record = run_experiment(
        "gan_stabilization",
        mode="fit_generate",
        X=_data().tolist(),
        model_config={
            "latent_dim": 4,
            "generator_type": "transformer",
            "discriminator_type": "mlp",
            "generator_hidden_dims": [8],
            "discriminator_hidden_dims": [8],
            "token_width": 2,
            "model_dim": 8,
            "nhead": 2,
            "transformer_layers": 1,
            "feedforward_dim": 16,
        },
        training_config={
            "epochs": 1,
            "batch_size": 6,
            "device": "cpu",
            "generated_sample_count": 5,
            "similarity_critic_enabled": True,
            "similarity_model_dim": 8,
            "similarity_embedding_dim": 4,
            "similarity_heads": 2,
            "feature_matching_weight": 0.05,
            "attention_conditioning_strength": 0.1,
            "tangent_negative_weight": 0.1,
            "tangent_epsilon": 0.02,
            "coupled_discriminator": True,
            "transformer_blend_start": 0.0,
            "transformer_blend_end": 0.5,
            "transformer_blend_warmup_epochs": 1,
            "random_state": 9,
        },
    )
    generated = np.asarray(record["generated_samples"])
    assert generated.shape == (5, 5)
    assert np.isfinite(generated).all()
    history = record["history"]
    assert len(history) == 1
    for key in ("similarity_critic_loss", "feature_matching_loss", "tangent_negative_loss", "attention_conditioning_strength", "transformer_discriminator_blend"):
        assert key in history[0]
        assert np.isfinite(float(history[0][key]))
