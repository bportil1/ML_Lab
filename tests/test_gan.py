from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from ml_lab import generative
from ml_lab.generative import gan
from ml_lab.generative.gan.evaluation import distribution_metrics
from ml_lab.generative.gan.reporting import save_result


def _data(rows: int = 24, features: int = 5) -> pd.DataFrame:
    values = np.random.default_rng(4).normal(size=(rows, features))
    return pd.DataFrame(values, columns=[f"f{i}" for i in range(features)])


def test_generative_package_is_torch_lazy():
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src")
    code = "import sys; import ml_lab.generative; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_gan_config_rejects_ignored_validation_split():
    with pytest.raises(ValueError, match="validation_fraction"):
        gan.GANTrainingConfig(validation_fraction=0.2)


def test_distribution_metrics_are_zero_for_identical_samples():
    values = np.arange(20, dtype=float).reshape(10, 2)
    metrics = distribution_metrics(values, values.copy())
    assert metrics["featurewise_wasserstein_mean"] == pytest.approx(0.0)
    assert metrics["mean_absolute_mean_shift"] == pytest.approx(0.0)
    assert metrics["mean_absolute_std_shift"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("generator_type", "discriminator_type"),
    [
        ("mlp", "mlp"),
        ("transformer", "mlp"),
        ("mlp", "transformer"),
        ("transformer", "transformer"),
    ],
)
def test_all_stable_gan_architecture_combinations_train(generator_type, discriminator_type):
    pytest.importorskip("torch")
    result = generative.gan.run(
        _data(),
        model_config=gan.GANModelConfig(
            latent_dim=4,
            generator_type=generator_type,
            discriminator_type=discriminator_type,
            generator_hidden_dims=(8,),
            discriminator_hidden_dims=(8,),
            token_width=2,
            model_dim=8,
            nhead=2,
            transformer_layers=1,
            feedforward_dim=16,
        ),
        training_config=gan.GANTrainingConfig(
            epochs=1,
            batch_size=8,
            generated_sample_count=7,
            device="cpu",
            random_state=3,
        ),
    )
    assert result.generated_samples.shape == (7, 5)
    assert result.metadata["generator_type"] == generator_type
    assert result.metadata["discriminator_type"] == discriminator_type
    assert len(result.history) == 1
    assert np.isfinite(result.generated_samples).all()
    assert np.isfinite(result.metrics["featurewise_wasserstein_mean"])


def test_gan_seed_is_reproducible_on_cpu():
    pytest.importorskip("torch")
    kwargs = dict(
        model_config=gan.GANModelConfig(
            latent_dim=3,
            generator_hidden_dims=(8,),
            discriminator_hidden_dims=(8,),
        ),
        training_config=gan.GANTrainingConfig(
            epochs=2,
            batch_size=8,
            generated_sample_count=6,
            device="cpu",
            random_state=11,
        ),
    )
    first = gan.run(_data(), **kwargs)
    second = gan.run(_data(), **kwargs)
    np.testing.assert_allclose(first.generated_samples, second.generated_samples, atol=1e-7)


def test_gan_reporting_writes_headless_artifacts(tmp_path):
    pytest.importorskip("torch")
    result = gan.run(
        _data(),
        model_config=gan.GANModelConfig(
            latent_dim=3,
            generator_hidden_dims=(8,),
            discriminator_hidden_dims=(8,),
        ),
        training_config=gan.GANTrainingConfig(
            epochs=1,
            batch_size=8,
            generated_sample_count=5,
            device="cpu",
        ),
    )
    output = save_result(result, tmp_path / "gan")
    expected = {
        "result.json",
        "generated_samples.csv",
        "training_history.csv",
        "generator.pt",
        "discriminator.pt",
        "scaler.joblib",
    }
    assert expected.issubset({path.name for path in output.iterdir()})


def test_gan_result_can_generate_additional_samples():
    pytest.importorskip("torch")
    result = gan.run(
        _data(),
        model_config=gan.GANModelConfig(
            latent_dim=3,
            generator_hidden_dims=(8,),
            discriminator_hidden_dims=(8,),
        ),
        training_config=gan.GANTrainingConfig(epochs=1, batch_size=8, device="cpu"),
    )
    samples = gan.sample(result, 4, random_state=17)
    assert samples.shape == (4, 5)
    assert np.isfinite(samples).all()
