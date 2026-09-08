from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


def test_energy_training_discovery_is_torch_lazy():
    code = (
        "import sys; from ml_lab import energy_based; "
        "assert 'torch' not in sys.modules; "
        "assert [s.id for s in energy_based.training.list_schemes()] == ['cdk']"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_cdk_config_nested_mapping_and_validation():
    from ml_lab.energy_based.training import CDKTrainingConfig

    config = CDKTrainingConfig.from_mapping({
        "epochs": 3,
        "model_selection": {"enabled": True, "patience": 2},
        "partition_monitoring": {"enabled": True, "schedule": "final_only"},
    })
    assert config.epochs == 3
    assert config.model_selection.enabled is True
    assert config.model_selection.patience == 2
    assert config.partition_monitoring.enabled is True
    with pytest.raises(ValueError):
        CDKTrainingConfig(batch_size=0)


@pytest.mark.parametrize("family", ["bernoulli", "gaussian", "student_t_poe"])
def test_cdk_trains_all_stable_rbm_families(family):
    torch = pytest.importorskip("torch")
    from ml_lab import energy_based
    from ml_lab.energy_based.training import CDKTrainingConfig, DistributionMonitoringConfig

    rng = np.random.default_rng(4)
    X = rng.random((18, 4), dtype=np.float32) if family == "bernoulli" else rng.normal(size=(18, 4)).astype(np.float32)
    model = energy_based.rbm.create(family, 4, 3)
    initial = model.W.detach().clone()
    result = energy_based.training.fit(
        model,
        X,
        config=CDKTrainingConfig(
            epochs=2,
            batch_size=7,
            gibbs_steps=1,
            random_state=9,
            device="cpu",
            distribution_monitoring=DistributionMonitoringConfig(
                enabled=True,
                interval=1,
                sample_size=8,
                gibbs_steps=1,
                sinkhorn_iterations=5,
            ),
        ),
    )
    assert result.stop_reason == "epochs_completed"
    assert len(result.history) == 2
    assert not torch.allclose(initial, model.W)
    assert np.isfinite(result.metrics["reconstruction_error"])
    assert np.isfinite(result.metrics["energy_gap"])
    assert result.generated_samples.shape[1] == 4
    assert result.metadata["gradient_method"] == "autograd_cd_energy_difference"


def test_batching_never_drops_entire_tiny_dataset():
    pytest.importorskip("torch")
    from ml_lab import energy_based
    from ml_lab.energy_based.training import CDKTrainingConfig, DistributionMonitoringConfig

    X = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    result = energy_based.training.run(
        X,
        family="bernoulli",
        hidden_dim=2,
        training_config=CDKTrainingConfig(
            epochs=1,
            batch_size=128,
            drop_last_batch=True,
            device="cpu",
            distribution_monitoring=DistributionMonitoringConfig(enabled=False),
        ),
    )
    batching = result.metadata["batching"]
    assert batching["effective_batch_size"] == 2
    assert batching["effective_drop_last"] is True
    assert batching["batch_count"] == 1


def test_persistent_chains_survive_full_and_tail_batch_sizes():
    pytest.importorskip("torch")
    from ml_lab import energy_based
    from ml_lab.energy_based.training import CDKTrainingConfig, DistributionMonitoringConfig

    X = np.random.default_rng(7).random((10, 3), dtype=np.float32)
    result = energy_based.training.run(
        X,
        family="bernoulli",
        hidden_dim=2,
        training_config=CDKTrainingConfig(
            epochs=3,
            batch_size=4,
            persistent=True,
            device="cpu",
            distribution_monitoring=DistributionMonitoringConfig(enabled=False),
        ),
    )
    assert len(result.history) == 3
    assert all(record["batch_count"] == 3 for record in result.history)


def test_monitoring_does_not_change_training_rng_trajectory():
    torch = pytest.importorskip("torch")
    from ml_lab import energy_based
    from ml_lab.energy_based.training import CDKTrainingConfig, DistributionMonitoringConfig

    X = np.random.default_rng(11).random((20, 4), dtype=np.float32)
    base_kwargs = dict(epochs=3, batch_size=5, gibbs_steps=2, random_state=101, device="cpu")

    a = energy_based.training.run(
        X,
        family="bernoulli",
        hidden_dim=3,
        training_config=CDKTrainingConfig(
            **base_kwargs,
            distribution_monitoring=DistributionMonitoringConfig(enabled=False),
        ),
    )
    b = energy_based.training.run(
        X,
        family="bernoulli",
        hidden_dim=3,
        training_config=CDKTrainingConfig(
            **base_kwargs,
            distribution_monitoring=DistributionMonitoringConfig(
                enabled=True,
                interval=1,
                sample_size=10,
                gibbs_steps=3,
                sinkhorn_iterations=5,
            ),
        ),
    )
    assert torch.allclose(a.model.W, b.model.W)
    assert torch.allclose(a.model.v_bias, b.model.v_bias)
    assert torch.allclose(a.model.h_bias, b.model.h_bias)


def test_exact_bernoulli_partition_matches_direct_enumeration():
    torch = pytest.importorskip("torch")
    from ml_lab import energy_based
    from ml_lab.energy_based.training.config import PartitionMonitoringConfig
    from ml_lab.energy_based.training.diagnostics import estimate_log_partition

    model = energy_based.rbm.create("bernoulli", 4, 2)
    estimate = estimate_log_partition(
        model,
        PartitionMonitoringConfig(
            enabled=True,
            estimator="auto",
            exact_bernoulli_visible_limit=4,
        ),
    )
    states = torch.tensor(list(itertools.product([0.0, 1.0], repeat=4)))
    direct = float(torch.logsumexp(-model.energy(states), dim=0).detach())
    assert estimate.method == "exact_bernoulli_enumeration"
    assert estimate.log_z == pytest.approx(direct)


def test_partition_checkpoint_records_likelihood_and_method():
    pytest.importorskip("torch")
    from ml_lab import energy_based
    from ml_lab.energy_based.training import CDKTrainingConfig, DistributionMonitoringConfig, PartitionMonitoringConfig

    X = np.random.default_rng(21).integers(0, 2, size=(12, 3)).astype(np.float32)
    result = energy_based.training.run(
        X,
        family="bernoulli",
        hidden_dim=2,
        training_config=CDKTrainingConfig(
            epochs=2,
            batch_size=6,
            device="cpu",
            distribution_monitoring=DistributionMonitoringConfig(enabled=False),
            partition_monitoring=PartitionMonitoringConfig(
                enabled=True,
                schedule="final_only",
                exact_bernoulli_visible_limit=4,
            ),
        ),
    )
    assert result.history[-1]["partition_checkpoint"] is True
    assert result.history[-1]["partition_method"] == "exact_bernoulli_enumeration"
    assert np.isfinite(result.metrics["log_partition_estimate"])
    assert np.isfinite(result.metrics["mean_log_likelihood_estimate"])



def test_autograd_cd_updates_continuous_family_shape_parameters():
    torch = pytest.importorskip("torch")
    from ml_lab import energy_based
    from ml_lab.energy_based.training.objective import contrastive_divergence_objective

    for family, parameter_name in (("gaussian", "log_sigma"), ("student_t_poe", "log_nu")):
        torch.manual_seed(13)
        model = energy_based.rbm.create(family, 3, 2)
        positive = torch.randn(8, 3)
        negative = torch.randn(8, 3) + 0.5
        loss = contrastive_divergence_objective(model, positive, negative)
        loss.backward()
        gradient = getattr(model, parameter_name).grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


def test_relative_best_model_tracker_restores_snapshot():
    torch = pytest.importorskip("torch")
    from ml_lab.energy_based.training import BestModelSelectionConfig
    from ml_lab.energy_based.training.model_selection import BestModelTracker

    model = torch.nn.Linear(1, 1, bias=False)
    tracker = BestModelTracker(BestModelSelectionConfig(enabled=True, patience=2, restore_best=True))
    with torch.no_grad():
        model.weight.fill_(1.0)
    tracker.update(model, metric=2.0, epoch=0)
    with torch.no_grad():
        model.weight.fill_(2.0)
    tracker.update(model, metric=1.0, epoch=1)
    with torch.no_grad():
        model.weight.fill_(9.0)
    tracker.restore(model)
    assert model.weight.item() == pytest.approx(2.0)

def test_rbm_service_task_runs_headlessly():
    pytest.importorskip("torch")
    from ml_lab.flask_adapter.service import execute_task

    payload = {
        "X": np.random.default_rng(5).random((12, 3)).tolist(),
        "family": "bernoulli",
        "hidden_dim": 2,
        "training_config": {
            "epochs": 1,
            "batch_size": 6,
            "device": "cpu",
            "distribution_monitoring": {"enabled": False},
        },
    }
    output = execute_task("rbm", payload)
    assert output["task"] == "rbm"
    assert output["result"]["scheme"] == "cdk"
    assert output["result"]["metadata"]["model"]["family"] == "bernoulli"


def test_rbm_train_cli_writes_artifacts(tmp_path):
    pytest.importorskip("torch")
    data = pd.DataFrame(np.random.default_rng(3).random((12, 3)), columns=["a", "b", "c"])
    csv = tmp_path / "data.csv"
    out = tmp_path / "out"
    data.to_csv(csv, index=False)
    env = dict(os.environ)
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml_lab",
            "rbm-train",
            str(csv),
            "--family",
            "bernoulli",
            "--hidden-dim",
            "2",
            "--epochs",
            "1",
            "--batch-size",
            "6",
            "--device",
            "cpu",
            "--no-distribution-metrics",
            "--output",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert "Results written to" in completed.stdout
    for name in ("result.json", "training_history.csv", "model.pt"):
        assert (out / name).exists()
    record = json.loads((out / "result.json").read_text())
    assert record["scheme"] == "cdk"
