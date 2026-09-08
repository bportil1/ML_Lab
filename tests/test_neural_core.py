from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from ml_lab.neural import EarlyStopping, NeuralTrainingConfig, TrainingHistory, split_validation_indices


def test_neural_package_is_torch_lazy():
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src")
    code = "import sys; import ml_lab.neural; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_validation_split_is_deterministic_and_disjoint():
    first = split_validation_indices(20, 0.2, random_state=7)
    second = split_validation_indices(20, 0.2, random_state=7)
    np.testing.assert_array_equal(first.train_indices, second.train_indices)
    np.testing.assert_array_equal(first.validation_indices, second.validation_indices)
    assert first.validation_count == 4
    assert set(first.train_indices).isdisjoint(set(first.validation_indices))
    assert sorted(np.concatenate([first.train_indices, first.validation_indices]).tolist()) == list(range(20))


def test_early_stopping_and_history_are_task_neutral():
    history = TrainingHistory()
    stopping = EarlyStopping(patience=2, min_delta=0.01, mode="min")
    values = [1.0, 0.9, 0.905, 0.91]
    stopped = False
    for epoch, value in enumerate(values, start=1):
        history.append(epoch, loss=value)
        _, stopped = stopping.update(value, epoch)
        if stopped:
            break
    assert stopped is True
    assert stopping.best_epoch == 2
    assert history.last["epoch"] == 4


def test_neural_training_config_validates_common_controls():
    config = NeuralTrainingConfig(epochs=5, batch_size=4, device="cpu", checkpoint_path="best.pt")
    assert config.restore_best is True
    with pytest.raises(ValueError):
        NeuralTrainingConfig(epochs=0)
