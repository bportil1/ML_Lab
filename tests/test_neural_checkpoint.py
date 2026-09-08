from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from ml_lab import representation
from ml_lab.neural import BestModelCheckpoint, load_checkpoint


def test_best_model_checkpoint_round_trip(tmp_path: Path):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "best.pt"
    checkpoint = BestModelCheckpoint("loss", min_delta=0.0, path=checkpoint_path)
    original = {key: value.detach().clone() for key, value in model.state_dict().items()}
    assert checkpoint.update(model, 1.0, 1, optimizer=optimizer)
    assert checkpoint_path.exists()

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10.0)
    checkpoint.restore(model)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key])

    fresh = torch.nn.Linear(3, 2)
    payload = load_checkpoint(checkpoint_path, fresh)
    assert payload["best_epoch"] == 1
    for key, value in fresh.state_dict().items():
        torch.testing.assert_close(value, original[key])


def test_autoencoder_uses_shared_callbacks_and_checkpointing(tmp_path: Path):
    class Recorder:
        def __init__(self):
            self.events = []

        def on_train_begin(self, context):
            self.events.append(("begin", context["monitor"]))

        def on_epoch_end(self, epoch, metrics, context):
            self.events.append(("epoch", epoch, metrics["train_mse"]))

        def on_train_end(self, context):
            self.events.append(("end", context["result"].best_epoch))

    rng = np.random.default_rng(4)
    X = rng.normal(size=(36, 5))
    recorder = Recorder()
    checkpoint_path = tmp_path / "autoencoder-best.pt"
    result = representation.autoencode(
        X,
        model_config=representation.MLPAutoencoderConfig(hidden_dims=(7,), latent_dim=2),
        training_config=representation.AutoencoderTrainingConfig(
            epochs=6,
            batch_size=12,
            validation_fraction=0.2,
            patience=None,
            random_state=5,
            device="cpu",
            checkpoint_path=str(checkpoint_path),
        ),
        callbacks=[recorder],
    )
    assert checkpoint_path.exists()
    assert recorder.events[0][0] == "begin"
    assert len([event for event in recorder.events if event[0] == "epoch"]) == 6
    assert recorder.events[-1][0] == "end"
    assert result.metadata["best_epoch"] is not None
    assert result.metadata["checkpoint_path"] == str(checkpoint_path)
    assert result.metadata["deterministic"] is True
