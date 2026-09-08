from __future__ import annotations

import numpy as np
import pytest


torch = pytest.importorskip("torch")


def test_transformer_autoencoder_builds_and_encodes():
    from ml_lab.experimental.linux_binary_identification.transformer_autoencoder import (
        TransformerAutoencoderConfig,
        build_transformer_autoencoder,
    )

    config = TransformerAutoencoderConfig(
        token_dim=6,
        model_dim=8,
        latent_dim=4,
        nhead=2,
        encoder_layers=1,
        decoder_layers=1,
        feedforward_dim=16,
        max_tokens=5,
    )
    model = build_transformer_autoencoder(config)
    x = torch.rand(2, 5, 6)
    reconstruction = model(x)
    latent = model.encode(x)
    assert tuple(reconstruction.shape) == (2, 5, 6)
    assert tuple(latent.shape) == (2, 4)


def test_max_clique_dqn_returns_valid_clique():
    from ml_lab.experimental.max_clique_rl.dqn import DQNConfig, train_dqn_clique

    graph = np.array(
        [
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=int,
    )
    result = train_dqn_clique(
        graph,
        config=DQNConfig(episodes=4, hidden_dim=8, epsilon=0.2, epsilon_min=0.1, seed=3),
        device="cpu",
    )
    assert result["valid"] is True
    assert result["clique_size"] >= 1
