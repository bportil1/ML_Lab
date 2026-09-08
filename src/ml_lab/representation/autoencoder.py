from __future__ import annotations

from .config import MLPAutoencoderConfig


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - exercised in environments without neural extras
        raise RuntimeError("MLP autoencoder support requires: pip install 'ml-lab[neural]'") from exc
    return torch, nn


def build_mlp_autoencoder(input_dim: int, config: MLPAutoencoderConfig | None = None):
    """Build a configurable symmetric MLP autoencoder without owning a training loop."""
    if input_dim < 1:
        raise ValueError("input_dim must be positive")
    config = config or MLPAutoencoderConfig()
    torch, nn = _torch()

    activation_factories = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.01),
    }
    output_factories = {
        "none": nn.Identity,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
    }

    def dense_stack(dims: list[int], *, final_activation: bool = True):
        layers = []
        for index, (left, right) in enumerate(zip(dims, dims[1:])):
            layers.append(nn.Linear(left, right))
            is_last = index == len(dims) - 2
            if not is_last or final_activation:
                layers.append(activation_factories[config.activation]())
                if config.dropout > 0 and not is_last:
                    layers.append(nn.Dropout(config.dropout))
        return nn.Sequential(*layers)

    class MLPAutoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            encoder_dims = [input_dim, *config.hidden_dims, config.latent_dim]
            decoder_dims = [config.latent_dim, *reversed(config.hidden_dims), input_dim]
            self.encoder = dense_stack(encoder_dims, final_activation=False)
            decoder_layers = list(dense_stack(decoder_dims, final_activation=False).children())
            decoder_layers.append(output_factories[config.output_activation]())
            self.decoder = nn.Sequential(*decoder_layers)
            self.input_dim = input_dim
            self.latent_dim = config.latent_dim
            self.model_config = config

        def encode(self, x):
            return self.encoder(x)

        def decode(self, z):
            return self.decoder(z)

        def forward(self, x):
            return self.decode(self.encode(x))

    return MLPAutoencoder()
