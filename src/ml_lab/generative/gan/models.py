from __future__ import annotations

import math
from typing import Any

from ml_lab.neural.backend import require_torch

from .config import GANModelConfig


def _activation(nn: Any, name: str):
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.2)
    raise ValueError(f"unsupported activation: {name}")


def _output_activation(nn: Any, name: str):
    if name == "none":
        return nn.Identity()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"unsupported output activation: {name}")


def _mlp_generator(output_dim: int, config: GANModelConfig):
    torch = require_torch(purpose="GAN model construction")
    from torch import nn

    layers: list[Any] = []
    current = config.latent_dim
    for width in config.generator_hidden_dims:
        layers.extend([nn.Linear(current, int(width)), _activation(nn, config.generator_activation)])
        if config.dropout:
            layers.append(nn.Dropout(config.dropout))
        current = int(width)
    layers.extend([nn.Linear(current, output_dim), _output_activation(nn, config.output_activation)])

    class MLPGenerator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.latent_dim = config.latent_dim
            self.output_dim = output_dim
            self.network = nn.Sequential(*layers)

        def forward(self, z: Any):
            if z.ndim != 2 or int(z.shape[-1]) != self.latent_dim:
                raise ValueError(f"expected latent tensor with shape (batch, {self.latent_dim})")
            return self.network(z)

    return MLPGenerator()


def _mlp_discriminator(input_dim: int, config: GANModelConfig):
    require_torch(purpose="GAN model construction")
    from torch import nn

    layers: list[Any] = []
    current = input_dim
    for width in config.discriminator_hidden_dims:
        layers.extend([nn.Linear(current, int(width)), _activation(nn, config.discriminator_activation)])
        if config.dropout:
            layers.append(nn.Dropout(config.dropout))
        current = int(width)
    layers.append(nn.Linear(current, 1))

    class MLPDiscriminator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.network = nn.Sequential(*layers)

        def forward(self, x: Any):
            if x.ndim != 2 or int(x.shape[-1]) != self.input_dim:
                raise ValueError(f"expected feature tensor with shape (batch, {self.input_dim})")
            return self.network(x).reshape(-1)

    return MLPDiscriminator()


def _transformer_generator(output_dim: int, config: GANModelConfig):
    torch = require_torch(purpose="Transformer GAN model construction")
    from torch import nn

    token_count = int(math.ceil(output_dim / config.token_width))
    if token_count > config.max_tokens:
        raise ValueError(
            f"output requires {token_count} tokens but max_tokens={config.max_tokens}"
        )

    class TransformerGenerator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.latent_dim = config.latent_dim
            self.output_dim = output_dim
            self.token_count = token_count
            self.token_width = config.token_width
            self.latent_projection = nn.Linear(config.latent_dim, config.model_dim)
            self.queries = nn.Parameter(torch.empty(1, config.max_tokens, config.model_dim))
            self.position_embeddings = nn.Parameter(
                torch.empty(1, config.max_tokens, config.model_dim)
            )
            layer = nn.TransformerEncoderLayer(
                d_model=config.model_dim,
                nhead=config.nhead,
                dim_feedforward=config.feedforward_dim,
                dropout=config.transformer_dropout,
                activation=config.transformer_activation,
                batch_first=True,
                norm_first=config.norm_first,
            )
            self.transformer = nn.TransformerEncoder(
                layer,
                num_layers=config.transformer_layers,
                norm=nn.LayerNorm(config.model_dim),
                enable_nested_tensor=False,
            )
            self.output_projection = nn.Linear(config.model_dim, config.token_width)
            self.output_activation = _output_activation(nn, config.output_activation)
            nn.init.normal_(self.queries, mean=0.0, std=0.02)
            nn.init.normal_(self.position_embeddings, mean=0.0, std=0.02)

        def forward(self, z: Any):
            if z.ndim != 2 or int(z.shape[-1]) != self.latent_dim:
                raise ValueError(f"expected latent tensor with shape (batch, {self.latent_dim})")
            batch = int(z.shape[0])
            condition = self.latent_projection(z).unsqueeze(1)
            tokens = self.queries[:, : self.token_count].expand(batch, -1, -1)
            tokens = tokens + self.position_embeddings[:, : self.token_count] + condition
            encoded = self.transformer(tokens)
            values = self.output_activation(self.output_projection(encoded))
            return values.reshape(batch, -1)[:, : self.output_dim]

    return TransformerGenerator()


def _transformer_discriminator(input_dim: int, config: GANModelConfig):
    torch = require_torch(purpose="Transformer GAN model construction")
    from torch import nn

    token_count = int(math.ceil(input_dim / config.token_width))
    padded_dim = token_count * config.token_width
    if token_count > config.max_tokens:
        raise ValueError(
            f"input requires {token_count} tokens but max_tokens={config.max_tokens}"
        )

    class TransformerDiscriminator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.token_count = token_count
            self.padded_dim = padded_dim
            self.input_projection = nn.Linear(config.token_width, config.model_dim)
            self.position_embeddings = nn.Parameter(
                torch.empty(1, config.max_tokens, config.model_dim)
            )
            layer = nn.TransformerEncoderLayer(
                d_model=config.model_dim,
                nhead=config.nhead,
                dim_feedforward=config.feedforward_dim,
                dropout=config.transformer_dropout,
                activation=config.transformer_activation,
                batch_first=True,
                norm_first=config.norm_first,
            )
            self.transformer = nn.TransformerEncoder(
                layer,
                num_layers=config.transformer_layers,
                norm=nn.LayerNorm(config.model_dim),
                enable_nested_tensor=False,
            )
            self.pool_score = nn.Linear(config.model_dim, 1)
            self.classifier = nn.Linear(config.model_dim, 1)
            nn.init.normal_(self.position_embeddings, mean=0.0, std=0.02)

        def forward(self, x: Any):
            if x.ndim != 2 or int(x.shape[-1]) != self.input_dim:
                raise ValueError(f"expected feature tensor with shape (batch, {self.input_dim})")
            if self.padded_dim != self.input_dim:
                padded = x.new_zeros((x.shape[0], self.padded_dim))
                padded[:, : self.input_dim] = x
                x = padded
            tokens = x.reshape(x.shape[0], self.token_count, config.token_width)
            embedded = self.input_projection(tokens) + self.position_embeddings[:, : self.token_count]
            encoded = self.transformer(embedded)
            scores = self.pool_score(encoded).squeeze(-1)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            pooled = torch.sum(weights * encoded, dim=1)
            return self.classifier(pooled).reshape(-1)

    return TransformerDiscriminator()


def build_generator(output_dim: int, config: GANModelConfig | None = None):
    if output_dim < 1:
        raise ValueError("output_dim must be positive")
    config = config or GANModelConfig()
    if config.generator_type == "mlp":
        return _mlp_generator(output_dim, config)
    if config.generator_type == "transformer":
        return _transformer_generator(output_dim, config)
    raise ValueError(f"unsupported generator_type: {config.generator_type}")


def build_discriminator(input_dim: int, config: GANModelConfig | None = None):
    if input_dim < 1:
        raise ValueError("input_dim must be positive")
    config = config or GANModelConfig()
    if config.discriminator_type == "mlp":
        return _mlp_discriminator(input_dim, config)
    if config.discriminator_type == "transformer":
        return _transformer_discriminator(input_dim, config)
    raise ValueError(f"unsupported discriminator_type: {config.discriminator_type}")


def build_gan_models(feature_dim: int, config: GANModelConfig | None = None):
    config = config or GANModelConfig()
    return build_generator(feature_dim, config), build_discriminator(feature_dim, config)
