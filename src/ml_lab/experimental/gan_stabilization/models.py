from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from ml_lab.generative.gan.config import GANModelConfig
from ml_lab.generative.gan.models import build_discriminator
from ml_lab.neural.backend import require_torch


class _Unavailable:  # pragma: no cover - marker only
    pass


def build_attention_similarity_critic(input_dim: int, *, token_width: int, model_dim: int, heads: int, embedding_dim: int, max_tokens: int = 1024):
    torch = require_torch(purpose="experimental GAN attention similarity critic")
    from torch import nn

    token_count = int(math.ceil(input_dim / token_width))
    if token_count > max_tokens:
        raise ValueError(f"input requires {token_count} tokens but max_tokens={max_tokens}")
    padded_dim = token_count * token_width

    class AttentionSimilarityCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.token_count = token_count
            self.padded_dim = padded_dim
            self.input_projection = nn.Linear(token_width, model_dim)
            self.position_embeddings = nn.Parameter(torch.empty(1, max_tokens, model_dim))
            self.attention = nn.MultiheadAttention(model_dim, heads, batch_first=True)
            self.norm = nn.LayerNorm(model_dim)
            self.projection = nn.Linear(model_dim, embedding_dim)
            nn.init.normal_(self.position_embeddings, mean=0.0, std=0.02)

        def _tokens(self, x: Any):
            if x.ndim != 2 or int(x.shape[-1]) != self.input_dim:
                raise ValueError(f"expected feature tensor with shape (batch, {self.input_dim})")
            if self.padded_dim != self.input_dim:
                padded = x.new_zeros((x.shape[0], self.padded_dim))
                padded[:, : self.input_dim] = x
                x = padded
            return x.reshape(x.shape[0], self.token_count, token_width)

        def encode(self, x: Any, *, return_attention: bool = False):
            tokens = self._tokens(x)
            embedded = self.input_projection(tokens) + self.position_embeddings[:, : self.token_count]
            attended, weights = self.attention(embedded, embedded, embedded, need_weights=True, average_attn_weights=False)
            pooled = self.norm(attended).mean(dim=1)
            embedding = self.projection(pooled)
            if return_attention:
                return embedding, pooled, weights
            return embedding

        def forward(self, x: Any):
            return self.encode(x)

    return AttentionSimilarityCritic()


def build_latent_conditioner(context_dim: int, latent_dim: int):
    require_torch(purpose="experimental GAN attention conditioning")
    from torch import nn

    class LatentConditioner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Sequential(nn.Linear(context_dim, latent_dim), nn.Tanh())

        def forward(self, z: Any, context: Any, *, strength: float):
            if context.ndim == 1:
                context = context.unsqueeze(0)
            if context.shape[0] == 1 and z.shape[0] != 1:
                context = context.expand(z.shape[0], -1)
            return z + float(strength) * self.projection(context)

    return LatentConditioner()


def build_coupled_discriminator(input_dim: int, config: GANModelConfig):
    require_torch(purpose="experimental coupled GAN discriminator")
    from torch import nn

    mlp = build_discriminator(input_dim, replace(config, discriminator_type="mlp"))
    transformer = build_discriminator(input_dim, replace(config, discriminator_type="transformer"))

    class CoupledDiscriminator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = mlp
            self.transformer = transformer
            self.blend = 0.0

        def set_blend(self, value: float) -> None:
            self.blend = max(0.0, min(1.0, float(value)))

        def forward(self, x: Any):
            return (1.0 - self.blend) * self.mlp(x) + self.blend * self.transformer(x)

    return CoupledDiscriminator()
