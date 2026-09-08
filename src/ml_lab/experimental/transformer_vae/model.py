from __future__ import annotations

from typing import Any

from ml_lab.neural.backend import require_torch

from .config import TransformerVAEConfig


def build_transformer_vae(token_dim: int, config: TransformerVAEConfig | None = None):
    """Build a Transformer VAE for generic token tensors.

    The approximate posterior is a diagonal Gaussian q(z|x). ``encode`` returns the
    posterior mean by default so representation extraction is deterministic; callers
    can request a reparameterized sample explicitly.
    """
    if token_dim < 1:
        raise ValueError("token_dim must be positive")
    config = config or TransformerVAEConfig()
    torch = require_torch(purpose="experimental Transformer VAE support")
    from torch import nn

    class TransformerVAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.token_dim = int(token_dim)
            self.input_projection = nn.Linear(token_dim, config.model_dim)
            self.position_embeddings = nn.Parameter(
                torch.empty(1, config.max_tokens, config.model_dim)
            )
            self.decoder_queries = nn.Parameter(
                torch.empty(1, config.max_tokens, config.model_dim)
            )

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.model_dim,
                nhead=config.nhead,
                dim_feedforward=config.feedforward_dim,
                dropout=config.dropout,
                activation=config.transformer_activation,
                batch_first=True,
                norm_first=config.norm_first,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=config.encoder_layers,
                norm=nn.LayerNorm(config.model_dim),
                enable_nested_tensor=False,
            )
            self.pool_score = nn.Linear(config.model_dim, 1)
            self.to_mu = nn.Linear(config.model_dim, config.latent_dim)
            self.to_logvar = nn.Linear(config.model_dim, config.latent_dim)
            self.from_latent = nn.Linear(config.latent_dim, config.model_dim)

            decoder_layer = nn.TransformerDecoderLayer(
                d_model=config.model_dim,
                nhead=config.nhead,
                dim_feedforward=config.feedforward_dim,
                dropout=config.dropout,
                activation=config.transformer_activation,
                batch_first=True,
                norm_first=config.norm_first,
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=config.decoder_layers,
                norm=nn.LayerNorm(config.model_dim),
            )
            self.output_projection = nn.Linear(config.model_dim, token_dim)
            self.output_activation = _output_activation(nn, config.output_activation)
            self.reset_parameters()

        def reset_parameters(self) -> None:
            nn.init.normal_(self.position_embeddings, mean=0.0, std=0.02)
            nn.init.normal_(self.decoder_queries, mean=0.0, std=0.02)

        def _check_tokens(self, x: Any) -> None:
            if x.ndim != 3:
                raise ValueError("expected token tensor with shape (batch, tokens, token_dim)")
            if int(x.shape[-1]) != self.token_dim:
                raise ValueError(f"expected token_dim={self.token_dim}, got {x.shape[-1]}")
            if int(x.shape[1]) < 1 or int(x.shape[1]) > config.max_tokens:
                raise ValueError(f"token count must be in [1, {config.max_tokens}]")

        @staticmethod
        def _token_padding_mask(element_mask: Any | None) -> Any | None:
            if element_mask is None:
                return None
            if element_mask.ndim != 3:
                raise ValueError("element_mask must have shape (batch, tokens, token_dim)")
            return ~element_mask.to(dtype=torch.bool).any(dim=-1)

        def encode_distribution(self, x: Any, *, element_mask: Any | None = None):
            self._check_tokens(x)
            padding_mask = self._token_padding_mask(element_mask)
            embedded = self.input_projection(x) + self.position_embeddings[:, : x.shape[1]]
            encoded = self.encoder(embedded, src_key_padding_mask=padding_mask)
            scores = self.pool_score(encoded).squeeze(-1)
            if padding_mask is not None:
                scores = scores.masked_fill(padding_mask, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            pooled = torch.sum(weights * encoded, dim=1)
            mu = self.to_mu(pooled)
            logvar = self.to_logvar(pooled).clamp(config.logvar_min, config.logvar_max)
            return mu, logvar

        @staticmethod
        def reparameterize(mu: Any, logvar: Any):
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std

        def encode(
            self,
            x: Any,
            *,
            element_mask: Any | None = None,
            sample: bool = False,
        ):
            mu, logvar = self.encode_distribution(x, element_mask=element_mask)
            if sample:
                return self.reparameterize(mu, logvar)
            return mu

        def decode(
            self,
            z: Any,
            *,
            token_count: int,
            element_mask: Any | None = None,
        ):
            if z.ndim != 2 or int(z.shape[-1]) != config.latent_dim:
                raise ValueError(
                    f"expected latent tensor with shape (batch, {config.latent_dim})"
                )
            if token_count < 1 or token_count > config.max_tokens:
                raise ValueError(f"token_count must be in [1, {config.max_tokens}]")
            padding_mask = self._token_padding_mask(element_mask)
            memory = self.from_latent(z).unsqueeze(1)
            target = self.decoder_queries[:, :token_count].expand(z.shape[0], -1, -1)
            decoded = self.decoder(
                tgt=target,
                memory=memory,
                tgt_key_padding_mask=padding_mask,
            )
            return self.output_activation(self.output_projection(decoded))

        def forward(
            self,
            x: Any,
            *,
            element_mask: Any | None = None,
            sample: bool = True,
        ):
            mu, logvar = self.encode_distribution(x, element_mask=element_mask)
            z = self.reparameterize(mu, logvar) if sample else mu
            reconstruction = self.decode(
                z,
                token_count=int(x.shape[1]),
                element_mask=element_mask,
            )
            return reconstruction, mu, logvar, z

        def sample_prior(
            self,
            count: int,
            *,
            token_count: int,
            element_mask: Any | None = None,
            device: Any | None = None,
        ):
            if count < 1:
                raise ValueError("count must be positive")
            parameter = next(self.parameters())
            target_device = device or parameter.device
            z = torch.randn(count, config.latent_dim, device=target_device, dtype=parameter.dtype)
            return self.decode(z, token_count=token_count, element_mask=element_mask)

    return TransformerVAE()


def _output_activation(nn: Any, name: str):
    if name == "none":
        return nn.Identity()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"unsupported output activation: {name}")
