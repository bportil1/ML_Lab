from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "The experimental Transformer autoencoder requires PyTorch. "
            "Install ML Lab with the experimental-neural extra or install torch separately."
        ) from exc
    return torch, nn


@dataclass(frozen=True, slots=True)
class TransformerAutoencoderConfig:
    token_dim: int
    model_dim: int = 128
    latent_dim: int = 64
    nhead: int = 4
    encoder_layers: int = 2
    decoder_layers: int = 2
    feedforward_dim: int = 256
    dropout: float = 0.0
    max_tokens: int = 1024

    def validate(self) -> None:
        if self.token_dim <= 0 or self.model_dim <= 0 or self.latent_dim <= 0:
            raise ValueError("token_dim, model_dim, and latent_dim must be positive")
        if self.model_dim % self.nhead != 0:
            raise ValueError("model_dim must be divisible by nhead")
        if self.encoder_layers <= 0 or self.decoder_layers <= 0:
            raise ValueError("encoder_layers and decoder_layers must be positive")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def build_transformer_autoencoder(config: TransformerAutoencoderConfig):
    """Build the cleaned experimental successor to the legacy TransformerAE.

    Unlike the original prototype, this model does not own an optimizer, does
    not override ``nn.Module.train()``, and accepts pre-tokenized tensors only.
    """
    config.validate()
    torch, nn = _require_torch()

    class ExperimentalTransformerAutoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.input_projection = nn.Linear(config.token_dim, config.model_dim)
            self.position = nn.Parameter(torch.zeros(1, config.max_tokens, config.model_dim))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.model_dim,
                nhead=config.nhead,
                dim_feedforward=config.feedforward_dim,
                dropout=config.dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.encoder_layers)
            self.pool = nn.Linear(config.model_dim, 1)
            self.to_latent = nn.Linear(config.model_dim, config.latent_dim)
            self.from_latent = nn.Linear(config.latent_dim, config.model_dim)
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=config.model_dim,
                nhead=config.nhead,
                dim_feedforward=config.feedforward_dim,
                dropout=config.dropout,
                batch_first=True,
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.decoder_layers)
            self.output_projection = nn.Linear(config.model_dim, config.token_dim)

        def _embed(self, x):
            if x.ndim != 3:
                raise ValueError("expected token tensor with shape (batch, tokens, token_dim)")
            if x.shape[-1] != config.token_dim:
                raise ValueError(f"expected token_dim={config.token_dim}, got {x.shape[-1]}")
            if x.shape[1] > config.max_tokens:
                raise ValueError(f"token sequence exceeds max_tokens={config.max_tokens}")
            return self.input_projection(x) + self.position[:, : x.shape[1]]

        def encode(self, x):
            encoded = self.encoder(self._embed(x))
            weights = torch.softmax(self.pool(encoded), dim=1)
            pooled = torch.sum(weights * encoded, dim=1)
            return self.to_latent(pooled)

        def decode(self, z, *, token_count: int):
            if token_count <= 0 or token_count > config.max_tokens:
                raise ValueError("token_count is outside the configured range")
            memory = self.from_latent(z).unsqueeze(1)
            target = self.position[:, :token_count].expand(z.shape[0], -1, -1)
            decoded = self.decoder(tgt=target, memory=memory)
            return self.output_projection(decoded)

        def forward(self, x):
            z = self.encode(x)
            return self.decode(z, token_count=x.shape[1])

    return ExperimentalTransformerAutoencoder()


def train_array_autoencoder(
    X: np.ndarray,
    *,
    config: TransformerAutoencoderConfig | None = None,
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    seed: int = 42,
    device: str | None = None,
) -> dict[str, Any]:
    """Train the experimental model on a numeric token tensor.

    This is intentionally a small incubator trainer.  A future stable
    ``ml_lab.neural`` trainer will replace it before the model can graduate.
    """
    torch, _ = _require_torch()
    values = np.asarray(X, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("X must have shape (samples, tokens, token_dim)")
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")

    if config is None:
        config = TransformerAutoencoderConfig(token_dim=int(values.shape[-1]), max_tokens=int(values.shape[1]))
    config.validate()
    if values.shape[-1] != config.token_dim:
        raise ValueError("X token dimension does not match config.token_dim")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_transformer_autoencoder(config).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()
    tensor = torch.from_numpy(values)
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(tensor, batch_size=batch_size, shuffle=True, generator=generator)

    history: list[float] = []
    model.train()
    for _ in range(epochs):
        losses: list[float] = []
        for batch in loader:
            batch = batch.to(resolved_device)
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))

    model.eval()
    with torch.no_grad():
        latent = model.encode(tensor.to(resolved_device)).detach().cpu().numpy()

    return {
        "model": model,
        "latent": latent,
        "history": history,
        "config": config.to_record(),
        "device": str(resolved_device),
    }
