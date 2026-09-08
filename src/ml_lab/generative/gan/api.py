from __future__ import annotations

from typing import Any, Iterable

from .config import GANModelConfig, GANTrainingConfig
from .results import GANResult


def run(
    X: Any,
    *,
    model_config: GANModelConfig | None = None,
    training_config: GANTrainingConfig | None = None,
    callbacks: Iterable[Any] | None = None,
) -> GANResult:
    from .training import train_gan

    return train_gan(
        X,
        model_config=model_config,
        training_config=training_config,
        callbacks=callbacks,
    )


def sample(
    result: GANResult,
    count: int,
    *,
    random_state: int = 42,
) -> Any:
    """Generate additional samples from a fitted :class:`GANResult`."""
    if count < 1:
        raise ValueError("count must be positive")
    import numpy as np

    from ml_lab.neural import seed_everything
    from ml_lab.neural.backend import require_torch

    torch = require_torch(purpose="GAN sample generation")
    generator = result.generator
    try:
        device = next(generator.parameters()).device
    except StopIteration as exc:  # pragma: no cover - all built-ins have parameters
        raise ValueError("generator has no parameters") from exc
    seed_everything(random_state, deterministic=True, torch_module=torch)
    generator.eval()
    latent_dim = int(result.metadata["latent_dim"])
    with torch.no_grad():
        z = torch.randn(count, latent_dim, device=device)
        generated_training = generator(z).detach().cpu().numpy()
    generated = (
        generated_training
        if result.transformer is None
        else result.transformer.inverse_transform(generated_training)
    )
    return np.asarray(generated, dtype=float)
