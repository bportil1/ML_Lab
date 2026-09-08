from __future__ import annotations

from typing import Any


def symmetric_infonce(q: Any, k: Any, *, temperature: float = 0.1):
    """Symmetric in-batch InfoNCE for matched augmented views."""
    import torch
    import torch.nn.functional as F

    if q.ndim != 2 or k.ndim != 2 or q.shape != k.shape:
        raise ValueError("q and k must be matching 2-D embedding tensors")
    if q.shape[0] < 2:
        raise ValueError("InfoNCE requires at least two examples")
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    logits = q @ k.T / float(temperature)
    labels = torch.arange(q.shape[0], device=q.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def embedding_moment_loss(fake_embeddings: Any, real_embeddings: Any):
    """Distribution-level feature matching without arbitrary fake/real pairing."""
    import torch.nn.functional as F

    if fake_embeddings.ndim != 2 or real_embeddings.ndim != 2:
        raise ValueError("embeddings must be 2-D")
    mean_loss = F.mse_loss(fake_embeddings.mean(0), real_embeddings.mean(0))
    fake_var = fake_embeddings.var(0, unbiased=False)
    real_var = real_embeddings.var(0, unbiased=False)
    return mean_loss + F.mse_loss(fake_var, real_var)
