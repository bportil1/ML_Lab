from __future__ import annotations


def validate_sharpness(value: float) -> float:
    value = float(value)
    if value <= 0:
        raise ValueError("sharpness must be > 0")
    return value


def validate_dropout(value: float) -> float:
    value = float(value)
    if not 0.0 <= value < 1.0:
        raise ValueError("dropout must satisfy 0 <= dropout < 1")
    return value


def sharp_sigmoid(logits, sharpness: float):
    from ml_lab.neural.backend import require_torch

    torch = require_torch(purpose="RBM sampling")
    temperature = max(validate_sharpness(sharpness), 1e-6)
    return torch.sigmoid(logits / temperature)


def hidden_dropout(sample, p: float, *, training: bool):
    from ml_lab.neural.backend import require_torch

    torch = require_torch(purpose="RBM sampling")
    p = validate_dropout(p)
    if not training or p == 0.0:
        return sample
    keep_probability = 1.0 - p
    mask = torch.bernoulli(torch.full_like(sample, keep_probability))
    return sample * mask
