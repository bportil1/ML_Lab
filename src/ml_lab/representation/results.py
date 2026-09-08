from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

RepresentationMethod = Literal["pca", "mlp_autoencoder"]


@dataclass(slots=True)
class RepresentationResult:
    method: RepresentationMethod
    latent: np.ndarray
    reconstruction: np.ndarray
    metrics: dict[str, float]
    feature_names: list[str]
    model: Any
    transformer: Any = None
    history: list[dict[str, float | int | None]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "task": "representation",
            "method": self.method,
            "latent_shape": list(self.latent.shape),
            "reconstruction_shape": list(self.reconstruction.shape),
            "metrics": self.metrics,
            "feature_names": self.feature_names,
            "history": self.history,
            "metadata": self.metadata,
        }
