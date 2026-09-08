from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class GANResult:
    generated_samples: np.ndarray
    metrics: dict[str, float]
    feature_names: list[str]
    generator: Any
    discriminator: Any
    transformer: Any = None
    history: list[dict[str, float | int | None]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "task": "gan",
            "generated_shape": list(self.generated_samples.shape),
            "metrics": self.metrics,
            "feature_names": self.feature_names,
            "history": self.history,
            "metadata": self.metadata,
        }
