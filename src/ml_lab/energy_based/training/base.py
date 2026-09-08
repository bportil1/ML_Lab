from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TrainingResult:
    model: Any
    history: list[dict[str, Any]]
    scheme: str
    stop_reason: str
    feature_names: list[str]
    metrics: dict[str, Any] = field(default_factory=dict)
    generated_samples: Any | None = None
    reference_samples: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "scheme": self.scheme,
            "stop_reason": self.stop_reason,
            "feature_names": list(self.feature_names),
            "metrics": dict(self.metrics),
            "history": [dict(record) for record in self.history],
            "generated_shape": None if self.generated_samples is None else list(self.generated_samples.shape),
            "reference_shape": None if self.reference_samples is None else list(self.reference_samples.shape),
            "metadata": dict(self.metadata),
        }


class TrainingScheme(ABC):
    name: str

    @abstractmethod
    def fit(self, model: Any, data: Any, **kwargs: Any) -> TrainingResult:
        raise NotImplementedError
