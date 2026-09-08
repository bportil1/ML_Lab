from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingSchemeSpec:
    id: str
    name: str
    description: str

    def to_record(self) -> dict[str, str]:
        return {"id": self.id, "name": self.name, "description": self.description}


_SPECS = (
    TrainingSchemeSpec(
        id="cdk",
        name="Contrastive Divergence CD-k",
        description="Autograd-consistent CD-k with optional persistent chains, monitoring, and relative best-model selection.",
    ),
)


def list_schemes() -> tuple[TrainingSchemeSpec, ...]:
    return _SPECS


def get_scheme(name: str) -> TrainingSchemeSpec:
    key = str(name).strip().lower().replace("-", "")
    if key in {"cd", "cdk", "contrastivedivergence"}:
        return _SPECS[0]
    raise ValueError(f"unknown energy-based training scheme {name!r}; available: cdk")
