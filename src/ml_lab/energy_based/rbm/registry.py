from __future__ import annotations

from dataclasses import dataclass

from .config import canonical_family, default_settings


@dataclass(frozen=True)
class RBMFamilySpec:
    id: str
    name: str
    aliases: tuple[str, ...]
    visible_distribution: str
    description: str

    def to_record(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "aliases": list(self.aliases),
            "visible_distribution": self.visible_distribution,
            "description": self.description,
            "default_settings": default_settings(self.id),
        }


_SPECS = (
    RBMFamilySpec(
        id="bernoulli",
        name="Bernoulli RBM",
        aliases=("bern",),
        visible_distribution="Bernoulli",
        description="Binary-visible, binary-hidden restricted Boltzmann machine.",
    ),
    RBMFamilySpec(
        id="gaussian",
        name="Gaussian RBM",
        aliases=("gauss",),
        visible_distribution="Diagonal Gaussian",
        description="Continuous-visible Gaussian/Bernoulli restricted Boltzmann machine.",
    ),
    RBMFamilySpec(
        id="student_t_poe",
        name="Student-t Product-of-Experts RBM",
        aliases=("stud_t", "student_t", "stpoe"),
        visible_distribution="Student-t scale mixture",
        description="Heavy-tailed continuous-visible RBM using a Student-t scale-mixture formulation.",
    ),
)


def list_families() -> tuple[RBMFamilySpec, ...]:
    return _SPECS


def get_family(family: str) -> RBMFamilySpec:
    canonical = canonical_family(family)
    return next(spec for spec in _SPECS if spec.id == canonical)
