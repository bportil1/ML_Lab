"""Restricted Boltzmann machine model families.

The public registry/configuration layer remains PyTorch-free until a model is
actually built.
"""

from .api import create, create_from_config, run_chain, summarize
from .checkpoint import load, save
from .config import (
    BernoulliRBMSettings,
    GaussianRBMSettings,
    RBMConfig,
    StudentTPoERBMSettings,
    canonical_family,
    default_settings,
)
from .registry import RBMFamilySpec, get_family, list_families


def __getattr__(name: str):
    if name in {"BernoulliRBM", "GaussianRBM", "StudentTPoERBM"}:
        from . import models
        return getattr(models, name)
    raise AttributeError(name)


__all__ = [
    "BernoulliRBM",
    "BernoulliRBMSettings",
    "GaussianRBM",
    "GaussianRBMSettings",
    "RBMConfig",
    "RBMFamilySpec",
    "StudentTPoERBM",
    "StudentTPoERBMSettings",
    "canonical_family",
    "create",
    "create_from_config",
    "default_settings",
    "get_family",
    "list_families",
    "load",
    "run_chain",
    "save",
    "summarize",
]
