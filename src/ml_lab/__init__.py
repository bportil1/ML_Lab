"""ML Lab: reusable machine-learning operations for standalone and host-mounted callers."""

from . import classification, clustering, data, energy_based, generative, neural, optimization, preprocessing, regression, representation
from .core import EstimatorSpec
from .registry import list_estimators

__version__ = "0.19.0"

__all__ = ["EstimatorSpec", "classification", "clustering", "data", "energy_based", "generative", "neural", "optimization", "preprocessing", "regression", "representation", "list_estimators"]
