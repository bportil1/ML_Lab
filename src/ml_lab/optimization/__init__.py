"""Generic optimization/search infrastructure.

Firefly is adapted from the current HSQA_DBN optimizer without modifying the
HSQA_DBN source repository. The stable contracts are task-neutral and can be
used by ML_Lab models, PAH modules, or external callers.
"""

from .api import optimize
from .base import OptimizationAlgorithm
from .config import FireflyConfig
from .evaluator import CandidateEvaluator, FunctionEvaluator
from .fitness import FitnessEvaluator, FitnessTermSpec
from .registry import available_optimizers, create_optimizer, register_optimizer
from .result import CandidateEvaluation, OptimizationResult
from .runtime import OptimizationRuntime
from .space import ParameterSpec, SearchSpace

__all__ = [
    "CandidateEvaluation",
    "CandidateEvaluator",
    "FireflyConfig",
    "FitnessEvaluator",
    "FitnessTermSpec",
    "FunctionEvaluator",
    "OptimizationAlgorithm",
    "OptimizationResult",
    "OptimizationRuntime",
    "ParameterSpec",
    "SearchSpace",
    "available_optimizers",
    "create_optimizer",
    "optimize",
    "register_optimizer",
]
