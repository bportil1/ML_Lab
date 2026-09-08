"""Stable energy-based training schemes.

Registry/configuration imports remain PyTorch-free. PyTorch is loaded only when
an actual training run is started.
"""

from .api import fit, run
from .base import TrainingResult, TrainingScheme
from .config import (
    AdaptiveMomentumConfig,
    BestModelSelectionConfig,
    CDKTrainingConfig,
    DistributionMonitoringConfig,
    PartitionMonitoringConfig,
)
from .registry import TrainingSchemeSpec, get_scheme, list_schemes
from .reporting import save_result


def __getattr__(name: str):
    if name == "CDKTrainer":
        from .cdk import CDKTrainer
        return CDKTrainer
    raise AttributeError(name)


__all__ = [
    "AdaptiveMomentumConfig",
    "BestModelSelectionConfig",
    "CDKTrainer",
    "CDKTrainingConfig",
    "DistributionMonitoringConfig",
    "PartitionMonitoringConfig",
    "TrainingResult",
    "TrainingScheme",
    "TrainingSchemeSpec",
    "fit",
    "get_scheme",
    "list_schemes",
    "run",
    "save_result",
]
