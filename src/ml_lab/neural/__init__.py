"""Shared neural runtime and training infrastructure.

This package deliberately avoids importing PyTorch at import time. Stable neural
models may depend on these contracts while non-neural ML_Lab installations remain
lightweight.
"""

from .callbacks import CallbackList, TrainingCallback
from .checkpoint import BestModelCheckpoint, load_checkpoint
from .config import DeviceName, NeuralTrainingConfig
from .data import ValidationSplit, split_validation_indices
from .history import TrainingHistory
from .runtime import resolve_device, seed_everything, torch_generator
from .selection import EarlyStopping
from .trainer import NeuralTrainer, NeuralTrainingResult

__all__ = [
    "BestModelCheckpoint",
    "CallbackList",
    "DeviceName",
    "EarlyStopping",
    "NeuralTrainer",
    "NeuralTrainingConfig",
    "NeuralTrainingResult",
    "TrainingCallback",
    "TrainingHistory",
    "ValidationSplit",
    "load_checkpoint",
    "resolve_device",
    "seed_everything",
    "split_validation_indices",
    "torch_generator",
]
