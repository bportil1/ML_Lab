"""Stable energy-based models and training schemes.

Importing this package does not import PyTorch. Model/trainer implementations
are loaded only when an energy-based operation is explicitly executed.
"""

from . import rbm, training

__all__ = ["rbm", "training"]
