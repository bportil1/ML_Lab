"""Generic optimization algorithm contract."""
from __future__ import annotations

from abc import ABC, abstractmethod


class OptimizationAlgorithm(ABC):
    name: str

    @abstractmethod
    def optimize(self):
        raise NotImplementedError
