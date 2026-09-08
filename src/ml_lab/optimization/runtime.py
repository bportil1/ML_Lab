"""Lightweight execution policy for generic candidate optimizers."""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import multiprocessing as mp
import os
import random
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OptimizationRuntime:
    device: str = "auto"
    deterministic: bool = False
    candidate_parallelism: str = "auto"
    candidate_workers: int = 1
    gpu_memory_fraction: float = 0.75
    multiprocessing_start_method: str = "spawn"
    cpu_threads_per_worker: int | None = None
    clear_cuda_cache_between_candidates: bool = True

    def __post_init__(self):
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("optimization runtime device must be auto, cpu, cuda, or mps")
        if self.candidate_parallelism not in {"auto", "serial", "process"}:
            raise ValueError("candidate_parallelism must be auto, serial, or process")
        if self.candidate_workers < 1:
            raise ValueError("candidate_workers must be >= 1")
        if not 0 < self.gpu_memory_fraction <= 1:
            raise ValueError("gpu_memory_fraction must satisfy 0 < value <= 1")

    @property
    def torch_available(self) -> bool:
        return importlib.util.find_spec("torch") is not None

    @property
    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        if not self.torch_available:
            return "cpu"
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def is_cuda(self) -> bool:
        return self.resolved_device == "cuda"

    @property
    def effective_candidate_parallelism(self) -> str:
        if self.candidate_parallelism != "auto":
            return self.candidate_parallelism
        if self.is_cuda:
            return "serial"
        return "process" if self.candidate_workers > 1 else "serial"

    def seed_all(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        if self.torch_available:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def configure_torch(self) -> None:
        if not self.torch_available:
            return
        import torch
        if self.cpu_threads_per_worker is not None:
            torch.set_num_threads(int(self.cpu_threads_per_worker))
        if self.deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

    def clear_cache(self) -> None:
        if self.is_cuda and self.clear_cuda_cache_between_candidates and self.torch_available:
            import torch
            torch.cuda.empty_cache()

    def multiprocessing_context(self):
        return mp.get_context(self.multiprocessing_start_method)

    def gpu_budget_bytes(self) -> int:
        if not self.is_cuda or not self.torch_available:
            return 0
        import torch
        props = torch.cuda.get_device_properties(torch.device("cuda"))
        return int(props.total_memory * self.gpu_memory_fraction)

    def metadata(self) -> dict[str, Any]:
        return {
            "device_requested": self.device,
            "device_resolved": self.resolved_device,
            "deterministic": self.deterministic,
            "candidate_parallelism_requested": self.candidate_parallelism,
            "candidate_parallelism_effective": self.effective_candidate_parallelism,
            "candidate_workers": self.candidate_workers,
            "gpu_memory_fraction": self.gpu_memory_fraction,
            "multiprocessing_start_method": self.multiprocessing_start_method,
            "cpu_threads_per_worker": self.cpu_threads_per_worker,
        }
