from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class PartitionedRBMTrainingConfig:
    """Hierarchical feature-block pretraining followed by merged RBM refinement.

    The partitioning is always along the visible-feature axis. Samples are never
    partitioned. Hidden units are divided independently into balanced contiguous
    blocks and merged alongside the visible blocks.
    """

    initial_partitions: int = 4
    merge_factor: int = 2
    epoch_decay: float = 0.25
    min_epochs: int = 1
    stage_epochs: tuple[int, ...] | None = None
    cross_weight_initialization: str = "zero"
    monitor_intermediate_stages: bool = False

    def __post_init__(self) -> None:
        if int(self.initial_partitions) < 1:
            raise ValueError("initial_partitions must be >= 1")
        if int(self.merge_factor) < 2:
            raise ValueError("merge_factor must be >= 2")
        if not 0 < float(self.epoch_decay) <= 1:
            raise ValueError("epoch_decay must satisfy 0 < value <= 1")
        if int(self.min_epochs) < 1:
            raise ValueError("min_epochs must be >= 1")
        if self.cross_weight_initialization not in {"zero", "random"}:
            raise ValueError("cross_weight_initialization must be zero or random")
        if self.stage_epochs is not None:
            if not self.stage_epochs:
                raise ValueError("stage_epochs cannot be empty")
            if any(int(value) < 1 for value in self.stage_epochs):
                raise ValueError("every stage_epochs entry must be >= 1")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "PartitionedRBMTrainingConfig":
        options = dict(values or {})
        if "stage_epochs" in options and options["stage_epochs"] is not None:
            options["stage_epochs"] = tuple(int(value) for value in options["stage_epochs"])
        return cls(**options)

    def epochs_for_level(self, level: int, *, base_epochs: int) -> int:
        level = int(level)
        if level < 0:
            raise ValueError("level must be >= 0")
        if self.stage_epochs is not None:
            index = min(level, len(self.stage_epochs) - 1)
            return int(self.stage_epochs[index])
        return max(int(self.min_epochs), int(round(int(base_epochs) * (float(self.epoch_decay) ** level))))

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        if record["stage_epochs"] is not None:
            record["stage_epochs"] = list(record["stage_epochs"])
        return record
