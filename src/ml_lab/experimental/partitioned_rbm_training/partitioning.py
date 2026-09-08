from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class PartitionBlock:
    visible_start: int
    visible_end: int
    hidden_start: int
    hidden_end: int
    lineage: tuple[int, ...] = field(default_factory=tuple)
    scalar_state: dict[str, float] = field(default_factory=dict, compare=False)

    @property
    def visible_dim(self) -> int:
        return int(self.visible_end - self.visible_start)

    @property
    def hidden_dim(self) -> int:
        return int(self.hidden_end - self.hidden_start)

    def to_record(self) -> dict[str, object]:
        return {
            "visible_start": int(self.visible_start),
            "visible_end": int(self.visible_end),
            "visible_dim": self.visible_dim,
            "hidden_start": int(self.hidden_start),
            "hidden_end": int(self.hidden_end),
            "hidden_dim": self.hidden_dim,
            "lineage": list(self.lineage),
            "scalar_state": dict(self.scalar_state),
        }


def balanced_ranges(total: int, parts: int) -> list[tuple[int, int]]:
    total = int(total)
    parts = int(parts)
    if total < 1:
        raise ValueError("total must be >= 1")
    if parts < 1 or parts > total:
        raise ValueError("parts must satisfy 1 <= parts <= total")
    base, remainder = divmod(total, parts)
    ranges: list[tuple[int, int]] = []
    start = 0
    for index in range(parts):
        size = base + (1 if index < remainder else 0)
        end = start + size
        ranges.append((start, end))
        start = end
    assert start == total
    return ranges


def initial_blocks(visible_dim: int, hidden_dim: int, partitions: int) -> list[PartitionBlock]:
    partitions = int(partitions)
    if partitions > int(visible_dim):
        raise ValueError("initial_partitions cannot exceed visible_dim")
    if partitions > int(hidden_dim):
        raise ValueError("initial_partitions cannot exceed hidden_dim")
    visible = balanced_ranges(visible_dim, partitions)
    hidden = balanced_ranges(hidden_dim, partitions)
    return [
        PartitionBlock(v0, v1, h0, h1, lineage=(index,))
        for index, ((v0, v1), (h0, h1)) in enumerate(zip(visible, hidden))
    ]


def _weighted_scalar_state(blocks: Iterable[PartitionBlock]) -> dict[str, float]:
    blocks = list(blocks)
    keys = sorted({key for block in blocks for key in block.scalar_state})
    output: dict[str, float] = {}
    for key in keys:
        weighted = [(block.visible_dim, block.scalar_state[key]) for block in blocks if key in block.scalar_state]
        denominator = sum(weight for weight, _ in weighted)
        if denominator:
            output[key] = sum(weight * value for weight, value in weighted) / denominator
    return output


def merge_blocks(blocks: list[PartitionBlock], *, merge_factor: int = 2) -> list[PartitionBlock]:
    if int(merge_factor) < 2:
        raise ValueError("merge_factor must be >= 2")
    if not blocks:
        return []
    merged: list[PartitionBlock] = []
    for start in range(0, len(blocks), int(merge_factor)):
        group = blocks[start : start + int(merge_factor)]
        merged.append(
            PartitionBlock(
                visible_start=group[0].visible_start,
                visible_end=group[-1].visible_end,
                hidden_start=group[0].hidden_start,
                hidden_end=group[-1].hidden_end,
                lineage=tuple(value for block in group for value in block.lineage),
                scalar_state=_weighted_scalar_state(group),
            )
        )
    return merged
