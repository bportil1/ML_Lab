from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TrainingHistory:
    records: list[dict[str, Any]] = field(default_factory=list)

    def append(self, epoch: int, **metrics: Any) -> dict[str, Any]:
        record = {"epoch": int(epoch), **metrics}
        self.records.append(record)
        return record

    @property
    def last(self) -> dict[str, Any] | None:
        return self.records[-1] if self.records else None

    def to_records(self) -> list[dict[str, Any]]:
        return [dict(record) for record in self.records]
