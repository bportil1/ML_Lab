from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Literal

ExperimentalStatus = Literal["prototype", "experimental", "candidate", "deprecated"]
_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class ExperimentalManifest:
    """Lightweight metadata for an experimental module.

    The manifest deliberately contains an import path rather than an imported
    module object. Listing experiments therefore does not import experimental
    dependencies or execute experimental code.
    """

    id: str
    name: str
    module_path: str
    description: str = ""
    status: ExperimentalStatus = "experimental"
    capabilities: tuple[str, ...] = ()
    entrypoint: str = "run"
    version: str | None = None
    notes: str = ""

    def validate(self) -> None:
        if not _ID_PATTERN.fullmatch(self.id):
            raise ValueError("experimental id must match ^[a-z][a-z0-9_]*$")
        if not self.name.strip():
            raise ValueError("experimental name must not be empty")
        if not self.module_path.strip():
            raise ValueError("experimental module_path must not be empty")
        if not self.entrypoint.strip():
            raise ValueError("experimental entrypoint must not be empty")

    def to_record(self) -> dict[str, object]:
        record = asdict(self)
        record["capabilities"] = list(self.capabilities)
        return record
