"""Explicit, locally owned source-adapter registry (ML-UW-29).

Neither the registry nor the StructuredObject core knows source domains.
No global registrations, implicit source reads or runtime files are created.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Any

from ..contracts import StructuredContractError
from ..object import StructuredObject


@dataclass(frozen=True, slots=True)
class AdaptationResult:
    """Validated result plus an honest account of information not represented."""

    object: StructuredObject
    source_schema: str
    warnings: tuple[str, ...] = ()
    losses: tuple[str, ...] = ()

    @property
    def lossless(self) -> bool:
        return not self.losses

    def require_lossless(self) -> StructuredObject:
        if self.losses:
            raise StructuredContractError('lossy source conversion: ' + ', '.join(self.losses))
        return self.object


class SourceAdapter(Protocol):
    """An adapter converts a supplied source artifact, never an implicit path."""

    def adapt(self, source: Any) -> AdaptationResult: ...


class AdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, SourceAdapter] = {}

    def register(self, name: str, adapter: SourceAdapter) -> None:
        if not isinstance(name, str) or not name.strip() or name in self._adapters:
            raise StructuredContractError('adapter name must be nonempty and unique')
        if not callable(getattr(adapter, 'adapt', None)):
            raise StructuredContractError('adapter must implement adapt(source)')
        self._adapters[name] = adapter

    def adapt(self, name: str, source: Any) -> AdaptationResult:
        try:
            adapter = self._adapters[name]
        except (KeyError, TypeError) as exc:
            raise StructuredContractError(f'unregistered source adapter: {name!r}') from exc
        result = adapter.adapt(source)
        if not isinstance(result, AdaptationResult):
            raise StructuredContractError('adapter did not return an AdaptationResult')
        return result

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))
