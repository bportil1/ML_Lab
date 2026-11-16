"""Shared, conservative column alignment for C5 and checkpointed C7 exports.

XML namespace prefixes are presentation aliases, not field identities.  The
single-file preview may call a field ``a:value`` in one source and ``b:value``
in another even when both resolve to the same expanded QName and canonical
field ID.  Collection exports must use the canonical IDs to align them.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping
from typing import Any

from .xml_null_export import COMMENTS


def _descriptor(column: Any) -> tuple[str, tuple[Any, ...]]:
    """Extract the same identity from preview objects or C7 JSON columns."""
    if isinstance(column, Mapping):
        name = column['name']
        identity = (column['strategy'], column['branch'], tuple(column['field_ids']),
                    tuple(column['element_paths']))
    else:
        name = column.name
        identity = (column.strategy, column.repeated_branch_canonical_path,
                    tuple(column.source_field_ids),
                    tuple(column.source_element_canonical_paths))
    return name, identity


def register_collection_columns(
    columns: Iterable[Any],
    headers: OrderedDict[str, tuple[Any, ...]],
    seen: dict[tuple[Any, ...], str],
    *,
    child: bool = False,
) -> dict[str, str]:
    """Register canonical identities and return per-source display-name aliases.

    Keep the earliest verified source's human-readable column name (thus no
    change for previously working collections), but recognize equivalent QNames
    whose prefixes differ.  Never merge different identities that happen to
    share one label; that would conflate scientifically different fields.
    """
    aliases: dict[str, str] = {}
    targets: set[str] = set()
    for column in columns:
        name, identity = _descriptor(column)
        if name in {'__source_file', '__source_row', COMMENTS} or (
            child and name == '__parent_record' and identity[0] != 'link'
        ):
            raise ValueError(f'XML column conflicts with reserved provenance column: {name}')
        if name in aliases:
            raise ValueError(f'Duplicate XML source column: {name!r}')
        existing_identity = headers.get(name)
        if existing_identity is not None and existing_identity != identity:
            raise ValueError(f'XML column {name!r} has conflicting meanings across sources')
        target = seen.get(identity, name)
        if target in targets:
            raise ValueError(f'Multiple XML source columns map to the same field: {target!r}')
        if target in headers and headers[target] != identity:
            raise ValueError(f'XML column {target!r} has conflicting meanings across sources')
        headers.setdefault(target, identity)
        seen.setdefault(identity, target)
        aliases[name] = target
        targets.add(target)
    return aliases


def align_collection_row(row: Mapping[str, Any], aliases: Mapping[str, str]) -> dict[str, Any]:
    """Rename only known field keys; leave lineage keys untouched."""
    return {aliases.get(key, key): value for key, value in row.items()}
