from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .intake import inspect_file

_DATASET_SCHEMA = "ml-lab.dataset-reference@1"
_EVENT_SCHEMA = "ml-lab.transformation-event@1"
_LINEAGE_SCHEMA = "ml-lab.data-lineage@1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _software_version() -> str:
    try:
        return metadata.version("ml-lab")
    except metadata.PackageNotFoundError:  # pragma: no cover - source checkout without installation
        return "development"


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stable_value(value: Any) -> str:
    if value is pd.NA or value is pd.NaT or value is None:
        return "null:"
    if isinstance(value, (np.bool_, bool)):
        return "bool:1" if bool(value) else "bool:0"
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return f"int:{int(value)}"
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if math.isnan(numeric):
            return "null:"
        if math.isinf(numeric):
            return "float:+inf" if numeric > 0 else "float:-inf"
        return f"float:{numeric.hex()}"
    if isinstance(value, pd.Timestamp):
        return f"datetime:{value.isoformat()}"
    try:
        if pd.isna(value):
            return "null:"
    except (TypeError, ValueError):
        pass
    return "str:" + str(value)


def logical_table_sha256(frame: pd.DataFrame) -> str:
    """Return a deterministic logical-table fingerprint.

    The digest is independent of CSV/TSV delimiter and line endings, but intentionally
    preserves column order, row order, parsed dtypes, and values. It complements rather
    than replaces the exact byte-level SHA-256 of the source file.
    """
    digest = hashlib.sha256()
    digest.update(b"ml-lab.logical-table@1\0")
    for column, dtype in zip(frame.columns, frame.dtypes):
        digest.update(str(column).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(dtype).encode("utf-8"))
        digest.update(b"\0")
    digest.update(b"\xff")
    for row in frame.itertuples(index=False, name=None):
        for value in row:
            encoded = _stable_value(value).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        digest.update(b"\xfe")
    return digest.hexdigest()


def _read_frame(path: Path) -> tuple[pd.DataFrame, Any]:
    record = inspect_file(path)
    if not record.supported or record.format != "delimited_text":
        raise ValueError(f"unsupported provenance dataset: {path}")
    if record.parse_status == "failed" or record.encoding is None or record.delimiter is None:
        raise ValueError(f"dataset could not be parsed: {record.error or path}")
    frame = pd.read_csv(
        path,
        sep=record.delimiter,
        encoding=record.encoding,
        header=0 if record.has_header else None,
        on_bad_lines="skip",
        keep_default_na=True,
    )
    if not record.has_header:
        frame.columns = list(record.columns)
    return frame, record


def describe_dataset(
    path: str | Path,
    *,
    frame: pd.DataFrame | None = None,
    source_record: Any | None = None,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if frame is None:
        frame, source_record = _read_frame(resolved)
    byte_sha = getattr(source_record, "sha256", None) if source_record is not None else None
    byte_sha = str(byte_sha) if byte_sha else _sha256(resolved)
    logical_sha = logical_table_sha256(frame)
    return {
        "schema": _DATASET_SCHEMA,
        "dataset_id": f"dataset:sha256:{logical_sha}",
        "path": str(resolved),
        "sha256": byte_sha,
        "logical_sha256": logical_sha,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "column_names": [str(column) for column in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
    }


def provenance_sidecar_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    return resolved.with_suffix(resolved.suffix + ".provenance.json")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_provenance_event(path: str | Path) -> dict[str, Any]:
    candidate = Path(path).expanduser().resolve()
    if candidate.suffix != ".json" or not candidate.name.endswith(".provenance.json"):
        candidate = provenance_sidecar_path(candidate)
    if not candidate.is_file():
        raise FileNotFoundError(f"provenance event not found: {candidate}")
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    if payload.get("schema") != _EVENT_SCHEMA:
        raise ValueError(f"unsupported provenance schema: {payload.get('schema')!r}")
    payload["provenance_path"] = str(candidate)
    return payload


def _parent_reference(source: Mapping[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    warnings: list[str] = []
    sidecar = provenance_sidecar_path(str(source["path"]))
    if not sidecar.is_file():
        return None, warnings
    try:
        parent = load_provenance_event(sidecar)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        warnings.append(f"Ignored unreadable parent provenance sidecar {sidecar}: {exc}")
        return None, warnings
    parent_derived = parent.get("derived", {})
    if parent_derived.get("sha256") != source.get("sha256"):
        warnings.append(
            "Ignored parent provenance because the current source bytes no longer match "
            f"the recorded derived SHA-256 in {sidecar}."
        )
        return None, warnings
    return {
        "event_id": parent.get("event_id"),
        "provenance_path": str(sidecar),
        "dataset_id": parent_derived.get("dataset_id"),
    }, warnings


def persist_transformation_provenance(
    *,
    source_path: str | Path,
    derived_path: str | Path,
    recipe: Mapping[str, Any],
    operations: Iterable[Mapping[str, Any]],
    recipe_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    warnings: Iterable[str] = (),
    created_at: str | None = None,
) -> dict[str, Any]:
    """Persist an authoritative ML Lab transformation event beside a derived dataset."""
    source = describe_dataset(source_path)
    derived = describe_dataset(derived_path)
    recipe_snapshot = json.loads(json.dumps(dict(recipe), default=str))
    operation_records = [json.loads(json.dumps(dict(item), default=str)) for item in operations]
    recipe_sha = _json_sha256(recipe_snapshot)
    parent, parent_warnings = _parent_reference(source)
    all_warnings = [str(item) for item in warnings] + parent_warnings
    identity = {
        "source_dataset_id": source["dataset_id"],
        "derived_dataset_id": derived["dataset_id"],
        "recipe_sha256": recipe_sha,
    }
    transformation_id = "transformation:sha256:" + _json_sha256(identity)
    event_created_at = created_at or datetime.now(timezone.utc).isoformat()
    event_id = "event:sha256:" + _json_sha256({
        "transformation_id": transformation_id,
        "created_at": event_created_at,
        "source_path": source["path"],
        "derived_path": derived["path"],
    })
    changes = {
        "rows_before": source["rows"],
        "rows_after": derived["rows"],
        "row_delta": int(derived["rows"] - source["rows"]),
        "columns_before": source["columns"],
        "columns_after": derived["columns"],
        "column_delta": int(derived["columns"] - source["columns"]),
        "operation_count": len(operation_records),
        "coercion_failures": int(sum(int(item.get("coercion_failures", 0) or 0) for item in operation_records)),
    }
    event = {
        "schema": _EVENT_SCHEMA,
        "event_id": event_id,
        "transformation_id": transformation_id,
        "created_at": event_created_at,
        "authoritative": True,
        "relation": "derived_from",
        "software": {"name": "ml-lab", "version": _software_version()},
        "source": source,
        "derived": derived,
        "parent": parent,
        "recipe": {
            "schema": recipe_snapshot.get("schema", "ml-lab.transformation-recipe@1"),
            "sha256": recipe_sha,
            "path": (str(Path(recipe_path).expanduser().resolve()) if recipe_path else None),
            "snapshot": recipe_snapshot,
        },
        "manifest_path": (str(Path(manifest_path).expanduser().resolve()) if manifest_path else None),
        "changes": changes,
        "operations": operation_records,
        "warnings": all_warnings,
    }
    sidecar = provenance_sidecar_path(derived_path)
    _atomic_write_json(sidecar, event)
    event["provenance_path"] = str(sidecar)
    return event


def trace_lineage(path: str | Path, *, max_depth: int = 100) -> dict[str, Any]:
    """Trace recorded ML Lab transformation events from a dataset back to its raw root.

    Only explicit ML Lab provenance sidecars are treated as authoritative edges. Similarity
    or dataset-comparison heuristics are intentionally not promoted into lineage. Parent
    traversal follows the event linkage recorded when the transformation was created; a
    sidecar that merely appears beside a source later does not retroactively create history.
    """
    target_path = Path(path).expanduser().resolve()
    if not target_path.is_file():
        raise FileNotFoundError(f"dataset not found: {target_path}")
    target = describe_dataset(target_path)
    current_path = target_path
    events_reverse: list[dict[str, Any]] = []
    warnings: list[str] = []
    valid = True
    visited_events: set[str] = set()
    expected_event_id: str | None = None
    root: dict[str, Any] = target

    for _ in range(max_depth):
        sidecar = provenance_sidecar_path(current_path)
        if not sidecar.is_file():
            root = describe_dataset(current_path) if current_path.is_file() else root
            break
        try:
            event = load_provenance_event(sidecar)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            warnings.append(f"Could not read provenance sidecar {sidecar}: {exc}")
            valid = False
            break

        event_id = str(event.get("event_id") or "")
        if not event_id:
            warnings.append(f"Provenance sidecar has no event_id: {sidecar}")
            valid = False
            break
        if event_id in visited_events:
            warnings.append(f"Lineage cycle detected at provenance event {event_id}.")
            valid = False
            break
        visited_events.add(event_id)
        if expected_event_id is not None and event_id != expected_event_id:
            warnings.append(
                f"Parent event mismatch: expected {expected_event_id}, found {event_id} in {sidecar}."
            )
            valid = False

        if not current_path.is_file():
            warnings.append(f"Recorded derived dataset is missing: {current_path}")
            valid = False
            break
        current_sha = _sha256(current_path)
        if event.get("derived", {}).get("sha256") != current_sha:
            warnings.append(f"Derived dataset hash no longer matches recorded provenance: {current_path}")
            valid = False
        events_reverse.append(event)

        source_payload = event.get("source") or {}
        source_value = source_payload.get("path")
        if not source_value:
            warnings.append(f"Provenance event {event_id} has no source path.")
            valid = False
            break
        source_path = Path(str(source_value)).expanduser().resolve()
        root = dict(source_payload)
        if not source_path.is_file():
            warnings.append(f"Recorded source dataset is missing: {source_path}")
            valid = False
            break
        if source_payload.get("sha256") != _sha256(source_path):
            warnings.append(f"Source dataset hash no longer matches recorded provenance: {source_path}")
            valid = False

        parent = event.get("parent")
        if not parent:
            root = describe_dataset(source_path)
            break
        expected_event_id = str(parent.get("event_id") or "") or None
        recorded_parent_path = parent.get("provenance_path")
        actual_parent_path = provenance_sidecar_path(source_path)
        if recorded_parent_path:
            try:
                recorded_resolved = Path(str(recorded_parent_path)).expanduser().resolve()
            except OSError:
                recorded_resolved = actual_parent_path
            if recorded_resolved != actual_parent_path:
                warnings.append(
                    "Recorded parent provenance path does not match the source dataset sidecar: "
                    f"{recorded_resolved} != {actual_parent_path}"
                )
                valid = False
        current_path = source_path
    else:
        warnings.append(f"Lineage exceeded max_depth={max_depth}; possible cycle or unusually deep chain.")
        valid = False

    events = list(reversed(events_reverse))
    lineage_identity = {
        "target_dataset_id": target["dataset_id"],
        "event_ids": [event.get("event_id") for event in events],
    }
    return {
        "schema": _LINEAGE_SCHEMA,
        "lineage_id": "lineage:sha256:" + _json_sha256(lineage_identity),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "recorded" if events else "unrecorded_root",
        "authoritative": bool(events) and valid,
        "chain_valid": valid,
        "root": root,
        "target": target,
        "event_count": len(events),
        "events": events,
        "warnings": warnings,
    }

def save_lineage(lineage: Mapping[str, Any], path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    _atomic_write_json(destination, lineage)
    return destination


__all__ = [
    "describe_dataset",
    "load_provenance_event",
    "logical_table_sha256",
    "persist_transformation_provenance",
    "provenance_sidecar_path",
    "save_lineage",
    "trace_lineage",
]
