from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .provenance import _atomic_write_json, _json_sha256, _sha256, load_provenance_event
from .types import XmlStructureArtifact

_HISTORY_SCHEMA = "ml-lab.xml-extraction-history@1"
_COMPARE_SCHEMA = "ml-lab.xml-extraction-recipe-comparison@1"


def xml_extraction_family_id(*, source_sha256: str, source_fingerprint: str) -> str:
    identity = {
        "source_sha256": str(source_sha256),
        "source_fingerprint": str(source_fingerprint),
    }
    return "xml-extraction-family:sha256:" + _json_sha256(identity)


def xml_extraction_history_path(history_root: str | Path, source_fingerprint: str) -> Path:
    root = Path(history_root).expanduser().resolve()
    safe = source_fingerprint.removeprefix("sha256:").replace(":", "_")
    return root / ".xml-extractions" / f"{safe}.json"


def _event_matches_artifact(event: Mapping[str, Any], artifact: XmlStructureArtifact) -> bool:
    source = event.get("source") or {}
    return (
        event.get("operation_kind") == "xml_tabularization"
        and source.get("source_type") == "xml"
        and source.get("sha256") == artifact.source_sha256
        and source.get("fingerprint") == artifact.source_fingerprint
    )


def _entry_from_event(event: Mapping[str, Any]) -> dict[str, Any]:
    recipe = (event.get("recipe") or {}).get("snapshot") or {}
    derived = event.get("derived") or {}
    structure = event.get("structure_artifact") or {}
    supplementals = event.get("supplemental_datasets") or []
    dataset_path = str(derived.get("path") or "")
    status = "available"
    status_detail = None
    if not dataset_path or not Path(dataset_path).expanduser().is_file():
        status = "missing_dataset"
        status_detail = "Derived dataset is no longer present."
    elif derived.get("sha256"):
        try:
            if _sha256(Path(dataset_path).expanduser().resolve()) != derived.get("sha256"):
                status = "modified_dataset"
                status_detail = "Derived dataset bytes no longer match recorded provenance."
        except OSError as exc:
            status = "unreadable_dataset"
            status_detail = str(exc)
    return {
        "event_id": event.get("event_id"),
        "transformation_id": event.get("transformation_id"),
        "created_at": event.get("created_at"),
        "provenance_path": event.get("provenance_path"),
        "dataset": {
            "path": dataset_path,
            "dataset_id": derived.get("dataset_id"),
            "sha256": derived.get("sha256"),
            "logical_sha256": derived.get("logical_sha256"),
            "rows": derived.get("rows"),
            "columns": derived.get("columns"),
            "column_names": list(derived.get("column_names") or []),
        },
        "record_root_canonical_path": recipe.get("record_root_canonical_path"),
        "selected_field_ids": list(recipe.get("selected_field_ids") or []),
        "rules": list(recipe.get("rules") or []),
        "preview_signature": recipe.get("preview_signature"),
        "structure_artifact_path": structure.get("path"),
        "supplemental_dataset_paths": [str(item.get("path") or "") for item in supplementals],
        "status": status,
        "status_detail": status_detail,
    }


def _load_catalog(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("schema") != _HISTORY_SCHEMA:
        return None
    return payload


def _write_catalog(path: Path, artifact: XmlStructureArtifact, entries: list[dict[str, Any]]) -> dict[str, Any]:
    unique: dict[str, dict[str, Any]] = {}
    for entry in entries:
        event_id = str(entry.get("event_id") or "")
        if event_id:
            unique[event_id] = entry
    ordered = sorted(unique.values(), key=lambda item: str(item.get("created_at") or ""), reverse=True)
    payload = {
        "schema": _HISTORY_SCHEMA,
        "family_id": xml_extraction_family_id(
            source_sha256=artifact.source_sha256,
            source_fingerprint=artifact.source_fingerprint,
        ),
        "source": {
            "path": artifact.source_path,
            "sha256": artifact.source_sha256,
            "fingerprint": artifact.source_fingerprint,
            "structure_schema": artifact.schema,
        },
        "entry_count": len(ordered),
        "entries": ordered,
    }
    _atomic_write_json(path, payload)
    payload["history_path"] = str(path)
    return payload


def refresh_xml_extraction_history(
    artifact: XmlStructureArtifact,
    *,
    history_root: str | Path,
) -> dict[str, Any]:
    """Refresh the convenience history index from authoritative provenance events.

    The history index is not an authoritative lineage source. Every entry originates from
    a valid ML Lab XML tabularization sidecar and is revalidated when loaded.
    """
    root = Path(history_root).expanduser().resolve()
    catalog_path = xml_extraction_history_path(root, artifact.source_fingerprint)
    entries: list[dict[str, Any]] = []
    warnings: list[str] = []

    existing = _load_catalog(catalog_path)
    candidate_paths: set[Path] = set()
    if existing:
        for item in existing.get("entries") or []:
            raw = item.get("provenance_path")
            if raw:
                candidate_paths.add(Path(str(raw)).expanduser().resolve())

    if root.exists():
        try:
            candidate_paths.update(root.rglob("*.provenance.json"))
        except OSError as exc:
            warnings.append(f"Could not scan XML extraction history root {root}: {exc}")

    for sidecar in sorted(candidate_paths):
        try:
            event = load_provenance_event(sidecar)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            warnings.append(f"Ignored unreadable provenance sidecar {sidecar}: {exc}")
            continue
        if not _event_matches_artifact(event, artifact):
            continue
        entries.append(_entry_from_event(event))

    payload = _write_catalog(catalog_path, artifact, entries)
    payload["warnings"] = warnings
    return payload


def register_xml_extraction(
    provenance_or_dataset: str | Path,
    *,
    artifact: XmlStructureArtifact,
    history_root: str | Path,
) -> dict[str, Any]:
    event = load_provenance_event(provenance_or_dataset)
    if not _event_matches_artifact(event, artifact):
        raise ValueError("XML extraction provenance does not belong to this source artifact")
    catalog_path = xml_extraction_history_path(history_root, artifact.source_fingerprint)
    existing = _load_catalog(catalog_path) or {"entries": []}
    entries = list(existing.get("entries") or [])
    entries.append(_entry_from_event(event))
    return _write_catalog(catalog_path, artifact, entries)


def _resolve_history_event(
    artifact: XmlStructureArtifact,
    event_id: str,
    *,
    history_root: str | Path,
) -> dict[str, Any]:
    history = refresh_xml_extraction_history(artifact, history_root=history_root)
    match = next((entry for entry in history.get("entries", []) if entry.get("event_id") == event_id), None)
    if match is None:
        raise KeyError(f"XML extraction event is not recorded for this source: {event_id}")
    provenance_path = match.get("provenance_path")
    if not provenance_path:
        raise ValueError(f"XML extraction event has no provenance path: {event_id}")
    event = load_provenance_event(provenance_path)
    if not _event_matches_artifact(event, artifact):
        raise ValueError("XML extraction event source no longer matches this XML artifact")
    return event


def load_xml_extraction_state(
    artifact: XmlStructureArtifact,
    event_id: str,
    *,
    history_root: str | Path,
) -> dict[str, Any]:
    event = _resolve_history_event(artifact, event_id, history_root=history_root)
    recipe = (event.get("recipe") or {}).get("snapshot") or {}
    return {
        "schema": "ml-lab.xml-extraction-reentry@1",
        "family_id": xml_extraction_family_id(
            source_sha256=artifact.source_sha256,
            source_fingerprint=artifact.source_fingerprint,
        ),
        "event": _entry_from_event(event),
        "selection": {
            "source_fingerprint": artifact.source_fingerprint,
            "record_root_canonical_path": recipe.get("record_root_canonical_path"),
            "selected_field_ids": list(recipe.get("selected_field_ids") or []),
            "collection_rules": list(recipe.get("rules") or []),
            "confirmed_preview_signature": recipe.get("preview_signature"),
        },
    }


def _rule_map(recipe: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in recipe.get("rules") or []:
        if not isinstance(raw, Mapping):
            continue
        branch = str(raw.get("branch_canonical_path") or "")
        if branch:
            result[branch] = json.loads(json.dumps(dict(raw), sort_keys=True, default=str))
    return result


def compare_xml_extraction_recipes(
    artifact: XmlStructureArtifact,
    left_event_id: str,
    right_event_id: str,
    *,
    history_root: str | Path,
) -> dict[str, Any]:
    left = _resolve_history_event(artifact, left_event_id, history_root=history_root)
    right = _resolve_history_event(artifact, right_event_id, history_root=history_root)
    left_recipe = (left.get("recipe") or {}).get("snapshot") or {}
    right_recipe = (right.get("recipe") or {}).get("snapshot") or {}

    left_fields = set(str(item) for item in left_recipe.get("selected_field_ids") or [])
    right_fields = set(str(item) for item in right_recipe.get("selected_field_ids") or [])
    left_rules = _rule_map(left_recipe)
    right_rules = _rule_map(right_recipe)
    rule_paths = sorted(set(left_rules) | set(right_rules))
    rule_changes = []
    for path in rule_paths:
        before = left_rules.get(path)
        after = right_rules.get(path)
        if before == after:
            continue
        rule_changes.append({
            "branch_canonical_path": path,
            "left": before,
            "right": after,
            "change": "added" if before is None else "removed" if after is None else "changed",
        })

    left_root = left_recipe.get("record_root_canonical_path")
    right_root = right_recipe.get("record_root_canonical_path")
    return {
        "schema": _COMPARE_SCHEMA,
        "family_id": xml_extraction_family_id(
            source_sha256=artifact.source_sha256,
            source_fingerprint=artifact.source_fingerprint,
        ),
        "left": _entry_from_event(left),
        "right": _entry_from_event(right),
        "same_source": True,
        "record_root": {
            "left": left_root,
            "right": right_root,
            "changed": left_root != right_root,
        },
        "fields": {
            "added": sorted(right_fields - left_fields),
            "removed": sorted(left_fields - right_fields),
            "unchanged": sorted(left_fields & right_fields),
        },
        "rules": {
            "changed_count": len(rule_changes),
            "changes": rule_changes,
        },
        "same_recipe": (
            left_root == right_root
            and left_fields == right_fields
            and left_rules == right_rules
        ),
    }


__all__ = [
    "compare_xml_extraction_recipes",
    "load_xml_extraction_state",
    "refresh_xml_extraction_history",
    "register_xml_extraction",
    "xml_extraction_family_id",
    "xml_extraction_history_path",
]
