from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .provenance import (
    _EVENT_SCHEMA,
    _atomic_write_json,
    _json_sha256,
    _sha256,
    _software_version,
    describe_dataset,
    load_provenance_event,
    provenance_sidecar_path,
)
from .types import XmlMaterializationResult, XmlStructureArtifact
from .xml_history import xml_extraction_family_id
from .xml_ingest import analyze_xml_structure
from .xml_preview import preview_xml_tabularization

_XML_RECIPE_SCHEMA = "ml-lab.xml-extraction-recipe@1"


def _stable_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def build_xml_extraction_recipe(
    artifact: XmlStructureArtifact,
    *,
    record_root_canonical_path: str,
    selected_field_ids: Iterable[str],
    rules: Iterable[Mapping[str, Any]] | None,
    preview_signature: str,
) -> dict[str, Any]:
    return {
        "schema": _XML_RECIPE_SCHEMA,
        "source_fingerprint": artifact.source_fingerprint,
        "source_sha256": artifact.source_sha256,
        "structure_schema": artifact.schema,
        "namespace_map": [item.to_record() for item in artifact.namespaces],
        "record_root_canonical_path": str(record_root_canonical_path),
        "selected_field_ids": [str(item) for item in selected_field_ids],
        "rules": [_stable_copy(dict(item)) for item in (rules or ())],
        "preview_signature": str(preview_signature),
    }


def persist_xml_extraction_provenance(
    *,
    source_path: str | Path,
    artifact: XmlStructureArtifact,
    derived_path: str | Path,
    structure_artifact_path: str | Path,
    manifest_path: str | Path,
    child_table_paths: Iterable[str | Path],
    record_root_canonical_path: str,
    selected_field_ids: Iterable[str],
    rules: Iterable[Mapping[str, Any]] | None,
    preview_signature: str,
    warnings: Iterable[str] = (),
    created_at: str | None = None,
) -> dict[str, Any]:
    source = Path(source_path).expanduser().resolve()
    if _sha256(source) != artifact.source_sha256:
        raise ValueError("XML source bytes no longer match the analyzed structure artifact")
    derived = describe_dataset(derived_path)
    structure_path = Path(structure_artifact_path).expanduser().resolve()
    manifest = Path(manifest_path).expanduser().resolve()
    child_paths = [Path(item).expanduser().resolve() for item in child_table_paths]
    recipe = build_xml_extraction_recipe(
        artifact,
        record_root_canonical_path=record_root_canonical_path,
        selected_field_ids=selected_field_ids,
        rules=rules,
        preview_signature=preview_signature,
    )
    recipe_sha = _json_sha256(recipe)
    source_record = {
        "source_type": "xml",
        "path": str(source),
        "sha256": artifact.source_sha256,
        "fingerprint": artifact.source_fingerprint,
        "structure_schema": artifact.schema,
        "root_tag": artifact.root_tag,
        "element_count": artifact.element_count,
        "namespace_count": len(artifact.namespaces),
    }
    supplemental = [
        {
            **describe_dataset(path),
            "role": "supplemental_table",
        }
        for path in child_paths
    ]
    structure_record = {
        "path": str(structure_path),
        "sha256": _sha256(structure_path),
        "schema": artifact.schema,
    }
    identity = {
        "source_sha256": artifact.source_sha256,
        "derived_dataset_id": derived["dataset_id"],
        "recipe_sha256": recipe_sha,
    }
    transformation_id = "transformation:sha256:" + _json_sha256(identity)
    event_created_at = created_at or datetime.now(timezone.utc).isoformat()
    event_id = "event:sha256:" + _json_sha256(
        {
            "transformation_id": transformation_id,
            "created_at": event_created_at,
            "source_path": str(source),
            "derived_path": derived["path"],
        }
    )
    operations = [
        {
            "operation": "xml_record_root",
            "record_root_canonical_path": str(record_root_canonical_path),
        },
        {
            "operation": "xml_field_selection",
            "selected_field_ids": list(recipe["selected_field_ids"]),
            "field_count": len(recipe["selected_field_ids"]),
        },
        *[
            {
                "operation": "xml_collection_rule",
                "order": index,
                **_stable_copy(rule),
            }
            for index, rule in enumerate(recipe["rules"], start=1)
        ],
        {
            "operation": "xml_tabular_materialization",
            "preview_signature": preview_signature,
            "supplemental_table_count": len(supplemental),
        },
    ]
    family_id = xml_extraction_family_id(
        source_sha256=artifact.source_sha256,
        source_fingerprint=artifact.source_fingerprint,
    )
    event = {
        "schema": _EVENT_SCHEMA,
        "event_id": event_id,
        "transformation_id": transformation_id,
        "created_at": event_created_at,
        "authoritative": True,
        "relation": "derived_from",
        "operation_kind": "xml_tabularization",
        "extraction_family": {
            "family_id": family_id,
            "source_sha256": artifact.source_sha256,
            "source_fingerprint": artifact.source_fingerprint,
        },
        "software": {"name": "ml-lab", "version": _software_version()},
        "source": source_record,
        "derived": derived,
        "parent": None,
        "recipe": {
            "schema": _XML_RECIPE_SCHEMA,
            "sha256": recipe_sha,
            "path": None,
            "snapshot": recipe,
        },
        "manifest_path": str(manifest),
        "structure_artifact": structure_record,
        "supplemental_datasets": supplemental,
        "changes": {
            "rows_before": None,
            "rows_after": derived["rows"],
            "row_delta": None,
            "columns_before": None,
            "columns_after": derived["columns"],
            "column_delta": None,
            "operation_count": len(operations),
            "coercion_failures": 0,
        },
        "operations": operations,
        "warnings": [str(item) for item in warnings],
    }
    sidecar = provenance_sidecar_path(derived_path)
    _atomic_write_json(sidecar, event)
    event["provenance_path"] = str(sidecar)
    return event


def regenerate_xml_extraction(
    provenance_or_dataset: str | Path,
    *,
    output: str | Path | None = None,
    overwrite: bool = False,
) -> XmlMaterializationResult:
    """Regenerate a recorded XML extraction from its authoritative provenance recipe."""
    event = load_provenance_event(provenance_or_dataset)
    if event.get("operation_kind") != "xml_tabularization":
        raise ValueError("provenance event is not an XML tabularization event")
    recipe = event.get("recipe", {}).get("snapshot") or {}
    if recipe.get("schema") != _XML_RECIPE_SCHEMA:
        raise ValueError(f"unsupported XML extraction recipe schema: {recipe.get('schema')!r}")
    source_payload = event.get("source") or {}
    source_path = Path(str(source_payload.get("path") or "")).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"recorded XML source not found: {source_path}")
    current_sha = _sha256(source_path)
    expected_sha = str(source_payload.get("sha256") or recipe.get("source_sha256") or "")
    if not expected_sha or current_sha != expected_sha:
        raise ValueError("recorded XML source hash no longer matches the current source bytes")

    artifact = analyze_xml_structure(source_path)
    if artifact.source_fingerprint != recipe.get("source_fingerprint"):
        raise ValueError("recorded XML source fingerprint no longer matches the current source")
    root = str(recipe.get("record_root_canonical_path") or "")
    selected = [str(item) for item in recipe.get("selected_field_ids") or []]
    rules = recipe.get("rules") or []
    preview = preview_xml_tabularization(
        source_path,
        artifact,
        root,
        selected,
        rules=rules,
        max_rows=None,
        max_child_rows=None,
        max_output_rows=None,
    )
    expected_signature = str(recipe.get("preview_signature") or "")
    if preview.preview_signature != expected_signature:
        raise ValueError("recorded XML extraction recipe no longer reproduces its preview signature")

    from .xml_materialize import materialize_xml_tabularization

    target = output if output is not None else event.get("derived", {}).get("path")
    if not target:
        raise ValueError("recorded XML provenance has no derived output path")
    return materialize_xml_tabularization(
        source_path,
        artifact,
        root,
        selected,
        rules=rules,
        confirmed_preview_signature=preview.preview_signature,
        output=target,
        overwrite=overwrite,
    )


__all__ = [
    "build_xml_extraction_recipe",
    "persist_xml_extraction_provenance",
    "regenerate_xml_extraction",
]
