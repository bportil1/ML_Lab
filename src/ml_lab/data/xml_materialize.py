from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .provenance import describe_dataset
from .reporting import save_xml_structure
from .types import XmlMaterializationResult, XmlPreviewTable, XmlStructureArtifact
from .xml_preview import XmlPreviewError, preview_xml_tabularization
from .xml_rules import build_xml_collection_plan
from .xml_selection import build_xml_record_selection
from .xml_history import register_xml_extraction
from .xml_provenance import persist_xml_extraction_provenance


class XmlMaterializationError(ValueError):
    """Raised when a confirmed XML extraction cannot be materialized safely."""


_MATERIALIZATION_SCHEMA = "ml-lab.xml-materialization@1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return value


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return slug or "child"


def _unique_child_paths(destination: Path, tables: Iterable[XmlPreviewTable]) -> dict[str, Path]:
    used: set[str] = set()
    result: dict[str, Path] = {}
    for table in tables:
        base = _safe_slug(table.name)
        candidate = base
        counter = 2
        while candidate.casefold() in used:
            candidate = f"{base}_{counter}"
            counter += 1
        used.add(candidate.casefold())
        result[table.name] = destination.with_name(f"{destination.stem}__{candidate}{destination.suffix}")
    return result


def _atomic_write_table(path: Path, table: XmlPreviewTable, *, delimiter: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [column.name for column in table.columns]
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, delimiter=delimiter, extrasaction="ignore")
            writer.writeheader()
            for row in table.rows:
                writer.writerow({name: _json_cell(row.get(name)) for name in columns})
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _default_output(source: Path) -> Path:
    return source.with_name(f"{source.stem}-tabular.csv")


def materialize_xml_tabularization(
    path: str | Path,
    artifact: XmlStructureArtifact,
    record_root_canonical_path: str,
    selected_field_ids: Iterable[str],
    *,
    rules: Iterable[Mapping[str, Any]] | None = None,
    confirmed_preview_signature: str,
    output: str | Path | None = None,
    overwrite: bool = False,
    history_root: str | Path | None = None,
) -> XmlMaterializationResult:
    """Materialize one confirmed XML extraction as a normal ML Lab tabular dataset.

    The main result is CSV/TSV and can immediately enter the ordinary Data Lab pathways.
    The analyzed XML structure is persisted separately. Authoritative lineage remains in the
    provenance sidecar; an optional history index only makes sibling extractions discoverable.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"not a file: {source}")
    if source.suffix.casefold() != ".xml":
        raise XmlMaterializationError("XML materialization source must end in .xml")
    confirmed = str(confirmed_preview_signature or "").strip()
    if not confirmed:
        raise XmlMaterializationError("a confirmed preview signature is required before materialization")

    selected_tuple = tuple(dict.fromkeys(str(field_id) for field_id in selected_field_ids))
    if not selected_tuple:
        raise XmlMaterializationError("select at least one XML field before materialization")

    selection = build_xml_record_selection(artifact, record_root_canonical_path)
    plan = build_xml_collection_plan(artifact, record_root_canonical_path, selected_tuple, rules=rules)
    if not plan.ready_for_preview:
        raise XmlMaterializationError(
            "repeated-branch rules are not ready for materialization "
            f"({plan.unresolved_branch_count} unresolved, {plan.invalid_rule_count} invalid)"
        )

    try:
        full = preview_xml_tabularization(
            source,
            artifact,
            record_root_canonical_path,
            selected_tuple,
            rules=rules,
            max_rows=None,
            max_child_rows=None,
            max_output_rows=None,
        )
    except XmlPreviewError as exc:
        raise XmlMaterializationError(str(exc)) from exc

    if full.preview_signature != confirmed:
        raise XmlMaterializationError(
            "the confirmed preview no longer matches the current source/root/fields/rules; rebuild and confirm the preview"
        )
    if full.main_table is None:
        raise XmlMaterializationError("XML extraction did not produce a main table")
    if full.main_table.truncated or any(table.truncated for table in full.child_tables):
        raise XmlMaterializationError("materialization unexpectedly produced a truncated table")

    destination = Path(output).expanduser() if output is not None else _default_output(source)
    destination = destination.resolve()
    if destination.suffix.casefold() not in {".csv", ".tsv"}:
        raise XmlMaterializationError("XML materialized dataset output must end in .csv or .tsv")
    if destination == source:
        raise XmlMaterializationError("XML materialization cannot overwrite the source XML")

    structure_path = destination.with_suffix(destination.suffix + ".xml-structure.json")
    manifest_path = destination.with_suffix(destination.suffix + ".xml-extraction.json")
    child_paths = _unique_child_paths(destination, full.child_tables)
    all_outputs = [destination, structure_path, manifest_path, *child_paths.values()]
    existing = [candidate for candidate in all_outputs if candidate.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "XML materialization artifact already exists: "
            + ", ".join(str(candidate) for candidate in existing)
            + "; pass overwrite=true explicitly"
        )

    delimiter = "\t" if destination.suffix.casefold() == ".tsv" else ","
    _atomic_write_table(destination, full.main_table, delimiter=delimiter)
    for table in full.child_tables:
        _atomic_write_table(child_paths[table.name], table, delimiter=delimiter)
    save_xml_structure(artifact, structure_path)

    dataset = describe_dataset(destination)
    child_datasets = [describe_dataset(child_paths[table.name]) for table in full.child_tables]
    created_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema": _MATERIALIZATION_SCHEMA,
        "created_at": created_at,
        "source": {
            "path": str(source),
            "sha256": artifact.source_sha256,
            "fingerprint": artifact.source_fingerprint,
            "structure_schema": artifact.schema,
        },
        "extraction": {
            "preview_signature": full.preview_signature,
            "record_selection": selection.to_record(),
            "collection_plan": plan.to_record(),
        },
        "dataset": dataset,
        "child_datasets": child_datasets,
        "structure_artifact": {
            "path": str(structure_path),
            "sha256": _sha256(structure_path),
            "schema": artifact.schema,
        },
        "warnings": list(full.warnings),
        "provenance": {
            "status": "not_recorded",
            "note": "This XML extraction manifest is not an authoritative lineage sidecar.",
        },
    }
    _atomic_write_json(manifest_path, manifest)

    provenance = persist_xml_extraction_provenance(
        source_path=source,
        artifact=artifact,
        derived_path=destination,
        structure_artifact_path=structure_path,
        manifest_path=manifest_path,
        child_table_paths=[child_paths[table.name] for table in full.child_tables],
        record_root_canonical_path=record_root_canonical_path,
        selected_field_ids=selected_tuple,
        rules=rules,
        preview_signature=full.preview_signature,
        warnings=full.warnings,
        created_at=created_at,
    )
    history = None
    if history_root is not None:
        history = register_xml_extraction(
            provenance["provenance_path"],
            artifact=artifact,
            history_root=history_root,
        )
    manifest["provenance"] = {
        "status": "recorded",
        "path": provenance["provenance_path"],
        "event_id": provenance["event_id"],
        "transformation_id": provenance["transformation_id"],
        "extraction_family_id": (provenance.get("extraction_family") or {}).get("family_id"),
        "history_path": history.get("history_path") if history else None,
    }
    _atomic_write_json(manifest_path, manifest)

    return XmlMaterializationResult(
        source_path=str(source),
        source_fingerprint=artifact.source_fingerprint,
        preview_signature=full.preview_signature,
        dataset_path=str(destination),
        structure_artifact_path=str(structure_path),
        manifest_path=str(manifest_path),
        provenance_path=str(provenance["provenance_path"]),
        row_count=int(dataset["rows"]),
        column_count=int(dataset["columns"]),
        columns=tuple(str(column) for column in dataset["column_names"]),
        child_table_paths=tuple(str(child_paths[table.name]) for table in full.child_tables),
        warnings=tuple(full.warnings),
    )


__all__ = ["XmlMaterializationError", "materialize_xml_tabularization"]
