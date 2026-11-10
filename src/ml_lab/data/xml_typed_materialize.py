"""Opt-in complete typed XML records, streamed to immutable project-local JSONL.

These artifacts are *representations*, never registered datasets or numerical encodings.
The source and existing C1--C7 extraction remain authoritative and untouched.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping
import xml.etree.ElementTree as ET

from .types import XmlStructureArtifact
from .xml_ingest import inspect_xml_structure
from .xml_typed_records import _matches, _parts, _render, _validate
from .xml_typed_representation import SCHEMA, RECIPE_SCHEMA, XmlTypedRepresentationError, represent_xml_element

SCHEMA_FULL = "ml-lab.xml-typed-materialization@1"
MAX_RECORDS = 1_000_000
MAX_OUTPUT_BYTES = 512 * 1024 * 1024


def _serialize(obj: Mapping[str, Any]) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _root(project_root: str | Path) -> Path:
    return Path(project_root).expanduser().resolve() / "artifacts" / "xml_typed_records"


def _source_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def materialize_typed_xml_records(
    project_root: str | Path, source_path: str | Path, artifact: XmlStructureArtifact,
    recipe: Mapping[str, Any], *, max_records: int = MAX_RECORDS,
    max_output_bytes: int = MAX_OUTPUT_BYTES, max_nodes_per_value: int = 10_000,
) -> dict[str, Any]:
    """Create a complete record stream or raise without publishing a partial artifact.

    ``record_index`` is source-order ordinal, not a stable identity across source edits.
    ``record_id`` is source-SHA-qualified ordinal. Source byte identity and the
    recipe must match the already-inspected profile; no external source is opened.
    """
    if not 1 <= max_records <= MAX_RECORDS or not 1 <= max_output_bytes <= MAX_OUTPUT_BYTES or not 1 <= max_nodes_per_value <= 100_000:
        raise XmlTypedRepresentationError("typed materialization limits exceeded")
    root_parts, selected, links = _validate(artifact, recipe)
    source = Path(source_path).expanduser().resolve()
    current = inspect_xml_structure(source)
    if current.source_sha256 != artifact.source_sha256 or current.source_fingerprint != artifact.source_fingerprint:
        raise XmlTypedRepresentationError("XML source differs from inspected structure")
    _validate(current, recipe)
    canonical_root = recipe["record_root_canonical_path"]
    expected = next(p.occurrence_count for p in current.element_profiles if p.canonical_path == canonical_root)
    if expected > max_records:
        raise XmlTypedRepresentationError("typed materialization record limit exceeded")
    directory = _root(project_root)
    directory.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    record_count = 0
    byte_count = 0
    digest = hashlib.sha256()
    stack: list[str] = []
    element_stack: list[ET.Element] = []
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=directory, prefix=".typed-", suffix=".tmp", delete=False) as out:
            temp_path = Path(out.name)
            with source.open("rb") as inp:
                for event, element in ET.iterparse(inp, events=("start", "end")):
                    if event == "start":
                        stack.append(str(element.tag))
                        element_stack.append(element)
                        continue
                    if tuple(stack) == root_parts:
                        index = record_count
                        variables: list[dict[str, Any]] = []
                        for field_path, kind in selected:
                            matches = _matches(element, _parts(field_path), root_parts)
                            variables.append({"canonical_path": field_path, "representation": kind,
                                              "present": bool(matches), "count": len(matches),
                                              "items": [_render(node, kind, field_path, address, max_nodes_per_value)
                                                        for node, address in matches]})
                        children: list[dict[str, Any]] = []
                        for child_path in links:
                            matches = _matches(element, _parts(child_path), root_parts)
                            children.append({"canonical_path": child_path, "count": len(matches),
                                             "records": [{"parent_record_index": index,
                                                          "occurrence_address": list(address),
                                                          "representation": represent_xml_element(
                                                              node, canonical_path=child_path,
                                                              max_nodes=max_nodes_per_value)["root"]}
                                                         for node, address in matches]})
                        record = {"record_index": index, "record_number": index + 1,
                                  "record_id": f"{current.source_sha256}:{canonical_root}:{index}",
                                  "record_root_canonical_path": canonical_root,
                                  "variables": variables, "linked_children": children}
                        encoded = _serialize(record)
                        byte_count += len(encoded)
                        if byte_count > max_output_bytes:
                            raise XmlTypedRepresentationError("typed materialization output byte limit exceeded")
                        out.write(encoded)
                        digest.update(encoded)
                        record_count += 1
                        element.clear()
                    elif not (len(stack) > len(root_parts) and tuple(stack[:len(root_parts)]) == root_parts):
                        # Discard finished non-record branches while retaining complete
                        # subtrees for active record roots and their selected descendants.
                        element.clear()
                    if len(element_stack) > 1 and not (len(stack) > len(root_parts) and tuple(stack[:len(root_parts)]) == root_parts):
                        element_stack[-2].remove(element)
                    stack.pop()
                    element_stack.pop()
            out.flush()
            os.fsync(out.fileno())
        if record_count != expected:
            raise XmlTypedRepresentationError("typed materialization record count differs from structure")
        if _source_hash(source) != current.source_sha256:
            raise XmlTypedRepresentationError("XML source changed during materialization")
        records_sha = digest.hexdigest()
        manifest = {"schema": SCHEMA_FULL, "representation_schema": SCHEMA,
                    "recipe_schema": RECIPE_SCHEMA, "recipe": dict(recipe),
                    "source_sha256": current.source_sha256,
                    "source_fingerprint": current.source_fingerprint,
                    "record_root_canonical_path": canonical_root,
                    "record_count": record_count, "source_record_count": expected,
                    "complete": True, "truncated": False, "encoding": "none",
                    "registered_dataset": False, "record_identity": "source_sha256+root_path+ordinal",
                    "records_sha256": records_sha, "records_bytes": byte_count,
                    "records_format": "jsonl"}
        manifest_bytes = _serialize(manifest)
        manifest_id = hashlib.sha256(manifest_bytes).hexdigest()
        records_path = directory / (records_sha + ".jsonl")
        manifest_path = directory / (manifest_id + ".json")
        # Never overwrite existing immutable content. Reused names must verify.
        if records_path.exists() or records_path.is_symlink():
            if records_path.is_symlink() or _source_hash(records_path) != records_sha:
                raise ValueError("Existing typed records artifact integrity mismatch")
        else:
            os.link(temp_path, records_path)  # no-clobber, atomic same-filesystem publish
        try:
            with manifest_path.open("xb") as stream:
                stream.write(manifest_bytes)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError:
            if manifest_path.is_symlink() or manifest_path.read_bytes() != manifest_bytes:
                raise ValueError("Existing typed records manifest integrity mismatch")
        return {"schema": SCHEMA_FULL, "artifact_id": manifest_id,
                "record_count": record_count, "complete": True, "registered_dataset": False,
                "relative_manifest_path": str(Path("artifacts") / "xml_typed_records" / (manifest_id + ".json")),
                "relative_records_path": str(Path("artifacts") / "xml_typed_records" / (records_sha + ".jsonl"))}
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def verify_typed_xml_materialization(project_root: str | Path, artifact_id: str) -> dict[str, Any]:
    """Verify manifest and complete data stream without loading the records into RAM."""
    if not isinstance(artifact_id, str) or len(artifact_id) != 64 or any(c not in "0123456789abcdef" for c in artifact_id):
        raise ValueError("Invalid typed materialization ID")
    directory = _root(project_root)
    path = directory / (artifact_id + ".json")
    if path.is_symlink():
        raise ValueError("Typed manifest cannot be a symlink")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != artifact_id or len(raw) > 1_000_000:
        raise ValueError("Typed manifest integrity mismatch")
    manifest = json.loads(raw)
    if (not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA_FULL or
            manifest.get("complete") is not True or manifest.get("truncated") is not False or
            manifest.get("encoding") != "none" or manifest.get("registered_dataset") is not False):
        raise ValueError("Unsupported typed materialization manifest")
    digest_id = manifest.get("records_sha256")
    if not isinstance(digest_id, str) or len(digest_id) != 64 or any(c not in "0123456789abcdef" for c in digest_id):
        raise ValueError("Invalid records SHA-256")
    records_path = directory / (digest_id + ".jsonl")
    if records_path.is_symlink() or records_path.stat().st_size != manifest.get("records_bytes"):
        raise ValueError("Typed record stream integrity mismatch")
    if _source_hash(records_path) != digest_id:
        raise ValueError("Typed record stream integrity mismatch")
    return manifest
