"""Project-local immutable typed-XML *preview snapshots* (P5-15).

No dataset registration, scientific encoding or legacy XML materialization.
Caller supplies a trusted preview returned by ``preview_typed_xml_records``;
this module never dereferences source paths from an artifact.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .xml_typed_records import PREVIEW_SCHEMA

ARTIFACT_SCHEMA = "ml-lab.xml-typed-preview-snapshot@1"
_MAX_BYTES = 8 * 1024 * 1024


def _encoded(document: Mapping[str, Any]) -> bytes:
    return (json.dumps(document, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _validated(preview: Mapping[str, Any]) -> None:
    if not isinstance(preview, Mapping) or preview.get("schema") != PREVIEW_SCHEMA:
        raise ValueError("Only versioned typed XML previews may be saved")
    if preview.get("registered_dataset") is not False or preview.get("encoding") != "none":
        raise ValueError("Preview cannot be encoded or registered as a dataset")
    if not isinstance(preview.get("source_sha256"), str) or len(preview["source_sha256"]) != 64:
        raise ValueError("Missing verified XML source fingerprint")
    if not isinstance(preview.get("source_fingerprint"), str):
        raise ValueError("Missing XML source fingerprint")
    records = preview.get("records")
    if not isinstance(records, list) or not 0 <= len(records) <= 200:
        raise ValueError("Invalid bounded preview records")
    if preview.get("preview_record_count") != len(records):
        raise ValueError("Preview record count mismatch")
    if type(preview.get("truncated")) is not bool:
        raise ValueError("Preview completeness marker missing")


def _artifact_dir(project_root: str | Path) -> Path:
    root = Path(project_root).expanduser().resolve()
    return root / "artifacts" / "xml_typed_previews"


def persist_typed_preview(project_root: str | Path, preview: Mapping[str, Any],
                          *, recipe: Mapping[str, Any]) -> dict[str, Any]:
    """Create or reuse a content-addressed, project-local, read-only snapshot.

    A snapshot is explicitly bounded; ``truncated`` prevents accidental claims
    that its records constitute a complete extraction. Files are created without
    overwrite and must not be edited after publication.
    """
    _validated(preview)
    if not isinstance(recipe, Mapping) or recipe.get("schema") != "ml-lab.xml-typed-recipe@1" or recipe.get("version") != 1:
        raise ValueError("Versioned typed recipe required")
    if recipe.get("record_root_canonical_path") != preview.get("record_root_canonical_path"):
        raise ValueError("Recipe/preview record root mismatch")
    if recipe.get("encoding") != "none" or recipe.get("legacy_extraction_modified") is not False:
        raise ValueError("Cannot persist encoded or legacy-modifying recipe")
    doc = {"schema": ARTIFACT_SCHEMA, "preview": dict(preview), "recipe": dict(recipe),
           "scope": "bounded_snapshot", "registered_dataset": False}
    data = _encoded(doc)
    if len(data) > _MAX_BYTES:
        raise ValueError("Typed preview exceeds maximum snapshot size")
    digest = hashlib.sha256(data).hexdigest()
    directory = _artifact_dir(project_root)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / (digest + ".json")
    try:
        with destination.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        # File already present from a repeated save; verify instead of trusting.
        if destination.is_symlink() or destination.read_bytes() != data:
            raise ValueError("Existing typed preview artifact differs from expected content")
    return {"schema": ARTIFACT_SCHEMA, "artifact_id": digest,
            "relative_path": str(Path("artifacts") / "xml_typed_previews" / (digest + ".json")),
            "source_sha256": preview["source_sha256"],
            "record_count": len(preview["records"]), "truncated": preview["truncated"],
            "registered_dataset": False}


def read_typed_preview(project_root: str | Path, artifact_id: str) -> dict[str, Any]:
    """Load and integrity-check a previously saved snapshot without source reads."""
    if not isinstance(artifact_id, str) or len(artifact_id) != 64 or any(c not in "0123456789abcdef" for c in artifact_id):
        raise ValueError("Invalid typed preview artifact ID")
    path = _artifact_dir(project_root) / (artifact_id + ".json")
    if path.is_symlink():
        raise ValueError("Typed preview artifact may not be a symlink")
    with path.open("rb") as handle:
        data = handle.read(_MAX_BYTES + 1)
    if len(data) > _MAX_BYTES or hashlib.sha256(data).hexdigest() != artifact_id:
        raise ValueError("Typed preview artifact integrity mismatch")
    doc = json.loads(data)
    if not isinstance(doc, dict) or doc.get("schema") != ARTIFACT_SCHEMA or doc.get("scope") != "bounded_snapshot":
        raise ValueError("Unsupported typed preview artifact")
    _validated(doc.get("preview"))
    return doc
