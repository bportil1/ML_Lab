"""P5-16: rebuildable project-local catalog for immutable typed XML artifacts.

Authoritative source is the project artifact filesystem. The index is disposable;
no scientific datasets, original XML sources, or manifests are modified.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
import sqlite3
from typing import Any

from .xml_typed_artifacts import read_typed_preview
from .xml_typed_materialize import verify_typed_xml_materialization

SCHEMA = "ml-lab.xml-catalog-index@1"
_HEX = re.compile(r"[0-9a-f]{64}\Z")


class XmlCatalogError(ValueError):
    """Invalid index input or an unsupported on-disk index version."""


def _project(root: str | Path) -> Path:
    path = Path(root).expanduser().resolve()
    if not path.is_dir():
        raise XmlCatalogError("Project root does not exist")
    return path


def _path(root: Path) -> Path:
    return root / "analysis" / "ml_lab" / "xml_catalog.sqlite3"


def _open(root: Path) -> sqlite3.Connection:
    dest = _path(root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink():
        raise XmlCatalogError("Catalog database must not be a symlink")
    db = sqlite3.connect(dest, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA busy_timeout=30000")
    version = db.execute("PRAGMA user_version").fetchone()[0]
    if version > 1:
        db.close()
        raise XmlCatalogError("Catalog schema is newer than supported version")
    if version == 0:
        try:
            with db:
                db.execute("""CREATE TABLE IF NOT EXISTS xml_artifacts (
                    kind TEXT NOT NULL CHECK (kind IN ('preview', 'materialization')),
                    artifact_id TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL,
                    source_fingerprint TEXT NOT NULL,
                    record_root TEXT NOT NULL,
                    record_count INTEGER NOT NULL,
                    complete INTEGER NOT NULL,
                    relative_path TEXT NOT NULL,
                    PRIMARY KEY (kind, artifact_id))""")
                db.execute("CREATE INDEX IF NOT EXISTS ix_xml_artifacts_source ON xml_artifacts(source_sha256, kind)")
                db.execute("""CREATE TABLE IF NOT EXISTS xml_catalog_metadata (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL)""")
                db.execute("PRAGMA user_version=1")
        except BaseException:
            db.close()
            raise
    return db


def _candidates(root: Path, kind: str):
    relative = "xml_typed_previews" if kind == "preview" else "xml_typed_records"
    directory = root / "artifacts" / relative
    if not directory.is_dir() or directory.is_symlink():
        return
    for file in sorted(directory.glob("*.json")):
        if _HEX.fullmatch(file.stem) and not file.is_symlink() and file.is_file():
            yield file.stem


def rebuild_xml_catalog(project_root: str | Path) -> dict[str, Any]:
    """Rebuild in a single transaction; reject invalid artifacts from the index.

    Readers never observe a partly replaced catalog. Verification reads immutable
    file artifacts; full-record verification hashes JSONL but does not load it.
    """
    root = _project(project_root)
    rows: list[tuple[Any, ...]] = []
    invalid = 0
    for kind in ("preview", "materialization"):
        for artifact_id in _candidates(root, kind):
            try:
                if kind == "preview":
                    doc = read_typed_preview(root, artifact_id)
                    preview = doc["preview"]
                    source_sha = preview["source_sha256"]
                    fingerprint = preview["source_fingerprint"]
                    record_root = preview["record_root_canonical_path"]
                    count = preview["preview_record_count"]
                    complete = not preview["truncated"]
                    relative = f"artifacts/xml_typed_previews/{artifact_id}.json"
                else:
                    manifest = verify_typed_xml_materialization(root, artifact_id)
                    source_sha = manifest["source_sha256"]
                    fingerprint = manifest["source_fingerprint"]
                    record_root = manifest["record_root_canonical_path"]
                    count = manifest["record_count"]
                    complete = manifest["complete"]
                    relative = f"artifacts/xml_typed_records/{artifact_id}.json"
                if (not isinstance(source_sha, str) or not _HEX.fullmatch(source_sha)
                    or not isinstance(fingerprint, str) or not isinstance(record_root, str)
                    or not isinstance(count, int) or isinstance(count, bool) or count < 0
                    or not isinstance(complete, bool)):
                    raise ValueError("Invalid artifact catalog fields")
                rows.append((kind, artifact_id, source_sha, fingerprint,
                             record_root, count, int(complete), relative))
            except (ValueError, KeyError, TypeError, OSError):
                invalid += 1
    db = _open(root)
    try:
        with db:
            db.execute("DELETE FROM xml_artifacts")
            db.executemany("""INSERT INTO xml_artifacts
                (kind, artifact_id, source_sha256, source_fingerprint,
                 record_root, record_count, complete, relative_path)
                 VALUES (?,?,?,?,?,?,?,?)""", rows)
            db.execute("INSERT OR REPLACE INTO xml_catalog_metadata VALUES (?,?)",
                       ("last_rebuilt_utc", datetime.now(timezone.utc).isoformat()))
    finally:
        db.close()
    return {"schema": SCHEMA, "indexed": len(rows), "invalid_skipped": invalid,
            "index_relative_path": "analysis/ml_lab/xml_catalog.sqlite3"}


def list_xml_catalog(project_root: str | Path, *, source_sha256: str | None = None,
                     kind: str | None = None, limit: int = 50, offset: int = 0) -> dict[str, Any]:
    """Paged metadata lookup only. Call rebuild to refresh after artifact writes."""
    root = _project(project_root)
    if type(limit) is not int or not 1 <= limit <= 500 or type(offset) is not int or offset < 0:
        raise XmlCatalogError("Invalid catalog pagination")
    if kind is not None and kind not in {"preview", "materialization"}:
        raise XmlCatalogError("Invalid artifact kind")
    if source_sha256 is not None and (not isinstance(source_sha256, str) or not _HEX.fullmatch(source_sha256)):
        raise XmlCatalogError("Invalid source SHA-256")
    if not _path(root).is_file():
        return {"schema": SCHEMA, "total": 0, "items": [], "stale_until_rebuilt": True}
    db = _open(root)
    try:
        where = []
        params: list[Any] = []
        if kind is not None:
            where.append("kind=?")
            params.append(kind)
        if source_sha256 is not None:
            where.append("source_sha256=?")
            params.append(source_sha256)
        clause = " WHERE " + " AND ".join(where) if where else ""
        total = db.execute("SELECT COUNT(*) FROM xml_artifacts" + clause, params).fetchone()[0]
        items = [dict(row) for row in db.execute(
            "SELECT * FROM xml_artifacts" + clause +
            " ORDER BY kind, artifact_id LIMIT ? OFFSET ?", [*params, limit, offset])]
        return {"schema": SCHEMA, "total": total, "items": items,
                "stale_until_rebuilt": True}
    finally:
        db.close()
