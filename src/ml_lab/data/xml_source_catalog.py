"""P5-16: immutable XML collection inventory snapshots and rebuildable source index.

The original XML files remain authoritative. This module never reparses sources
while rebuilding; snapshots represent the observation at their recorded hashes.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any

from .xml_catalog_index import _open, _project
from .xml_collection import XmlCollectionInventory

SCHEMA = "ml-lab.xml-source-catalog@1"
SNAPSHOT_SCHEMA = "ml-lab.xml-inventory-snapshot@1"
_ID = re.compile(r"[0-9a-f]{64}\Z")
_MAX_SNAPSHOT_BYTES = 16 * 1024 * 1024
_MAX_SNAPSHOTS = 2000
_MAX_SOURCES = 100000


def _directory(root: Path) -> Path:
    return root / "artifacts" / "xml_collection_inventories"


def _encoded(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _check_inventory(doc: Any) -> dict[str, Any]:
    if not isinstance(doc, dict) or doc.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError("Unsupported XML inventory snapshot")
    inv = doc.get("inventory")
    if not isinstance(inv, dict) or inv.get("schema") != "ml-lab.xml-collection-inventory@1":
        raise ValueError("Unsupported XML collection inventory")
    root = inv.get("root_path")
    settings = inv.get("settings")
    files = inv.get("files")
    if not isinstance(root, str) or not root or not isinstance(settings, dict) or not isinstance(files, list):
        raise ValueError("Malformed XML inventory")
    if len(files) > _MAX_SOURCES:
        raise ValueError("Too many source entries")
    seen = set()
    for row in files:
        if not isinstance(row, dict):
            raise ValueError("Invalid XML source")
        rel, status, sha = row.get("relative_path"), row.get("status"), row.get("sha256")
        if not isinstance(rel, str) or not rel or rel.startswith("/") or "\\" in rel or ".." in rel.split("/") or rel in seen:
            raise ValueError("Invalid or duplicate XML source path")
        seen.add(rel)
        if status not in {"parsed", "failed"} or not isinstance(sha, str):
            raise ValueError("Invalid XML source status")
        if status == "parsed" and not _ID.fullmatch(sha):
            raise ValueError("Missing parsed-source fingerprint")
        if status == "failed" and sha and not _ID.fullmatch(sha):
            raise ValueError("Invalid failed-source fingerprint")
        size = row.get("size_bytes")
        if type(size) is not int or size < 0:
            raise ValueError("Invalid source size")
    summary = inv.get("summary")
    if not isinstance(summary, dict) or summary.get("file_count") != len(files):
        raise ValueError("Inventory file count mismatch")
    if summary.get("parsed_file_count") != sum(x["status"] == "parsed" for x in files):
        raise ValueError("Inventory parsed count mismatch")
    if summary.get("failed_file_count") != sum(x["status"] == "failed" for x in files):
        raise ValueError("Inventory failed count mismatch")
    return inv


def save_xml_inventory_snapshot(project_root: str | Path,
                                inventory: XmlCollectionInventory) -> dict[str, Any]:
    """Explicitly publish a bounded immutable observation of a collection."""
    root = _project(project_root)
    if not isinstance(inventory, XmlCollectionInventory):
        raise TypeError("Expected existing XML collection inventory")
    doc = {"schema": SNAPSHOT_SCHEMA, "inventory": inventory.to_record(),
           "scope": "observed_inventory", "registered_dataset": False}
    _check_inventory(doc)
    data = _encoded(doc)
    if len(data) > _MAX_SNAPSHOT_BYTES:
        raise ValueError("XML inventory snapshot exceeds size limit")
    digest = hashlib.sha256(data).hexdigest()
    directory = _directory(root)
    if directory.is_symlink():
        raise ValueError("XML snapshot directory must not be a symlink")
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / (digest + ".json")
    try:
        with dest.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if dest.is_symlink() or dest.read_bytes() != data:
            raise ValueError("Existing XML inventory snapshot differs")
    return {"schema": SNAPSHOT_SCHEMA, "snapshot_id": digest,
            "relative_path": f"artifacts/xml_collection_inventories/{digest}.json",
            "source_count": len(inventory.files), "registered_dataset": False}


def read_xml_inventory_snapshot(project_root: str | Path, snapshot_id: str) -> dict[str, Any]:
    root = _project(project_root)
    if not isinstance(snapshot_id, str) or not _ID.fullmatch(snapshot_id):
        raise ValueError("Invalid XML inventory ID")
    dest = _directory(root) / (snapshot_id + ".json")
    if dest.is_symlink():
        raise ValueError("XML snapshot must not be a symlink")
    with dest.open("rb") as stream:
        data = stream.read(_MAX_SNAPSHOT_BYTES + 1)
    if len(data) > _MAX_SNAPSHOT_BYTES or hashlib.sha256(data).hexdigest() != snapshot_id:
        raise ValueError("XML inventory snapshot integrity mismatch")
    doc = json.loads(data)
    _check_inventory(doc)
    return doc


def _ensure_tables(db: sqlite3.Connection) -> None:
    db.execute("""CREATE TABLE IF NOT EXISTS xml_collections (
      snapshot_id TEXT PRIMARY KEY, root_path TEXT NOT NULL,
      file_count INTEGER NOT NULL, parsed_count INTEGER NOT NULL,
      failed_count INTEGER NOT NULL, relative_path TEXT NOT NULL)""")
    db.execute("""CREATE TABLE IF NOT EXISTS xml_sources (
      snapshot_id TEXT NOT NULL REFERENCES xml_collections(snapshot_id),
      relative_path TEXT NOT NULL, status TEXT NOT NULL,
      sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL,
      PRIMARY KEY(snapshot_id,relative_path))""")
    db.execute("CREATE INDEX IF NOT EXISTS ix_xml_sources_sha ON xml_sources(sha256, status)")


def rebuild_xml_source_catalog(project_root: str | Path) -> dict[str, Any]:
    """Index only authenticated snapshots in one transaction; no filesystem mutation."""
    root = _project(project_root)
    folder = _directory(root)
    collections: list[tuple[Any, ...]] = []
    sources: list[tuple[Any, ...]] = []
    invalid = 0
    if folder.is_dir() and not folder.is_symlink():
        candidates = sorted(folder.glob("*.json"))
        if len(candidates) > _MAX_SNAPSHOTS:
            raise ValueError("Too many XML inventory snapshots to index")
        for path in candidates:
            if not _ID.fullmatch(path.stem) or path.is_symlink() or not path.is_file():
                invalid += 1
                continue
            try:
                inv = read_xml_inventory_snapshot(root, path.stem)["inventory"]
                summary = inv["summary"]
                collections.append((path.stem, inv["root_path"], summary["file_count"],
                                    summary["parsed_file_count"], summary["failed_file_count"],
                                    f"artifacts/xml_collection_inventories/{path.name}"))
                for row in inv["files"]:
                    sources.append((path.stem, row["relative_path"], row["status"],
                                    row["sha256"], row["size_bytes"]))
                if len(sources) > _MAX_SOURCES:
                    raise ValueError("Too many XML catalog source rows")
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                invalid += 1
    db = _open(root)
    try:
        with db:
            _ensure_tables(db)
            db.execute("DELETE FROM xml_sources")
            db.execute("DELETE FROM xml_collections")
            db.executemany("INSERT INTO xml_collections VALUES (?,?,?,?,?,?)", collections)
            db.executemany("INSERT INTO xml_sources VALUES (?,?,?,?,?)", sources)
    finally:
        db.close()
    return {"schema": SCHEMA, "collections": len(collections), "sources": len(sources),
            "invalid_skipped": invalid, "stale_until_rebuilt": True}


def list_xml_sources(project_root: str | Path, *, snapshot_id: str | None = None,
                     sha256: str | None = None, limit: int = 50, offset: int = 0) -> dict[str, Any]:
    root = _project(project_root)
    if type(limit) is not int or not 1 <= limit <= 500 or type(offset) is not int or offset < 0:
        raise ValueError("Invalid XML source pagination")
    for val in (snapshot_id, sha256):
        if val is not None and (not isinstance(val, str) or not _ID.fullmatch(val)):
            raise ValueError("Invalid XML source filter")
    if not (root / "analysis/ml_lab/xml_catalog.sqlite3").is_file():
        return {"schema": SCHEMA, "total": 0, "items": [], "stale_until_rebuilt": True}
    db = _open(root)
    try:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='xml_sources'").fetchone():
            return {"schema": SCHEMA, "total": 0, "items": [], "stale_until_rebuilt": True}
        filters, params = [], []
        if snapshot_id is not None:
            filters.append("snapshot_id=?"); params.append(snapshot_id)
        if sha256 is not None:
            filters.append("sha256=?"); params.append(sha256)
        where = " WHERE " + " AND ".join(filters) if filters else ""
        total = db.execute("SELECT COUNT(*) FROM xml_sources" + where, params).fetchone()[0]
        rows = db.execute("SELECT * FROM xml_sources" + where +
                          " ORDER BY snapshot_id, relative_path LIMIT ? OFFSET ?",
                          [*params, limit, offset]).fetchall()
        return {"schema": SCHEMA, "total": total, "items": [dict(x) for x in rows],
                "stale_until_rebuilt": True}
    finally:
        db.close()
