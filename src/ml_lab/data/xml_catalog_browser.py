"""P5-16 read-only, bounded catalog lookups for the XML workspace.

The index is disposable; catalog browsing never opens source XML, creates
snapshots, registers datasets, or implicitly rebuilds indexes.
"""
from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Any

from .xml_catalog_index import _open, _project, list_xml_catalog
from .xml_source_catalog import list_xml_sources
from .xml_recipe_catalog import list_xml_recipes

SCHEMA = "ml-lab.xml-catalog-browser@1"
SECTIONS = ("collections", "sources", "recipes", "artifacts")


def browse_xml_catalog(project_root: str | Path, *, section: str = "collections",
                       source_sha256: str | None = None,
                       inventory_snapshot_id: str | None = None,
                       artifact_kind: str | None = None,
                       limit: int = 25, offset: int = 0) -> dict[str, Any]:
    """Return one page; no automatic rebuild and no unbounded full-table fetch."""
    root = _project(project_root)
    if section not in SECTIONS:
        raise ValueError("Unsupported XML catalog section")
    if type(limit) is not int or not 1 <= limit <= 100 or type(offset) is not int or offset < 0:
        raise ValueError("Invalid XML catalog pagination")
    if section == "sources":
        result = list_xml_sources(root, snapshot_id=inventory_snapshot_id,
                                  sha256=source_sha256, limit=limit, offset=offset)
    elif section == "recipes":
        result = list_xml_recipes(root, inventory_snapshot_id=inventory_snapshot_id,
                                  source_sha256=source_sha256, limit=limit, offset=offset)
    elif section == "artifacts":
        result = list_xml_catalog(root, source_sha256=source_sha256,
                                  kind=artifact_kind, limit=limit, offset=offset)
    else:
        if source_sha256 or inventory_snapshot_id or artifact_kind:
            raise ValueError("Filters are not supported for collection snapshots")
        db_path = root / "analysis/ml_lab/xml_catalog.sqlite3"
        if not db_path.is_file():
            result = {"total": 0, "items": [], "stale_until_rebuilt": True}
        else:
            db = _open(root)
            try:
                exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='xml_collections'").fetchone()
                if not exists:
                    result = {"total": 0, "items": [], "stale_until_rebuilt": True}
                else:
                    total = db.execute("SELECT COUNT(*) FROM xml_collections").fetchone()[0]
                    items = [dict(row) for row in db.execute(
                        "SELECT * FROM xml_collections ORDER BY snapshot_id LIMIT ? OFFSET ?",
                        (limit, offset))]
                    result = {"total": total, "items": items, "stale_until_rebuilt": True}
            finally:
                db.close()
    return {"schema": SCHEMA, "section": section, "limit": limit, "offset": offset,
            "total": result["total"], "items": result["items"],
            "stale_until_rebuilt": result["stale_until_rebuilt"]}
