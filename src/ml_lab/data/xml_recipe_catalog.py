"""P5-16: immutable, opt-in collection-recipe snapshots and disposable SQLite lookup.

This indexes original C4 recipe *evidence*. It cannot authorize extraction,
merge, dataset registration, or claim source freshness after inventory.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Mapping

from .xml_catalog_index import _open, _project
from .xml_source_catalog import read_xml_inventory_snapshot

SCHEMA = "ml-lab.xml-recipe-catalog@1"
SNAPSHOT_SCHEMA = "ml-lab.xml-recipe-snapshot@1"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_MAX_BYTES = 2 * 1024 * 1024
_MAX_RECIPES = 2000
_MAX_SOURCES = 1000


def _dir(root: Path) -> Path:
    return root / "artifacts" / "xml_collection_recipes"


def _bytes(doc: Any) -> bytes:
    return (json.dumps(doc, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _verified_recipe(recipe: Any) -> list[tuple[str, str]]:
    """Validate the constrained provenance subset; do not reinterpret C4 rules."""
    if not isinstance(recipe, dict) or recipe.get("schema") != "ml-lab.xml-collection-recipe@1":
        raise ValueError("Unsupported XML collection recipe")
    if (recipe.get("status") != "ready_for_collection_preview"
            or recipe.get("structural_recipe_validated") is not True
            or recipe.get("merge_authorized") is not False
            or recipe.get("csv_generated") is not False
            or recipe.get("blocking_reasons") != []):
        raise ValueError("Only structurally validated, non-authorizing C4 recipes may be stored")
    if not isinstance(recipe.get("root_path"), str) or not recipe["root_path"]:
        raise ValueError("Missing collection root")
    if not isinstance(recipe.get("record_root_canonical_path"), str) or not recipe["record_root_canonical_path"]:
        raise ValueError("Missing record root")
    fields = recipe.get("selected_field_ids")
    if not isinstance(fields, list) or not fields or not all(isinstance(f, str) and f for f in fields) or len(set(fields)) != len(fields):
        raise ValueError("Invalid recipe field identifiers")
    sources = recipe.get("selected_sources")
    if not isinstance(sources, list) or not 1 <= len(sources) <= _MAX_SOURCES:
        raise ValueError("Invalid recipe source list")
    found = set()
    result = []
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Invalid recipe source")
        path = source.get("relative_path")
        fingerprint = source.get("source_fingerprint")
        if (not isinstance(path, str) or not path or path.startswith("/") or "\\" in path
                or any(part in {"", ".", ".."} for part in path.split("/")) or path in found):
            raise ValueError("Invalid or duplicate recipe source path")
        if not isinstance(fingerprint, str) or not fingerprint.startswith("sha256:") or not _HEX.fullmatch(fingerprint[7:]):
            raise ValueError("Invalid recipe source fingerprint")
        found.add(path)
        result.append((path, fingerprint[7:]))
    return result


def _checked_doc(doc: Any) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    if not isinstance(doc, dict) or doc.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError("Unsupported recipe snapshot")
    recipe = doc.get("recipe")
    sources = _verified_recipe(recipe)
    if doc.get("scope") != "observed_validated_recipe" or doc.get("registered_dataset") is not False:
        raise ValueError("Recipe snapshot scope is invalid")
    inventory_id = doc.get("inventory_snapshot_id")
    if inventory_id is not None and (not isinstance(inventory_id, str) or not _HEX.fullmatch(inventory_id)):
        raise ValueError("Invalid linked inventory")
    return recipe, sources


def _verify_link(root: Path, recipe: dict[str, Any], sources: list[tuple[str, str]], inventory_id: str | None) -> None:
    if inventory_id is None:
        return
    inventory = read_xml_inventory_snapshot(root, inventory_id)["inventory"]
    if inventory["root_path"] != recipe["root_path"]:
        raise ValueError("Inventory and recipe collection roots differ")
    by_path = {x["relative_path"]: x for x in inventory["files"]}
    for path, digest in sources:
        observed = by_path.get(path)
        if observed is None or observed["status"] != "parsed" or observed["sha256"] != digest:
            raise ValueError("Inventory and recipe source evidence differ")


def save_xml_recipe_snapshot(project_root: str | Path, recipe: Mapping[str, Any],
                             *, inventory_snapshot_id: str | None = None) -> dict[str, Any]:
    """Explicit immutable publication; source XML and C4 recipe files stay authoritative."""
    root = _project(project_root)
    doc = {"schema": SNAPSHOT_SCHEMA, "scope": "observed_validated_recipe",
           "registered_dataset": False, "inventory_snapshot_id": inventory_snapshot_id,
           "recipe": dict(recipe)}
    checked, sources = _checked_doc(doc)
    _verify_link(root, checked, sources, inventory_snapshot_id)
    data = _bytes(doc)
    if len(data) > _MAX_BYTES:
        raise ValueError("Recipe snapshot too large")
    digest = hashlib.sha256(data).hexdigest()
    directory = _dir(root)
    if directory.is_symlink():
        raise ValueError("Recipe snapshot directory must not be a symlink")
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / (digest + ".json")
    try:
        with dest.open("xb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    except FileExistsError:
        if dest.is_symlink() or dest.read_bytes() != data:
            raise ValueError("Existing recipe snapshot differs")
    return {"schema": SNAPSHOT_SCHEMA, "snapshot_id": digest,
            "relative_path": f"artifacts/xml_collection_recipes/{digest}.json",
            "source_count": len(sources), "registered_dataset": False}


def read_xml_recipe_snapshot(project_root: str | Path, snapshot_id: str) -> dict[str, Any]:
    root = _project(project_root)
    if not isinstance(snapshot_id, str) or not _HEX.fullmatch(snapshot_id):
        raise ValueError("Invalid recipe snapshot ID")
    dest = _dir(root) / (snapshot_id + ".json")
    if dest.is_symlink():
        raise ValueError("Recipe snapshot must not be a symlink")
    with dest.open("rb") as f:
        data = f.read(_MAX_BYTES + 1)
    if len(data) > _MAX_BYTES or hashlib.sha256(data).hexdigest() != snapshot_id:
        raise ValueError("Recipe snapshot integrity mismatch")
    doc = json.loads(data)
    recipe, sources = _checked_doc(doc)
    _verify_link(root, recipe, sources, doc["inventory_snapshot_id"])
    return doc


def _tables(db: sqlite3.Connection) -> None:
    db.execute("""CREATE TABLE IF NOT EXISTS xml_recipes (
        snapshot_id TEXT PRIMARY KEY, inventory_snapshot_id TEXT,
        root_path TEXT NOT NULL, record_root TEXT NOT NULL,
        source_count INTEGER NOT NULL, field_count INTEGER NOT NULL,
        relative_path TEXT NOT NULL)""")
    db.execute("""CREATE TABLE IF NOT EXISTS xml_recipe_sources (
        snapshot_id TEXT NOT NULL REFERENCES xml_recipes(snapshot_id),
        relative_path TEXT NOT NULL, sha256 TEXT NOT NULL,
        PRIMARY KEY(snapshot_id, relative_path))""")
    db.execute("CREATE INDEX IF NOT EXISTS ix_xml_recipe_sources_sha ON xml_recipe_sources(sha256)")
    db.execute("CREATE INDEX IF NOT EXISTS ix_xml_recipes_inventory ON xml_recipes(inventory_snapshot_id)")


def rebuild_xml_recipe_catalog(project_root: str | Path) -> dict[str, Any]:
    """Recreate only recipe tables, transactionally, from authenticated snapshots."""
    root = _project(project_root)
    folder = _dir(root)
    recipes, sources, invalid = [], [], 0
    if folder.is_dir() and not folder.is_symlink():
        candidates = sorted(folder.glob("*.json"))
        if len(candidates) > _MAX_RECIPES:
            raise ValueError("Too many XML recipe snapshots")
        for path in candidates:
            if not _HEX.fullmatch(path.stem) or path.is_symlink() or not path.is_file():
                invalid += 1
                continue
            try:
                doc = read_xml_recipe_snapshot(root, path.stem)
                recipe, items = _checked_doc(doc)
                recipes.append((path.stem, doc["inventory_snapshot_id"],
                                recipe["root_path"], recipe["record_root_canonical_path"],
                                len(items), len(recipe["selected_field_ids"]),
                                f"artifacts/xml_collection_recipes/{path.name}"))
                sources.extend((path.stem, rel, sha) for rel, sha in items)
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                invalid += 1
    db = _open(root)
    try:
        with db:
            _tables(db)
            db.execute("DELETE FROM xml_recipe_sources")
            db.execute("DELETE FROM xml_recipes")
            db.executemany("INSERT INTO xml_recipes VALUES (?,?,?,?,?,?,?)", recipes)
            db.executemany("INSERT INTO xml_recipe_sources VALUES (?,?,?)", sources)
    finally:
        db.close()
    return {"schema": SCHEMA, "recipes": len(recipes), "source_links": len(sources),
            "invalid_skipped": invalid, "stale_until_rebuilt": True}


def list_xml_recipes(project_root: str | Path, *, inventory_snapshot_id: str | None = None,
                     source_sha256: str | None = None, limit: int = 50,
                     offset: int = 0) -> dict[str, Any]:
    root = _project(project_root)
    if type(limit) is not int or not 1 <= limit <= 500 or type(offset) is not int or offset < 0:
        raise ValueError("Invalid XML recipe pagination")
    for value in (inventory_snapshot_id, source_sha256):
        if value is not None and (not isinstance(value, str) or not _HEX.fullmatch(value)):
            raise ValueError("Invalid XML recipe filter")
    database = root / "analysis" / "ml_lab" / "xml_catalog.sqlite3"
    if not database.is_file():
        return {"schema": SCHEMA, "total": 0, "items": [], "stale_until_rebuilt": True}
    db = _open(root)
    try:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='xml_recipes'").fetchone():
            return {"schema": SCHEMA, "total": 0, "items": [], "stale_until_rebuilt": True}
        where, params = [], []
        if inventory_snapshot_id is not None:
            where.append("r.inventory_snapshot_id=?"); params.append(inventory_snapshot_id)
        if source_sha256 is not None:
            where.append("EXISTS (SELECT 1 FROM xml_recipe_sources s WHERE s.snapshot_id=r.snapshot_id AND s.sha256=?)")
            params.append(source_sha256)
        clause = " WHERE " + " AND ".join(where) if where else ""
        total = db.execute("SELECT COUNT(*) FROM xml_recipes r" + clause, params).fetchone()[0]
        rows = db.execute("SELECT r.* FROM xml_recipes r" + clause +
                          " ORDER BY r.snapshot_id LIMIT ? OFFSET ?", [*params, limit, offset]).fetchall()
        return {"schema": SCHEMA, "total": total, "items": [dict(row) for row in rows],
                "stale_until_rebuilt": True}
    finally:
        db.close()
