"""Explicit, bounded XML catalog index refresh (not an XML source rescan).

Each section is independently transactional. A failure in one leaves its last
successful index intact and other sections may have refreshed; report partial
success accurately. Source files and immutable snapshots are never rewritten.
"""
from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Any

from .xml_catalog_index import _project, rebuild_xml_catalog
from .xml_source_catalog import rebuild_xml_source_catalog
from .xml_recipe_catalog import rebuild_xml_recipe_catalog

SCHEMA = "ml-lab.xml-catalog-refresh@1"
SCOPES = ("all", "sources", "recipes", "artifacts")


def refresh_xml_catalog(project_root: str | Path, *, scope: str = "all") -> dict[str, Any]:
    """Refresh index from existing snapshots, reporting exact per-index outcomes.

    No collection scan, recipe execution, or typed extraction is triggered.
    Each constituent rebuild is atomic, but a multi-section refresh is NOT.
    """
    if scope not in SCOPES:
        raise ValueError("Unsupported XML catalog refresh scope")
    root = _project(project_root)
    rebuilders = {
        "sources": rebuild_xml_source_catalog,
        "recipes": rebuild_xml_recipe_catalog,
        "artifacts": rebuild_xml_catalog,
    }
    selected = ("sources", "recipes", "artifacts") if scope == "all" else (scope,)
    results: dict[str, dict[str, Any]] = {}
    for name in selected:
        try:
            payload = rebuilders[name](root)
        except (ValueError, OSError, RuntimeError, sqlite3.Error) as exc:
            results[name] = {"status": "failed", "error": str(exc)}
        else:
            results[name] = {"status": "rebuilt", "counts": {
                key: value for key, value in payload.items()
                if key in ("indexed", "collections", "sources", "recipes", "source_links", "invalid_skipped")
            }}
    failures = [name for name, entry in results.items() if entry["status"] == "failed"]
    return {"schema": SCHEMA, "scope": scope, "sections": results,
            "complete": not failures, "partial": bool(failures) and len(failures) != len(selected),
            "failed_sections": failures,
            "note": "Rebuild is per-section atomic, not atomic across all sections; existing artifact files remain authoritative."}
