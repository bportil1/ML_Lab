"""ML-UW-00: read-only route inventory and compatibility gate.

The audit can run without Flask, Torch or a configured ML_Lab project.
Later navigation changes must preserve the existing XML endpoints as adapters
until their callers are migrated and their contracts explicitly retired.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


_ROOT = Path(__file__).resolve().parents[1]


def _inventory() -> dict:
    completed = subprocess.run(
        [sys.executable, str(_ROOT / "tools" / "ml_uw_00_inventory.py")],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


def test_baseline_inventory_is_read_only_and_machine_readable():
    report = _inventory()
    assert report["audit_schema"] == "ml-lab.unified-workbench-source-inventory@1"
    assert report["project_version"]
    assert report["source_file_count"] > 0
    assert all(row["path"].startswith("/") for row in report["ui_routes"])
    assert all(report["component_presence"].values())


def test_xml_collection_legacy_endpoints_stay_discoverable():
    paths = set(_inventory()["xml_collection_route_paths"])
    assert {
        "/data/xml/collection",
        "/data/xml/collection/review",
        "/data/xml/collection/recipe",
        "/data/xml/collection/export",
        "/data/xml/collection/provenance",
    }.issubset(paths)
