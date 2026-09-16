from __future__ import annotations

import json
from pathlib import Path

from .types import DataInventory, DataProfileCollection


def save_inventory(inventory: DataInventory, output: str | Path) -> Path:
    destination = Path(output).expanduser()
    if destination.suffix.lower() != ".json":
        destination = destination / "inventory.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(inventory.to_record(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def save_profile(profile: DataProfileCollection, output: str | Path) -> Path:
    destination = Path(output).expanduser()
    if destination.suffix.lower() != ".json":
        destination = destination / "profile.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(profile.to_record(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
