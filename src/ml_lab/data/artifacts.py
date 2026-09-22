from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import DataProfileCollection

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(value: str, *, fallback: str = "profile") -> str:
    cleaned = _SAFE.sub("-", value.strip()).strip("-._")
    return cleaned[:80] or fallback


def persist_profile_run(
    collection: DataProfileCollection,
    *,
    output_root: str | Path = "ml_lab_results/data/profile_runs",
    run_id: str,
) -> dict[str, Any]:
    """Persist one UI/application profiling run and each profile separately."""
    root = Path(output_root).expanduser().resolve() / _slug(run_id, fallback="run")
    profiles_dir = root / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    profile_refs: list[dict[str, Any]] = []
    for index, profile in enumerate(collection.profiles):
        name = f"{index:03d}-{_slug(Path(profile.relative_path).name)}.json"
        destination = profiles_dir / name
        destination.write_text(json.dumps(profile.to_record(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        profile_refs.append(
            {
                "index": index,
                "relative_path": profile.relative_path,
                "source_path": profile.path,
                "source_sha256": profile.source_sha256,
                "profile_path": str(destination),
            }
        )

    collection_path = root / "profile_collection.json"
    collection_path.write_text(json.dumps(collection.to_record(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema": "ml-lab.data-profile-run@1",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "collection_path": str(collection_path),
        "profiles": profile_refs,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def load_profile_run(
    *,
    output_root: str | Path = "ml_lab_results/data/profile_runs",
    run_id: str,
) -> dict[str, Any]:
    root = Path(output_root).expanduser().resolve() / _slug(run_id, fallback="run")
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"profile run not found: {run_id}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["manifest_path"] = str(path)
    return manifest


def recent_profile_runs(
    *,
    output_root: str | Path = "ml_lab_results/data/profile_runs",
    limit: int = 20,
) -> list[dict[str, Any]]:
    root = Path(output_root).expanduser().resolve()
    if not root.is_dir():
        return []
    candidates = sorted(
        (path for path in root.glob("*/manifest.json") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    runs: list[dict[str, Any]] = []
    for path in candidates[: max(0, limit)]:
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        manifest["manifest_path"] = str(path)
        runs.append(manifest)
    return runs


def load_persisted_profile(manifest: dict[str, Any], index: int) -> dict[str, Any]:
    profiles = manifest.get("profiles", [])
    if index < 0 or index >= len(profiles):
        raise IndexError("profile index out of range")
    path = Path(profiles[index]["profile_path"])
    return json.loads(path.read_text(encoding="utf-8"))
