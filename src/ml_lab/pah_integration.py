"""Thin PAH discovery/runtime adapter for ML Lab.

This module intentionally depends only on ML Lab and the Python standard library.
It returns mapping-shaped PAH contracts so ML Lab remains independently usable
and does not take a dependency on the PAH host package.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Mapping

from ml_lab import __version__

MODULE_ID = "ml_lab"
_DEFAULT_PORT = 8769


def module_manifest() -> dict[str, Any]:
    """Return ML Lab's host-facing capability declaration."""
    return {
        "module_id": MODULE_ID,
        "display_name": "ML Lab",
        "version": __version__,
        "description": (
            "Reusable local machine-learning and Data Lab engine with intake, profiling, "
            "controlled transformation, comparison, formal provenance, and model workflows."
        ),
        "collections": ["ml_lab"],
        "capabilities": [
            "data_intake",
            "data_profiling",
            "data_transformation",
            "dataset_comparison",
            "data_provenance",
            "classification",
            "regression",
            "clustering",
            "representation_learning",
            "energy_based_modeling",
            "generative_modeling",
        ],
        "interfaces": ["callable_api", "standalone_ui", "embedded_ui", "detachable_ui"],
        "metadata": {
            "ui_owner": "ml_lab",
            "local_first": True,
            "artifact_discovery": True,
        },
    }


def _context_value(context: Any, name: str, default: Any = None) -> Any:
    if context is None:
        return default
    if isinstance(context, Mapping):
        return context.get(name, default)
    return getattr(context, name, default)


def _roots(context: Any) -> tuple[Path | None, Path]:
    project_raw = _context_value(context, "project_root")
    results_raw = _context_value(context, "results_root")
    project = Path(project_raw).expanduser().resolve() if project_raw else None
    if results_raw:
        results = Path(results_raw).expanduser().resolve()
    elif project is not None:
        results = project / "ml_lab_results"
    else:
        results = Path("ml_lab_results").resolve()
    return project, results


def _port(context: Any) -> int:
    ports = _context_value(context, "ports", {}) or {}
    try:
        return int(dict(ports).get(MODULE_ID, _DEFAULT_PORT))
    except (TypeError, ValueError):
        return _DEFAULT_PORT


def _artifact_id(kind: str, location: Path, payload: Mapping[str, Any] | None = None) -> str:
    digest = hashlib.sha256()
    digest.update(str(location.resolve()).encode("utf-8"))
    if payload:
        stable = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
        digest.update(stable.encode("utf-8"))
    return f"ml-lab-{kind}-{digest.hexdigest()[:20]}"


def _schema_parts(schema: str) -> tuple[str, str | None]:
    raw = str(schema or "").strip()
    if "@" in raw:
        schema_id, version = raw.rsplit("@", 1)
        return schema_id, version
    return raw, None


_SCHEMA_KINDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "ml-lab.data-inventory": ("data_inventory", ("data_intake",)),
    "ml-lab.data-profile": ("data_profile", ("data_profiling",)),
    "ml-lab.data-profile-collection": ("data_profile_collection", ("data_profiling",)),
    "ml-lab.data-profile-run": ("data_profile_run", ("data_profiling",)),
    "ml-lab.transformation-recipe": ("transformation_recipe", ("data_transformation",)),
    "ml-lab.derived-dataset": ("derived_dataset", ("data_transformation", "data_provenance")),
    "ml-lab.dataset-comparison": ("dataset_comparison", ("dataset_comparison",)),
    "ml-lab.dataset-comparison-collection": ("dataset_comparison_collection", ("dataset_comparison",)),
    "ml-lab.transformation-event": ("data_provenance", ("data_provenance",)),
    "ml-lab.data-lineage": ("data_lineage", ("data_provenance",)),
}


def _artifact_from_json(path: Path, payload: Mapping[str, Any], project: Path | None) -> dict[str, Any] | None:
    schema_id, schema_version = _schema_parts(str(payload.get("schema") or ""))
    spec = _SCHEMA_KINDS.get(schema_id)
    if spec is None:
        return None
    kind, capabilities = spec
    location = path
    metadata: dict[str, Any] = {"runtime_managed": True, "registry_alias": False, "record_path": str(path)}
    provenance: dict[str, Any] = {}

    if schema_id == "ml-lab.derived-dataset":
        derived = payload.get("derived") or {}
        if isinstance(derived, Mapping) and derived.get("path"):
            location = Path(str(derived["path"])).expanduser().resolve()
            metadata["manifest_path"] = str(path)
            metadata["sha256"] = derived.get("sha256")
            metadata["dataset_id"] = derived.get("dataset_id")
        prov = payload.get("provenance") or {}
        if isinstance(prov, Mapping):
            provenance = dict(prov)
    elif schema_id == "ml-lab.transformation-event":
        derived = payload.get("derived") or {}
        if isinstance(derived, Mapping) and derived.get("path"):
            metadata["derived_path"] = str(derived.get("path"))
        source = payload.get("source") or {}
        if isinstance(source, Mapping) and source.get("path"):
            metadata["source_path"] = str(source.get("path"))
        metadata["event_id"] = payload.get("event_id")
        provenance = {
            "parent": payload.get("parent"),
            "recipe_sha256": payload.get("recipe_sha256"),
        }
    elif schema_id == "ml-lab.data-profile-run":
        metadata["run_id"] = payload.get("run_id")
        metadata["profile_count"] = len(payload.get("profiles") or [])

    return {
        "artifact_id": _artifact_id(kind, path, payload),
        "kind": kind,
        "producer_module": MODULE_ID,
        "location": str(location),
        "schema_id": schema_id,
        "schema_version": schema_version,
        "media_type": "application/json" if location == path else "text/csv",
        "producer_version": __version__,
        "project_id": str(project) if project else None,
        "capabilities": list(capabilities),
        "validation_state": "valid",
        "metadata": metadata,
        "provenance": provenance,
    }


def discover_artifacts(context: Any = None) -> tuple[dict[str, Any], ...]:
    """Discover ML Lab artifacts in the current PAH results tree.

    Discovery is intentionally conservative: only JSON records carrying a known
    ML Lab schema are registered. Arbitrary project JSON files are ignored.
    """
    project, results = _roots(context)
    if not results.is_dir():
        return ()
    discovered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(results.rglob("*.json")):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        artifact = _artifact_from_json(path, payload, project)
        if artifact is None or artifact["artifact_id"] in seen:
            continue
        seen.add(artifact["artifact_id"])
        discovered.append(artifact)
    return tuple(discovered)


class MLLabRuntimeAdapter:
    """Own ML Lab's local UI lifecycle while PAH owns orchestration/presentation."""

    module_id = MODULE_ID

    def __init__(self) -> None:
        self._server = None
        self._thread: threading.Thread | None = None
        self._url: str | None = None
        self._binding: tuple[str, int, str] | None = None
        self._error: str | None = None

    def _running(self) -> bool:
        return bool(self._server is not None and self._thread is not None and self._thread.is_alive())

    def status(self, *, context: Any = None) -> dict[str, Any]:
        project, results = _roots(context)
        try:
            import flask  # noqa: F401
            available = True
            message = self._error
        except ImportError:
            available = False
            message = "ML Lab UI requires the optional 'ui' extra. Run PAH setup after adding the ML_Lab submodule."
        return {
            "module_id": MODULE_ID,
            "available": available,
            "running": self._running(),
            "launchable": available,
            "url": self._url if self._running() else None,
            "presentation": "module_ui",
            "message": message,
            "metadata": {
                "project_root": str(project) if project else None,
                "results_root": str(results),
                "bound_project": self._binding[2] if self._binding else None,
            },
        }

    def launch(self, *, context: Any = None, detached: bool = False) -> dict[str, Any]:
        from werkzeug.serving import make_server
        from ml_lab.ui import create_app

        project, results = _roots(context)
        host = str(_context_value(context, "host", "127.0.0.1") or "127.0.0.1")
        port = _port(context)
        project_key = str(project) if project else ""
        binding = (host, port, project_key)

        if self._running() and self._binding != binding:
            self.shutdown(context=context)
        if not self._running():
            profile_root = results / "data" / "profile_runs"
            derived_root = results / "data" / "derived"
            app = create_app(config={
                "ML_LAB_PROFILE_OUTPUT_ROOT": str(profile_root),
                "ML_LAB_DERIVED_OUTPUT_ROOT": str(derived_root),
            })
            try:
                server = make_server(host, port, app, threaded=True)
            except OSError as exc:
                self._error = f"Unable to start ML Lab on {host}:{port}: {exc}"
                raise RuntimeError(self._error) from exc
            thread = threading.Thread(target=server.serve_forever, name="ml-lab-pah-ui", daemon=True)
            thread.start()
            self._server = server
            self._thread = thread
            self._binding = binding
            self._url = f"http://{host}:{port}/"
            self._error = None

        return {
            "module_id": MODULE_ID,
            "launched": True,
            "presentation": "module_ui",
            "surface": "ml_lab",
            "url": self._url,
            "metadata": {
                "detached": bool(detached),
                "project_root": str(project) if project else None,
                "results_root": str(results),
            },
        }

    def artifacts(self, *, context: Any = None):
        return discover_artifacts(context)

    def shutdown(self, *, context: Any = None) -> None:
        server = self._server
        self._server = None
        self._url = None
        self._binding = None
        self._error = None
        if server is not None:
            try:
                server.shutdown()
            finally:
                server.server_close()
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)


def runtime_adapter() -> MLLabRuntimeAdapter:
    return MLLabRuntimeAdapter()


__all__ = ["MLLabRuntimeAdapter", "discover_artifacts", "module_manifest", "runtime_adapter"]
