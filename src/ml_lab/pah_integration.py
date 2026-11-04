"""Thin PAH discovery/runtime adapter for ML Lab.

This module intentionally depends only on ML Lab and the Python standard library.
It returns mapping-shaped PAH contracts so ML Lab remains independently usable
and does not take a dependency on the PAH host package.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
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
            "experiment_tracking",
            "model_execution",
            "synthetic_data_generation",
            "run_comparison",
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
    requested_results = Path(results_raw).expanduser().resolve() if results_raw else None

    if project is not None:
        legacy_default = (project / "ml_lab_results").resolve()
        # PAH historically supplied <project>/ml_lab_results. Keep truly custom
        # host roots, but move the old/default layout under the selected
        # project's top-level analysis directory.
        if requested_results is None or requested_results == legacy_default:
            results = (project / "analysis").resolve()
        else:
            results = requested_results
    elif requested_results is not None:
        results = requested_results
    else:
        results = Path("analysis").resolve()
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


def _registered_experiment_artifacts(project: Path) -> list[dict[str, Any]]:
    """Expose only completed, verified ML_Lab outputs via the module's own API.

    Existing projects without an experiment database are left untouched. The
    published catalogue is bounded to avoid unbounded host status requests.
    """
    database = project / "analysis" / "ml_lab" / "experiments.sqlite3"
    if not database.is_file() or database.is_symlink():
        return []
    from ml_lab.experiments import ExperimentStore, ExperimentStoreError

    catalog: list[dict[str, Any]] = []
    try:
        store = ExperimentStore(project)
    except (ExperimentStoreError, sqlite3.DatabaseError):
        # Discovery must not make a future/unknown local schema unlaunchable.
        return []
    executors = {
        "ml_lab.experiments.training@1": ("training", "model_execution"),
        "ml_lab.experiments.optimization@1": ("optimization", "model_optimization"),
        "ml_lab.experiments.synthetic@1": ("synthetic", "synthetic_data_generation"),
    }
    # Bound the amount of history offered through a synchronous host discovery.
    for experiment in store.list_experiments(limit=500):
        for run in store.list_runs(experiment["experiment_id"], limit=100):
            if run["status"] != "succeeded":
                continue
            executor = run["environment"].get("executor")
            if executor not in executors:
                continue
            family, capability = executors[executor]
            for artifact in store.artifacts.list_for_run(run["run_id"], limit=100):
                if artifact["role"] != "output" or artifact["kind"] not in {"result", "dataset", "history"}:
                    continue
                try:
                    if store.artifacts.verify(artifact["artifact_id"])["status"] != "valid":
                        continue
                except (ExperimentStoreError, OSError):
                    continue
                location = (project / artifact["path"]).resolve()
                catalog.append({
                    "artifact_id": f"ml-lab-experiment-{artifact['artifact_id']}",
                    "kind": ("synthetic_dataset" if family == "synthetic" and artifact["kind"] == "dataset"
                             else f"{family}_{artifact['kind']}"),
                    "producer_module": MODULE_ID,
                    "location": str(location),
                    "schema_id": "ml-lab.registered-experiment-output",
                    "schema_version": "1",
                    "media_type": artifact.get("media_type") or ("text/csv" if location.suffix == ".csv"
                                                               else "application/json" if location.suffix == ".json"
                                                               else "application/octet-stream"),
                    "producer_version": __version__,
                    "project_id": str(project),
                    "capabilities": ["experiment_tracking", capability],
                    "validation_state": "valid",
                    "metadata": {"runtime_managed": True, "registry_alias": False,
                                 "run_id": run["run_id"], "experiment_id": run["experiment_id"],
                                 "revision": run["revision"], "sha256": artifact["sha256"],
                                 "artifact_kind": artifact["kind"]},
                    "provenance": {"run_id": run["run_id"], "revision": run["revision"]},
                })
                # Avoid hashing/discovering an unbounded archive on a host
                # status call. Older history remains in the local store.
                if len(catalog) >= 250:
                    return catalog
    return catalog


def discover_artifacts(context: Any = None) -> tuple[dict[str, Any], ...]:
    """Discover ML Lab artifacts in the current PAH results tree.

    Discover known Data Lab sidecars and completed, hash-verified experiment
    outputs. Arbitrary project JSON files are ignored.
    """
    project, results = _roots(context)
    if not results.is_dir():
        return tuple(_registered_experiment_artifacts(project)) if project is not None else ()
    discovered: list[dict[str, Any]] = []
    seen: set[str] = set()
    allowed_roots = [results.resolve()] + ([project] if project is not None else [])
    for path in sorted(results.rglob("*.json")):
        if not path.is_file() or path.is_symlink():
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
        # A manifest is not permission to expose arbitrary local files. Only
        # advertise regular files within the configured results/project roots.
        location = Path(artifact["location"])
        if location.is_symlink() or not location.is_file():
            continue
        resolved = location.resolve()
        if not any(resolved.is_relative_to(root) for root in allowed_roots):
            continue
        seen.add(artifact["artifact_id"])
        discovered.append(artifact)
    if project is not None:
        discovered.extend(_registered_experiment_artifacts(project))
    return tuple(discovered)


class MLLabRuntimeAdapter:
    """Own ML Lab's local UI lifecycle while PAH owns orchestration/presentation."""

    module_id = MODULE_ID

    def __init__(self) -> None:
        self._server = None
        self._thread: threading.Thread | None = None
        self._url: str | None = None
        self._binding: tuple[str, int, str, str] | None = None
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
        host = str(_context_value(context, "host", "127.0.0.1") or "127.0.0.1")
        binding = (host, _port(context), str(project) if project else "", str(results))
        running_for_context = self._running() and self._binding == binding
        if self._running() and not running_for_context:
            message = "ML Lab is running for another project or output root; relaunch to switch."
        return {
            "module_id": MODULE_ID,
            "available": available,
            "running": running_for_context,
            "launchable": available,
            "url": self._url if running_for_context else None,
            "presentation": "module_ui",
            "message": message,
            "metadata": {
                "project_root": str(project) if project else None,
                "results_root": str(results),
                "bound_project": self._binding[2] if self._binding else None,
                "bound_results_root": self._binding[3] if self._binding else None,
            },
        }

    def launch(self, *, context: Any = None, detached: bool = False) -> dict[str, Any]:
        from werkzeug.serving import make_server
        from ml_lab.ui import create_app

        project, results = _roots(context)
        host = str(_context_value(context, "host", "127.0.0.1") or "127.0.0.1")
        port = _port(context)
        project_key = str(project) if project else ""
        binding = (host, port, project_key, str(results))

        if self._running() and self._binding != binding:
            self.shutdown(context=context)
        if not self._running():
            profile_root = results / "data" / "profile_runs"
            derived_root = results / "data" / "derived"
            app = create_app(config={
                "ML_LAB_PROFILE_OUTPUT_ROOT": str(profile_root),
                "ML_LAB_DERIVED_OUTPUT_ROOT": str(derived_root),
                # A hosted experiment must never silently bind to PAH's cwd.
                "ML_LAB_EXPERIMENT_PROJECT_ROOT": str(project) if project else None,
                # Suggestions in a hosted UI should start inside the selected
                # project, not PAH's cwd or the host account's home folder.
                "ML_LAB_PATH_AUTOCOMPLETE_ROOTS": [str(project)] if project else None,
            })
            # Hosted requests stay scoped to their selected PAH workspace;
            # standalone ML Lab remains independent of the monitoring package.
            if project is not None:
                try:
                    from pah_jobs.exchange import monitor_directory
                except ImportError:
                    pass
                else:
                    original_wsgi = app.wsgi_app
                    monitor_path = project / ".pah" / "job_monitor"

                    def monitored_wsgi(environ, start_response):
                        with monitor_directory(monitor_path):
                            return original_wsgi(environ, start_response)

                    app.wsgi_app = monitored_wsgi
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
