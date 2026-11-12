"""Instance-scoped workbench routes, kept separate from the large UI blueprint.

Only opaque instance identity is stored in the browser; dataset selections stay
in this project's ExperimentStore SQLite database.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from flask import Blueprint, g, jsonify, request, url_for


def install_workbench_routes(blueprint: Blueprint, *, name: str, project_root: Path):
    """Install workbench HTTP endpoints and return a context reader for pages."""
    cookie_name = "ml_lab_workbench_" + hashlib.sha256(
        f"{name}:{project_root}".encode("utf-8")
    ).hexdigest()[:20]

    def workbench_context(store=None):
        from ml_lab.experiments import (
            ExperimentStore, WorkbenchInstanceNotFound,
            create_workbench_instance, get_workbench_instance,
        )

        store = store if store is not None else ExperimentStore(project_root)
        identifier = request.cookies.get(cookie_name, "")
        try:
            context = get_workbench_instance(store, identifier)
        except WorkbenchInstanceNotFound:
            context = create_workbench_instance(store)
        g.ml_lab_workbench_instance_id = context["instance_id"]
        return context

    @blueprint.after_request
    def workbench_cookie(response):
        identifier = getattr(g, "ml_lab_workbench_instance_id", None)
        if identifier is not None:
            response.set_cookie(
                cookie_name, identifier, httponly=True, samesite="Lax",
                path=url_for(request.blueprint + ".index"),
            )
        return response

    @blueprint.get("/api/workbench/context")
    def workbench_context_api():
        """Get the active pinned dataset, revalidating its source before use."""
        return jsonify(workbench_context())

    @blueprint.post("/api/workbench/context/activate")
    def workbench_activate_api():
        from ml_lab.experiments import (
            ExperimentStore, ExperimentStoreError, WorkbenchConflict,
            activate_workbench_experiment,
        )

        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Expected a JSON object"}), 400
        try:
            context = workbench_context()
            updated = activate_workbench_experiment(
                ExperimentStore(project_root), context["instance_id"],
                payload.get("experiment_id"),
                expected_generation=payload.get("expected_generation"),
                revision=payload.get("revision"),
            )
        except WorkbenchConflict as exc:
            return jsonify({"error": str(exc)}), 409
        except ExperimentStoreError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(updated)

    @blueprint.post("/api/workbench/context/clear")
    def workbench_clear_api():
        from ml_lab.experiments import (
            ExperimentStore, ExperimentStoreError, WorkbenchConflict,
            clear_workbench_dataset,
        )

        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Expected a JSON object"}), 400
        try:
            context = workbench_context()
            updated = clear_workbench_dataset(
                ExperimentStore(project_root), context["instance_id"],
                expected_generation=payload.get("expected_generation"),
            )
        except WorkbenchConflict as exc:
            return jsonify({"error": str(exc)}), 409
        except ExperimentStoreError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(updated)

    return workbench_context
