"""Instance-scoped workbench routes, kept separate from the large UI blueprint.

Only opaque instance identity is stored in the browser; dataset selections stay
in this project's ExperimentStore SQLite database.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from flask import Blueprint, g, jsonify, request, render_template, url_for


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
            activate_workbench_experiment, get_workbench_instance,
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

    @blueprint.route("/workbench/dataset/select-file", methods=["GET", "POST"])
    def workbench_select_file():
        """Explicitly pin a verified project-local CSV/TSV to this instance.

        The read-only GET does not change state. POST checks both the file
        fingerprint and optimistic instance generation before creating any
        experiment revision; previous selections remain archived.
        """
        from ml_lab import __version__
        from ml_lab.experiments import (
            DatasetSelectionError, ExperimentStore, ExperimentStoreError,
            WorkbenchConflict, describe_selection, select_workbench_dataset,
        )

        values = request.form if request.method == "POST" else request.args
        path = values.get("path", "").strip()
        context = workbench_context()
        preview = None
        updated = None
        error = None
        status = 200
        if not path:
            error = "Select an existing project-local CSV or TSV."
            status = 400
        else:
            try:
                store = ExperimentStore(project_root)
                preview = describe_selection(store, path)
                if request.method == "POST":
                    raw_generation = request.form.get("expected_generation", "")
                    if not raw_generation.isdecimal():
                        raise DatasetSelectionError("Reload the form to confirm the current instance generation")
                    expected = int(raw_generation)
                    if (request.form.get("source_sha256", "") != preview["sha256"]
                            or request.form.get("source_dataset_id", "") != preview["dataset_id"]):
                        raise DatasetSelectionError("Dataset changed since the form was opened; reload it")
                    features = request.form.getlist("features")
                    targets = request.form.getlist("targets")
                    updated = select_workbench_dataset(
                        store, context["instance_id"], path,
                        expected_generation=expected,
                        feature_columns=features, target_columns=targets,
                        experiment_name=(request.form.get("experiment_name", "").strip() or None),
                    )
                    context = updated
            except WorkbenchConflict as exc:
                error, status = str(exc), 409
            except (ExperimentStoreError, OSError, ValueError, TypeError) as exc:
                error, status = str(exc), 400
        return render_template(
            "ml_lab_ui/workbench_dataset_select.html", version=__version__,
            source_path=path, context=context, preview=preview,
            updated=updated, error=error,
        ), status

    @blueprint.route("/workbench/experiment/activate", methods=["GET", "POST"])
    def workbench_activate_experiment():
        """Explicit, verified handoff of a saved experiment to the five-tab UI.

        A GET previews a revision and never switches datasets. A POST pins the
        *existing* revision in this browser instance after provenance checks
        and an optimistic generation comparison. Old experiments/runs stay put.
        """
        from ml_lab import __version__
        from ml_lab.experiments import (
            ExperimentStore, ExperimentStoreError, WorkbenchConflict,
            activate_workbench_experiment, get_workbench_instance,
        )
        from ml_lab.experiments.datasets import verify_selection

        values = request.form if request.method == "POST" else request.args
        experiment_id = values.get("experiment_id", "").strip()
        raw_revision = values.get("revision", "")
        context = workbench_context()
        experiment, verified, error, activated = None, False, None, False
        status = 200
        try:
            if not experiment_id:
                raise ValueError("Choose a saved experiment from General ML Tools")
            if raw_revision and (not raw_revision.isdecimal() or int(raw_revision) < 1):
                raise ValueError("Revision must be a positive integer")
            if request.method == "POST" and not raw_revision:
                raise ValueError("An explicit experiment revision is required")
            store = ExperimentStore(project_root)
            experiment = store.get_experiment(
                experiment_id, revision=int(raw_revision) if raw_revision else None,
            )
            # Even the preview does not represent stale paths as available.
            verify_selection(store, experiment["dataset_ref"])
            verified = True
            if request.method == "POST":
                generation = values.get("expected_generation", "")
                if not generation.isdecimal():
                    raise WorkbenchConflict("Workbench changed; reload before activating a saved experiment")
                if int(generation) != context["generation"]:
                    raise WorkbenchConflict("Workbench context changed; reload before activating")
                if (context["experiment_id"] == experiment_id
                        and context["experiment_revision"] == experiment["revision"]):
                    # Idempotent confirmation is a read, not a generation
                    # increment. Recheck the DB in case another tab switched.
                    latest = get_workbench_instance(store, context["instance_id"])
                    if (latest["generation"] != context["generation"]
                            or latest["status"] != "ready"
                            or latest["experiment_id"] != experiment_id
                            or latest["experiment_revision"] != experiment["revision"]):
                        raise WorkbenchConflict("Workbench context changed; reload before activating")
                    context = latest
                else:
                    context = activate_workbench_experiment(
                        store, context["instance_id"], experiment_id,
                        revision=experiment["revision"], expected_generation=int(generation),
                    )
                activated = True
        except WorkbenchConflict as exc:
            error, status = str(exc), 409
        except (ExperimentStoreError, OSError, ValueError, TypeError) as exc:
            error, status = str(exc), 400
        return render_template(
            "ml_lab_ui/workbench_experiment_activate.html",
            version=__version__, context=context, experiment=experiment,
            verified=verified, error=error, activated=activated,
        ), status

    return workbench_context
