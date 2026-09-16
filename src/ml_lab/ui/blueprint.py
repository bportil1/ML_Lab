from __future__ import annotations

import json
from typing import Any

try:
    from flask import Blueprint, jsonify, render_template, request
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("ML Lab UI requires: pip install 'ml-lab[ui]'") from exc

from ml_lab import __version__
from ml_lab.application import PayloadError, execute_task
from ml_lab.core.serialization import to_jsonable

from .service import capability_groups, capability_records, get_capability


def _pretty(value: Any) -> str:
    return json.dumps(to_jsonable(value), indent=2, sort_keys=True)


def create_ui_blueprint(name: str = "ml_lab_ui", *, enable_experimental: bool = False) -> Blueprint:
    """Create ML Lab's mountable first-party UI Blueprint.

    The host controls the URL prefix and surrounding authentication/chrome. ML Lab
    owns the actual workspace, templates, and execution semantics.
    """
    blueprint = Blueprint(
        name,
        __name__,
        template_folder="templates",
        static_folder="static",
        static_url_path="/static",
    )

    @blueprint.get("/")
    def index():
        return render_template(
            "ml_lab_ui/index.html",
            version=__version__,
            groups=capability_groups(enable_experimental=enable_experimental),
            experimental_enabled=enable_experimental,
        )

    @blueprint.route("/data", methods=["GET", "POST"])
    def data_lab():
        paths_text = request.form.get("paths", "")
        recursive = request.form.get("recursive", "1") == "1"
        include_hidden = request.form.get("include_hidden", "") == "1"
        action = request.form.get("action", "inspect")
        try:
            preview_rows = int(request.form.get("preview_rows", "20"))
        except ValueError:
            preview_rows = 20
        try:
            max_rows = int(request.form.get("max_rows", "100000"))
        except ValueError:
            max_rows = 100000
        try:
            relationship_rows = int(request.form.get("relationship_rows", "5000"))
        except ValueError:
            relationship_rows = 5000
        try:
            max_relationship_columns = int(request.form.get("max_relationship_columns", "25"))
        except ValueError:
            max_relationship_columns = 25

        inventory = None
        inventory_json = None
        profile_collection = None
        profile_json = None
        error = None
        if request.method == "POST":
            paths = [line.strip() for line in paths_text.splitlines() if line.strip()]
            if not paths:
                error = "Enter at least one file or directory path."
            else:
                try:
                    if action == "profile":
                        result = execute_task(
                            "data.profile",
                            {
                                "paths": paths,
                                "recursive": recursive,
                                "include_hidden": include_hidden,
                                "preview_rows": preview_rows,
                                "max_rows": max_rows,
                                "relationship_rows": relationship_rows,
                                "max_relationship_columns": max_relationship_columns,
                            },
                        )
                        profile_collection = result["result"]
                        profile_json = _pretty(profile_collection)
                        inventory = profile_collection.get("inventory")
                        inventory_json = _pretty(inventory) if inventory else None
                    else:
                        result = execute_task(
                            "data.inspect",
                            {
                                "paths": paths,
                                "recursive": recursive,
                                "include_hidden": include_hidden,
                                "preview_rows": preview_rows,
                            },
                        )
                        inventory = result["result"]
                        inventory_json = _pretty(inventory)
                except (PayloadError, OSError, TypeError, ValueError, KeyError) as exc:
                    error = f"{type(exc).__name__}: {exc}"

        return render_template(
            "ml_lab_ui/data.html",
            version=__version__,
            paths_text=paths_text,
            recursive=recursive,
            include_hidden=include_hidden,
            preview_rows=preview_rows,
            max_rows=max_rows,
            relationship_rows=relationship_rows,
            max_relationship_columns=max_relationship_columns,
            inventory=inventory,
            inventory_json=inventory_json,
            profile_collection=profile_collection,
            profile_json=profile_json,
            error=error,
            experimental_enabled=enable_experimental,
        )

    @blueprint.route("/task/<capability_id>", methods=["GET", "POST"])
    def task(capability_id: str):
        try:
            capability = get_capability(capability_id, enable_experimental=enable_experimental)
        except KeyError:
            return render_template("ml_lab_ui/not_found.html", capability_id=capability_id, version=__version__), 404

        task_id = capability.get("task")
        example = capability.get("example", {})
        payload_text = request.form.get("payload", _pretty(example))
        result_text: str | None = None
        error: str | None = None

        if request.method == "POST":
            if not task_id:
                error = "This capability does not yet expose a generic UI execution contract."
            elif not capability["available"]:
                missing = ", ".join(capability["missing_dependencies"])
                error = f"Missing optional dependency: {missing}"
            else:
                try:
                    payload = json.loads(payload_text)
                    result_text = _pretty(execute_task(task_id, payload))
                except json.JSONDecodeError as exc:
                    error = f"Invalid JSON payload: {exc}"
                except (PayloadError, TypeError, ValueError, KeyError) as exc:
                    error = f"{type(exc).__name__}: {exc}"

        return render_template(
            "ml_lab_ui/task.html",
            version=__version__,
            capability=capability,
            payload_text=payload_text,
            result_text=result_text,
            error=error,
            experimental_enabled=enable_experimental,
        )

    @blueprint.get("/api/capabilities")
    def capabilities_api():
        return jsonify({"version": __version__, "capabilities": capability_records(enable_experimental=enable_experimental)})

    return blueprint
