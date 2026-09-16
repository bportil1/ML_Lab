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
