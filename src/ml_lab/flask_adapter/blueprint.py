from __future__ import annotations

from typing import Any

try:
    from flask import Blueprint, jsonify, request
except ImportError as exc:  # pragma: no cover - exercised by the lazy wrapper when Flask is absent
    raise RuntimeError("Flask support requires: pip install 'ml-lab[flask]'") from exc

from ml_lab import __version__
from ml_lab.core.serialization import to_jsonable
from ml_lab.experimental import get_manifest, list_experiments, run_experiment
from ml_lab.registry import list_estimators

from .service import PayloadError, execute_task


def create_blueprint(name: str = "ml_lab", *, enable_experimental: bool = False) -> Blueprint:
    """Create a mountable ML Lab Blueprint.

    The caller owns URL placement, for example::

        app.register_blueprint(create_blueprint(), url_prefix="/tools/ml-lab")

    This makes the adapter safe to mount directly in PAH or several levels down
    inside another service without ML Lab knowing the host's route structure.
    """
    blueprint = Blueprint(name, __name__)

    @blueprint.get("/health")
    def health():
        return jsonify({"status": "ok", "module": "ml_lab", "version": __version__})

    @blueprint.get("/estimators")
    def estimators():
        task = request.args.get("task", "all")
        try:
            records = [
                {
                    "id": spec.id,
                    "name": spec.name,
                    "task": spec.task,
                    "param_grid": to_jsonable(spec.param_grid),
                    "preprocess": spec.preprocess,
                    "notes": spec.notes,
                }
                for spec in list_estimators(task)
            ]
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"estimators": records})

    @blueprint.post("/run/<task>")
    def run_task(task: str):
        try:
            payload = request.get_json(silent=False)
            return jsonify(to_jsonable(execute_task(task, payload)))
        except (PayloadError, TypeError, ValueError, KeyError) as exc:
            return jsonify({"error": str(exc), "type": type(exc).__name__}), 400

    if enable_experimental:
        @blueprint.get("/experimental")
        def experimental_list():
            return jsonify({"experiments": [manifest.to_record() for manifest in list_experiments()]})

        @blueprint.get("/experimental/<experiment_id>")
        def experimental_info(experiment_id: str):
            try:
                return jsonify(get_manifest(experiment_id).to_record())
            except KeyError as exc:
                return jsonify({"error": str(exc)}), 404

        @blueprint.post("/experimental/<experiment_id>/run")
        def experimental_run(experiment_id: str):
            try:
                payload: dict[str, Any] = request.get_json(silent=False) or {}
                return jsonify({"experiment": experiment_id, "result": to_jsonable(run_experiment(experiment_id, **payload))})
            except KeyError as exc:
                return jsonify({"error": str(exc)}), 404
            except (TypeError, ValueError, AttributeError) as exc:
                return jsonify({"error": str(exc), "type": type(exc).__name__}), 400

    return blueprint
