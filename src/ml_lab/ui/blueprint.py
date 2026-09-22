from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
from pathlib import Path
from typing import Any

try:
    from flask import Blueprint, abort, jsonify, redirect, render_template, request, send_file, url_for
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("ML Lab UI requires: pip install 'ml-lab[ui]'") from exc

from ml_lab import __version__, data
from ml_lab.application import PayloadError, execute_task
from ml_lab.core.serialization import to_jsonable

from .jobs import ProfileJobManager
from .service import capability_groups, capability_records, get_capability


def _pretty(value: Any) -> str:
    return json.dumps(to_jsonable(value), indent=2, sort_keys=True)


def _form_int(name: str, default: int) -> int:
    try:
        return int(request.form.get(name, str(default)))
    except ValueError:
        return default


def _profile_payload_from_form() -> dict[str, Any]:
    paths = [line.strip() for line in request.form.get("paths", "").splitlines() if line.strip()]
    if not paths:
        raise ValueError("Enter at least one file or directory path.")
    return {
        "paths": paths,
        "recursive": request.form.get("recursive", "") == "1",
        "include_hidden": request.form.get("include_hidden", "") == "1",
        "preview_rows": _form_int("preview_rows", 20),
        "max_rows": _form_int("max_rows", 100_000),
        "relationship_rows": _form_int("relationship_rows", 5_000),
        "max_relationship_columns": _form_int("max_relationship_columns", 25),
        "max_relationship_pairs": _form_int("max_relationship_pairs", 200),
        "outlier_iqr_multiplier": float(request.form.get("outlier_iqr_multiplier", "1.5")),
        "random_state": _form_int("random_state", 42),
    }


def create_ui_blueprint(
    name: str = "ml_lab_ui",
    *,
    enable_experimental: bool = False,
    profile_output_root: str | Path = "ml_lab_results/data/profile_runs",
) -> Blueprint:
    """Create ML Lab's mountable first-party UI Blueprint.

    The host controls URL placement and surrounding authentication/chrome. ML Lab
    owns the workspace and execution semantics. Profile jobs are in-process and
    persisted locally; no external worker or service is required.
    """
    blueprint = Blueprint(
        name,
        __name__,
        template_folder="templates",
        static_folder="static",
        static_url_path="/static",
    )
    jobs = ProfileJobManager(output_root=profile_output_root)
    token_secret = secrets.token_bytes(32)

    def sign_path(path: str) -> str:
        payload = base64.urlsafe_b64encode(str(Path(path).expanduser().resolve()).encode("utf-8")).decode("ascii").rstrip("=")
        signature = hmac.new(token_secret, payload.encode("ascii"), hashlib.sha256).hexdigest()
        return f"{payload}.{signature}"

    def verify_path(token: str) -> Path:
        try:
            payload, signature = token.rsplit(".", 1)
        except ValueError:
            abort(404)
        expected = hmac.new(token_secret, payload.encode("ascii"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            abort(404)
        try:
            padded = payload + "=" * (-len(payload) % 4)
            decoded = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            abort(404)
        path = Path(decoded)
        if not path.is_file():
            abort(404)
        return path

    def source_url(path: str) -> str:
        return url_for(request.blueprint + ".source_view", token=sign_path(path))

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
        preview_rows = _form_int("preview_rows", 20)
        max_rows = _form_int("max_rows", 100_000)
        relationship_rows = _form_int("relationship_rows", 5_000)
        max_relationship_columns = _form_int("max_relationship_columns", 25)
        inventory = None
        inventory_json = None
        error = None

        if request.method == "POST":
            paths = [line.strip() for line in paths_text.splitlines() if line.strip()]
            if not paths:
                error = "Enter at least one file or directory path."
            else:
                try:
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
            error=error,
            source_url=source_url,
            recent_runs=data.recent_profile_runs(output_root=profile_output_root, limit=12),
            experimental_enabled=enable_experimental,
        )

    @blueprint.post("/data/profile/start")
    def start_profile_job():
        try:
            payload = _profile_payload_from_form()
            job_id = jobs.start(payload)
        except (TypeError, ValueError) as exc:
            return render_template(
                "ml_lab_ui/profile_job.html",
                version=__version__,
                job={"status": "failed", "stage": "Could not start profile", "error": str(exc), "job_id": None},
                status_url=None,
                data_url=url_for(request.blueprint + ".data_lab"),
            ), 400
        return redirect(url_for(request.blueprint + ".profile_job", job_id=job_id))

    @blueprint.get("/data/profile/job/<job_id>")
    def profile_job(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            try:
                manifest = data.load_profile_run(output_root=profile_output_root, run_id=job_id)
            except FileNotFoundError:
                abort(404)
            job = {
                "job_id": job_id,
                "status": "completed",
                "stage": "Profile complete",
                "current": len(manifest.get("profiles", [])),
                "total": len(manifest.get("profiles", [])),
                "elapsed_seconds": None,
                "artifact_manifest": manifest,
                "warnings": [],
                "error": None,
            }
        return render_template(
            "ml_lab_ui/profile_job.html",
            version=__version__,
            job=job,
            status_url=url_for(request.blueprint + ".profile_job_status", job_id=job_id),
            data_url=url_for(request.blueprint + ".data_lab"),
        )

    @blueprint.get("/data/profile/job/<job_id>/status")
    def profile_job_status(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            try:
                manifest = data.load_profile_run(output_root=profile_output_root, run_id=job_id)
            except FileNotFoundError:
                return jsonify({"error": "profile job not found"}), 404
            job = {
                "job_id": job_id,
                "status": "completed",
                "stage": "Profile complete",
                "current": len(manifest.get("profiles", [])),
                "total": len(manifest.get("profiles", [])),
                "elapsed_seconds": None,
                "artifact_manifest": manifest,
                "warnings": [],
                "error": None,
            }
        manifest = job.get("artifact_manifest") or {}
        profile_links = [
            {
                "index": ref.get("index", index),
                "label": ref.get("relative_path", f"Profile {index + 1}"),
                "url": url_for(request.blueprint + ".profile_view", run_id=job_id, profile_index=index),
            }
            for index, ref in enumerate(manifest.get("profiles", []))
        ]
        return jsonify({**job, "profile_links": profile_links})

    @blueprint.get("/data/profile/<run_id>/<int:profile_index>")
    def profile_view(run_id: str, profile_index: int):
        try:
            manifest = data.load_profile_run(output_root=profile_output_root, run_id=run_id)
            profile = data.load_persisted_profile(manifest, profile_index)
        except (FileNotFoundError, IndexError, KeyError, json.JSONDecodeError):
            abort(404)
        source = profile.get("path")
        return render_template(
            "ml_lab_ui/profile.html",
            version=__version__,
            profile=profile,
            manifest=manifest,
            profile_index=profile_index,
            profile_json=_pretty(profile),
            source_url=(source_url(source) if source and Path(source).is_file() else None),
            raw_profile_url=url_for(request.blueprint + ".profile_json", run_id=run_id, profile_index=profile_index),
            data_url=url_for(request.blueprint + ".data_lab"),
        )

    @blueprint.get("/data/profile/<run_id>/<int:profile_index>/json")
    def profile_json(run_id: str, profile_index: int):
        try:
            manifest = data.load_profile_run(output_root=profile_output_root, run_id=run_id)
            ref = manifest["profiles"][profile_index]
            path = Path(ref["profile_path"])
        except (FileNotFoundError, IndexError, KeyError):
            abort(404)
        return send_file(path, mimetype="application/json", as_attachment=False)

    @blueprint.get("/data/source/<token>")
    def source_view(token: str):
        path = verify_path(token)
        try:
            record = data.inspect_file(path)
        except (OSError, ValueError):
            abort(404)
        if not record.supported or record.parse_status == "failed":
            abort(400)
        return render_template(
            "ml_lab_ui/source.html",
            version=__version__,
            source=record.to_record(),
            rows_url=url_for(request.blueprint + ".source_rows", token=token),
            raw_url=url_for(request.blueprint + ".source_raw", token=token),
            data_url=url_for(request.blueprint + ".data_lab"),
        )

    @blueprint.get("/data/source/<token>/rows")
    def source_rows(token: str):
        path = verify_path(token)
        raw_size = request.args.get("page_size", "50")
        page_size: int | None
        if raw_size.lower() == "all":
            page_size = None
        else:
            try:
                page_size = int(raw_size)
            except ValueError:
                return jsonify({"error": "invalid page_size"}), 400
        try:
            filters = json.loads(request.args.get("filters", "{}"))
            if not isinstance(filters, dict):
                raise ValueError("filters must be an object")
            result = execute_task(
                "data.table",
                {
                    "path": str(path),
                    "page": int(request.args.get("page", "1")),
                    "page_size": "all" if page_size is None else page_size,
                    "search": request.args.get("search", ""),
                    "filters": filters,
                    "sort_column": request.args.get("sort") or None,
                    "sort_direction": request.args.get("direction", "asc"),
                },
            )
        except (PayloadError, OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400
        return jsonify(result["result"])

    @blueprint.get("/data/source/<token>/raw")
    def source_raw(token: str):
        path = verify_path(token)
        if path.suffix.lower() not in {".csv", ".tsv"}:
            abort(400)
        return send_file(path, mimetype="text/plain", as_attachment=False, download_name=path.name)

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
