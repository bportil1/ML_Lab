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



def _split_names(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _scalar(raw: str) -> Any:
    text = raw.strip()
    if text.casefold() in {"null", "none", "na", "nan"}:
        return None
    if text.casefold() in {"true", "false"}:
        return text.casefold() == "true"
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def _mapping_lines(raw: str, *, label: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for number, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"{label} line {number} must use column=value")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{label} line {number} has an empty column name")
        result[key] = _scalar(value)
    return result


def _recipe_from_transform_form() -> dict[str, Any]:
    operations: list[dict[str, Any]] = []
    selected = _split_names(request.form.get("select_columns", ""))
    dropped = _split_names(request.form.get("drop_columns", ""))
    renames = _mapping_lines(request.form.get("rename_columns", ""), label="Rename")
    type_overrides = {key: str(value) for key, value in _mapping_lines(request.form.get("type_overrides", ""), label="Type override").items()}
    sentinels = [_scalar(value) for value in _split_names(request.form.get("sentinel_values", ""))]
    sentinel_columns = _split_names(request.form.get("sentinel_columns", ""))
    fills = _mapping_lines(request.form.get("fill_missing", ""), label="Fill missing")
    drop_missing_columns = _split_names(request.form.get("drop_missing_columns", ""))
    clean_columns = _split_names(request.form.get("clean_columns", ""))
    duplicate_columns = _split_names(request.form.get("duplicate_columns", ""))
    encode_columns = _split_names(request.form.get("encode_columns", ""))
    scale_columns = _split_names(request.form.get("scale_columns", ""))
    outlier_columns = _split_names(request.form.get("outlier_columns", ""))

    if selected:
        operations.append({"type": "select_columns", "columns": selected})
    if dropped:
        operations.append({"type": "drop_columns", "columns": dropped})
    if request.form.get("drop_all_missing", "") == "1" or request.form.get("drop_constant", "") == "1":
        operations.append({
            "type": "drop_quality_columns",
            "all_missing": request.form.get("drop_all_missing", "") == "1",
            "constant": request.form.get("drop_constant", "") == "1",
            "protected_columns": _split_names(request.form.get("protected_quality_columns", "")),
        })
    if renames:
        operations.append({"type": "rename_columns", "mapping": {key: str(value) for key, value in renames.items()}})
    if type_overrides:
        operations.append({"type": "coerce_types", "mapping": type_overrides, "errors": request.form.get("coerce_errors", "coerce")})
    if sentinels:
        operation: dict[str, Any] = {"type": "sentinel_to_missing", "values": sentinels}
        if sentinel_columns:
            operation["columns"] = sentinel_columns
        operations.append(operation)
    for column, value in fills.items():
        operations.append({"type": "fill_missing", "columns": [column], "method": "constant", "value": value})
    if drop_missing_columns:
        operations.append({"type": "drop_missing_rows", "columns": drop_missing_columns, "how": request.form.get("drop_missing_how", "any")})
    if clean_columns:
        operations.append({
            "type": "clean_strings",
            "columns": clean_columns,
            "strip": True,
            "collapse_whitespace": request.form.get("collapse_whitespace", "") == "1",
            "case": request.form.get("clean_case", "preserve"),
        })
    if request.form.get("drop_duplicates", "") == "1":
        operation = {"type": "drop_duplicates", "keep": request.form.get("duplicate_keep", "first")}
        if duplicate_columns:
            operation["columns"] = duplicate_columns
        operations.append(operation)

    filter_column = request.form.get("filter_column", "").strip()
    if filter_column:
        operator = request.form.get("filter_operator", "eq")
        operation = {"type": "filter_rows", "column": filter_column, "operator": operator}
        if operator not in {"is_missing", "not_missing"}:
            operation["value"] = _scalar(request.form.get("filter_value", ""))
        operations.append(operation)

    derive_source = request.form.get("derive_source", "").strip()
    derive_target = request.form.get("derive_target", "").strip()
    derive_pattern = request.form.get("derive_pattern", "").strip()
    if derive_source or derive_target or derive_pattern:
        if not (derive_source and derive_target and derive_pattern):
            raise ValueError("Regex derivation requires source column, target column, and pattern.")
        operations.append({
            "type": "derive",
            "method": "regex_extract",
            "source": derive_source,
            "target": derive_target,
            "pattern": derive_pattern,
            "group": _form_int("derive_group", 0),
        })
    if encode_columns:
        operations.append({"type": "encode_categorical", "columns": encode_columns, "method": "one_hot", "drop_first": request.form.get("encode_drop_first", "") == "1"})
    if scale_columns:
        operations.append({"type": "scale", "columns": scale_columns, "method": request.form.get("scale_method", "standard")})
    if outlier_columns:
        operations.append({
            "type": "outliers",
            "columns": outlier_columns,
            "method": request.form.get("outlier_method", "clip_iqr"),
            "iqr_multiplier": float(request.form.get("outlier_iqr_multiplier", "1.5")),
        })

    return {
        "schema": "ml-lab.transformation-recipe@1",
        "name": request.form.get("recipe_name", "Data Lab transformation").strip() or "Data Lab transformation",
        "description": request.form.get("recipe_description", "").strip(),
        "allow_malformed_rows": request.form.get("allow_malformed_rows", "") == "1",
        "operations": operations,
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

    def transform_url(path: str) -> str:
        return url_for(request.blueprint + ".data_transform", source_token=sign_path(path))

    def provenance_url(path: str) -> str:
        return url_for(request.blueprint + ".data_provenance", source_token=sign_path(path))

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
            transform_url=transform_url,
            provenance_url=provenance_url,
            recent_runs=data.recent_profile_runs(output_root=profile_output_root, limit=12),
            experimental_enabled=enable_experimental,
        )


    @blueprint.route("/data/compare", methods=["GET", "POST"])
    def data_compare():
        paths_text = request.form.get("paths", "")
        recursive = request.form.get("recursive", "1") == "1"
        include_hidden = request.form.get("include_hidden", "") == "1"
        max_pairs = _form_int("max_pairs", 200)
        result = None
        error = None
        if request.method == "POST":
            paths = [line.strip() for line in paths_text.splitlines() if line.strip()]
            if not paths:
                error = "Enter at least one file or directory path."
            else:
                try:
                    result = execute_task(
                        "data.compare",
                        {
                            "paths": paths,
                            "recursive": recursive,
                            "include_hidden": include_hidden,
                            "max_pairs": max_pairs,
                        },
                    )["result"]
                except (PayloadError, OSError, UnicodeError, TypeError, ValueError, KeyError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
        return render_template(
            "ml_lab_ui/compare.html",
            version=__version__,
            paths_text=paths_text,
            recursive=recursive,
            include_hidden=include_hidden,
            max_pairs=max_pairs,
            result=result,
            result_json=(_pretty(result) if result is not None else None),
            error=error,
            data_url=url_for(request.blueprint + ".data_lab"),
        )

    @blueprint.route("/data/provenance", methods=["GET", "POST"])
    def data_provenance():
        source = request.form.get("source", "").strip()
        source_token = request.args.get("source_token", "")
        if request.method == "GET" and source_token:
            source = str(verify_path(source_token))
        lineage = None
        error = None
        if request.method == "POST" or source:
            try:
                if not source:
                    raise ValueError("Choose a CSV/TSV dataset path.")
                lineage = execute_task("data.lineage", {"path": source})["result"]
            except (PayloadError, OSError, UnicodeError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                error = f"{type(exc).__name__}: {exc}"
        return render_template(
            "ml_lab_ui/provenance.html",
            version=__version__,
            source=source,
            lineage=lineage,
            lineage_json=(_pretty(lineage) if lineage is not None else None),
            error=error,
            data_url=url_for(request.blueprint + ".data_lab"),
        )

    @blueprint.route("/data/transform", methods=["GET", "POST"])
    def data_transform():
        source = request.form.get("source", "").strip()
        source_token = request.args.get("source_token", "")
        if request.method == "GET" and source_token:
            source = str(verify_path(source_token))
        recipe = None
        result = None
        error = None
        applied = False
        if request.method == "POST":
            try:
                if not source:
                    raise ValueError("Choose a CSV/TSV source path.")
                recipe = _recipe_from_transform_form()
                mode = request.form.get("mode", "preview")
                payload = {
                    "path": source,
                    "recipe": recipe,
                    "preview_rows": _form_int("preview_rows", 50),
                }
                if mode == "apply":
                    payload["output"] = request.form.get("output", "").strip() or None
                    payload["overwrite"] = request.form.get("overwrite", "") == "1"
                    result = execute_task("data.transform.apply", payload)["result"]
                    applied = True
                else:
                    result = execute_task("data.transform.preview", payload)["result"]
            except (PayloadError, OSError, UnicodeError, ValueError, TypeError, KeyError, FileExistsError, json.JSONDecodeError) as exc:
                error = f"{type(exc).__name__}: {exc}"
        return render_template(
            "ml_lab_ui/transform.html",
            version=__version__,
            source=source,
            recipe=recipe,
            recipe_json=(_pretty(recipe) if recipe is not None else None),
            result=result,
            result_json=(_pretty(result) if result is not None else None),
            error=error,
            applied=applied,
            provenance_url=(provenance_url(result["derived"]["path"]) if applied and result else None),
            data_url=url_for(request.blueprint + ".data_lab"),
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
            transform_url=url_for(request.blueprint + ".data_transform", source_token=token),
            provenance_url=url_for(request.blueprint + ".data_provenance", source_token=token),
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
