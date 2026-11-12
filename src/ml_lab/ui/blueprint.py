from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
from pathlib import Path
from typing import Any, Iterable

try:
    from flask import Blueprint, Response, abort, current_app, jsonify, redirect, render_template, request, send_file, url_for
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("ML Lab UI requires: pip install 'ml-lab[ui]'") from exc

from ml_lab import __version__, data
from ml_lab.application import PayloadError, execute_task
from ml_lab.core.serialization import to_jsonable

from .jobs import ProfileJobManager
from .training_jobs import TrainingJobManager
from .service import capability_groups, capability_records, get_capability
from .workspace import active_tool, project_scope, tool_records


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


def _transform_column_catalog(source: str) -> list[dict[str, Any]]:
    """Build lightweight UI metadata for column selection without changing data."""
    profile = data.profile_file(
        source,
        max_rows=5_000,
        relationship_rows=0,
        max_relationship_columns=2,
        max_relationship_pairs=0,
    )
    catalog: list[dict[str, Any]] = []
    for column in profile.columns:
        catalog.append({
            "name": column.name,
            "inferred_type": column.inferred_type,
            "missing_count": column.missing_count,
            "missing_rate": column.missing_rate,
            "unique_count": column.unique_count,
            "cardinality_ratio": column.cardinality_ratio,
            "constant": column.constant,
            "high_cardinality": column.unique_count >= 20 and column.cardinality_ratio >= 0.8,
        })
    return catalog


_TRANSFORM_GROUP_ORDER = [
    "select_columns",
    "drop_columns",
    "drop_quality_columns",
    "rename_columns",
    "coerce_types",
    "sentinel_to_missing",
    "fill_missing",
    "drop_missing_rows",
    "clean_strings",
    "drop_duplicates",
    "filter_rows",
    "derive",
    "encode_categorical",
    "scale",
    "outliers",
]


def _transform_group_order_from_form() -> list[str]:
    requested = _split_names(request.form.get("recipe_order", ""))
    ordered: list[str] = []
    for key in requested:
        if key in _TRANSFORM_GROUP_ORDER and key not in ordered:
            ordered.append(key)
    ordered.extend(key for key in _TRANSFORM_GROUP_ORDER if key not in ordered)
    return ordered


def _recipe_from_transform_form() -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {key: [] for key in _TRANSFORM_GROUP_ORDER}
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
        groups["select_columns"].append({"type": "select_columns", "columns": selected})
    if dropped:
        groups["drop_columns"].append({"type": "drop_columns", "columns": dropped})
    if request.form.get("drop_all_missing", "") == "1" or request.form.get("drop_constant", "") == "1":
        groups["drop_quality_columns"].append({
            "type": "drop_quality_columns",
            "all_missing": request.form.get("drop_all_missing", "") == "1",
            "constant": request.form.get("drop_constant", "") == "1",
            "protected_columns": _split_names(request.form.get("protected_quality_columns", "")),
        })
    if renames:
        groups["rename_columns"].append({"type": "rename_columns", "mapping": {key: str(value) for key, value in renames.items()}})
    if type_overrides:
        groups["coerce_types"].append({"type": "coerce_types", "mapping": type_overrides, "errors": request.form.get("coerce_errors", "coerce")})
    if sentinels:
        operation: dict[str, Any] = {"type": "sentinel_to_missing", "values": sentinels}
        if sentinel_columns:
            operation["columns"] = sentinel_columns
        groups["sentinel_to_missing"].append(operation)
    for column, value in fills.items():
        groups["fill_missing"].append({"type": "fill_missing", "columns": [column], "method": "constant", "value": value})
    if drop_missing_columns:
        groups["drop_missing_rows"].append({"type": "drop_missing_rows", "columns": drop_missing_columns, "how": request.form.get("drop_missing_how", "any")})
    if clean_columns:
        groups["clean_strings"].append({
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
        groups["drop_duplicates"].append(operation)

    filter_column = request.form.get("filter_column", "").strip()
    if filter_column:
        operator = request.form.get("filter_operator", "eq")
        operation = {"type": "filter_rows", "column": filter_column, "operator": operator}
        if operator not in {"is_missing", "not_missing"}:
            operation["value"] = _scalar(request.form.get("filter_value", ""))
        groups["filter_rows"].append(operation)

    derive_source = request.form.get("derive_source", "").strip()
    derive_target = request.form.get("derive_target", "").strip()
    derive_pattern = request.form.get("derive_pattern", "").strip()
    if derive_source or derive_target or derive_pattern:
        if not (derive_source and derive_target and derive_pattern):
            raise ValueError("Regex derivation requires source column, target column, and pattern.")
        groups["derive"].append({
            "type": "derive",
            "method": "regex_extract",
            "source": derive_source,
            "target": derive_target,
            "pattern": derive_pattern,
            "group": _form_int("derive_group", 0),
        })
    if encode_columns:
        groups["encode_categorical"].append({"type": "encode_categorical", "columns": encode_columns, "method": "one_hot", "drop_first": request.form.get("encode_drop_first", "") == "1"})
    if scale_columns:
        groups["scale"].append({"type": "scale", "columns": scale_columns, "method": request.form.get("scale_method", "standard")})
    if outlier_columns:
        groups["outliers"].append({
            "type": "outliers",
            "columns": outlier_columns,
            "method": request.form.get("outlier_method", "clip_iqr"),
            "iqr_multiplier": float(request.form.get("outlier_iqr_multiplier", "1.5")),
        })

    disabled = set(_split_names(request.form.get("recipe_disabled", "")))
    operations: list[dict[str, Any]] = []
    for group_key in _transform_group_order_from_form():
        if group_key in disabled:
            continue
        operations.extend(groups[group_key])

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
    profile_output_root: str | Path = "analysis/data/profile_runs",
    derived_output_root: str | Path = "analysis/data/derived",
    experiment_project_root: str | Path | None = None,
    path_autocomplete_roots: Iterable[str | Path] | None = None,
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
    # The selected project is explicit when hosted by PAH. Standalone defaults
    # to the process working directory, matching ExperimentStore's headless API.
    project_root = Path(experiment_project_root if experiment_project_root is not None else Path.cwd()).expanduser().resolve()
    training_jobs = TrainingJobManager(project_root)

    @blueprint.context_processor
    def _scientific_workspace_context():
        # Present on every old and new UI route without modifying service APIs.
        # Browser preferences never authorize access to a different project.
        endpoint = request.endpoint or ""
        own_endpoint = endpoint.removeprefix(blueprint.name + ".")
        return {
            "workspace_tools": tool_records(),
            "workspace_scope": project_scope(project_root, blueprint.name),
            "workspace_active_tool": active_tool(own_endpoint, request.args.get("family")),
        }

    raw_path_roots = list(path_autocomplete_roots) if path_autocomplete_roots is not None else [Path.cwd(), Path.home()]
    path_roots: list[Path] = []
    for raw_root in raw_path_roots:
        try:
            resolved_root = Path(raw_root).expanduser().resolve()
        except (OSError, RuntimeError):
            continue
        if resolved_root not in path_roots:
            path_roots.append(resolved_root)

    def _inside_path_root(path: Path) -> bool:
        for root in path_roots:
            try:
                path.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def _path_suggestion_record(path: Path) -> dict[str, Any]:
        is_dir = path.is_dir()
        return {
            "path": str(path),
            "name": path.name or str(path),
            "kind": "directory" if is_dir else "file",
        }

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

    def experiment_dataset_url(path: str) -> str | None:
        # Browsing arbitrary local files must not expose them as cross-project
        # experiment inputs; selection will independently validate containment.
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_relative_to(project_root):
            return None
        return url_for(request.blueprint + ".experiment_dataset", source_token=sign_path(str(resolved)))

    def workbench_dataset_url(path: str) -> str | None:
        """Allow an explicit post-extraction selection only for this project.

        The legacy Blueprint factory can be used without the unified workbench;
        only advertise the link if its route is actually installed by the host.
        The selection endpoint performs its own independent provenance checks.
        """
        if experiment_dataset_url(path) is None:
            return None
        endpoint = request.blueprint + ".workbench_select_file"
        if endpoint not in current_app.view_functions:
            return None
        return url_for(endpoint, path=path)

    def provenance_url(path: str) -> str:
        return url_for(request.blueprint + ".data_provenance", source_token=sign_path(path))

    def xml_structure_url(path: str) -> str:
        return url_for(request.blueprint + ".xml_structure_view", token=sign_path(path))

    @blueprint.get("/pah-shared/<asset>")
    def pah_shared_asset(asset: str):
        """Serve exact, packaged offline presentation assets; no copied UI engine."""
        if asset not in {"matrix_editor.mjs", "matrix.css"}:
            abort(404)
        from pah_shared import static_asset
        mime = "text/javascript" if asset.endswith(".mjs") else "text/css"
        return Response(static_asset(asset).read_bytes(), mimetype=mime,
                        headers={"Cache-Control": "private, max-age=3600"})

    @blueprint.get("/workspace")
    def workspace():
        """Free-navigation launcher. Selection is optional and non-mutating."""
        from ml_lab.experiments import ExperimentStore
        store = ExperimentStore(project_root)
        experiments = [row for row in store.list_experiments(limit=100)
                       if row["dataset_ref"].get("schema") == "ml-lab.experiment-dataset@1"]
        # Never silently default to a different experiment after project switch.
        return render_template("ml_lab_ui/workspace.html", version=__version__,
                               experiments=experiments, workspace_mode="workspace")

    @blueprint.get("/workspace/context")
    def workspace_context():
        """Read-only metadata lookup; never accepts filesystem paths or actions."""
        from ml_lab.experiments import ExperimentNotFound, ExperimentStore
        experiment_id = request.args.get("experiment_id", "").strip()
        run_id = request.args.get("run_id", "").strip()
        if not experiment_id or len(experiment_id) > 128 or len(run_id) > 128:
            abort(400)
        store = ExperimentStore(project_root)
        try:
            experiment = store.get_experiment(experiment_id)
            if experiment["dataset_ref"].get("schema") != "ml-lab.experiment-dataset@1":
                abort(404)
            runs = store.list_runs(experiment_id, limit=75)
            selected_run = None
            if run_id:
                selected_run = store.get_run(run_id)
                if selected_run["experiment_id"] != experiment_id:
                    abort(404)
        except ExperimentNotFound:
            abort(404)
        model = experiment["model_spec"]
        # The dataset path is a display label from an already authorized local
        # experiment, not an input to a file-read endpoint.
        payload = {
            "experiment_id": experiment["experiment_id"],
            "name": experiment["name"],
            "revision": experiment["revision"],
            "dataset": experiment["dataset_ref"].get("path", ""),
            "model_family": model.get("family", ""),
            "model_method": model.get("method", ""),
            "run_id": selected_run["run_id"] if selected_run else None,
            "runs": [{"run_id": row["run_id"], "revision": row["revision"],
                      "status": row["status"]} for row in runs],
        }
        response = jsonify(payload)
        response.headers["Cache-Control"] = "no-store"
        return response

    @blueprint.get("/workspace/results")
    def workspace_results():
        """Run browser without requiring the user to finish a model workflow."""
        from ml_lab.experiments import ExperimentNotFound, ExperimentStore
        store = ExperimentStore(project_root)
        experiments = [row for row in store.list_experiments(limit=100)
                       if row["dataset_ref"].get("schema") == "ml-lab.experiment-dataset@1"]
        experiment_id = request.args.get("experiment_id", "").strip()
        if len(experiment_id) > 128:
            abort(400)
        experiment = None
        runs = []
        if experiment_id:
            try:
                experiment = store.get_experiment(experiment_id)
                if experiment["dataset_ref"].get("schema") != "ml-lab.experiment-dataset@1":
                    abort(404)
                runs = store.list_runs(experiment_id, limit=100)
            except ExperimentNotFound:
                abort(404)
        return render_template("ml_lab_ui/workspace_results.html", version=__version__,
                               experiments=experiments, experiment=experiment, runs=runs)

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
            xml_structure_url=xml_structure_url,
            recent_runs=data.recent_profile_runs(output_root=profile_output_root, limit=12),
            experimental_enabled=enable_experimental,
        )


    @blueprint.route("/data/xml/catalog", methods=["GET", "POST"])
    def xml_catalog_browser():
        """Read-only indexed metadata; no implicit filesystem scan or rebuild."""
        from ml_lab.data.xml_catalog_browser import browse_xml_catalog, SECTIONS

        # POST is explicit; GET never scans snapshots or rebuilds an index.
        refresh_report = None
        if request.method == "POST":
            from ml_lab.data.xml_catalog_refresh import refresh_xml_catalog
            from urllib.parse import urlsplit
            origin = request.headers.get("Origin")
            # Reject an explicitly foreign Origin, without depending on proxies
            # setting one for same-origin form submissions.
            if origin:
                observed = urlsplit(origin)
                if observed.scheme != request.scheme or observed.netloc != request.host:
                    return ("Cross-origin catalog refresh denied", 403)
            if request.form.get("confirm") != "rebuild":
                return ("Catalog refresh requires explicit confirmation", 400)
            try:
                refresh_report = refresh_xml_catalog(project_root, scope=request.form.get("scope", "all"))
            except (ValueError, OSError) as exc:
                return ("Invalid catalog refresh request: " + str(exc), 400)
        section = request.args.get("section", "collections")
        source_sha = request.args.get("sha256", "").strip() or None
        inventory_id = request.args.get("inventory", "").strip() or None
        artifact_kind = request.args.get("kind", "").strip() or None
        try:
            limit = int(request.args.get("limit", "25"))
            offset = int(request.args.get("offset", "0"))
            if section not in SECTIONS:
                raise ValueError("Unknown section")
            filters = {"section": section, "limit": limit, "offset": offset}
            if section in ("sources", "recipes", "artifacts"):
                filters["source_sha256"] = source_sha
            if section in ("sources", "recipes"):
                filters["inventory_snapshot_id"] = inventory_id
            if section == "artifacts":
                filters["artifact_kind"] = artifact_kind
            page = browse_xml_catalog(project_root, **filters)
        except (ValueError, TypeError, OSError) as exc:
            return render_template("ml_lab_ui/xml_catalog.html", version=__version__,
                                   page=None, section=section, source_sha=source_sha,
                                   inventory_id=inventory_id, artifact_kind=artifact_kind,
                                   error=str(exc), refresh_report=refresh_report), 400
        return render_template("ml_lab_ui/xml_catalog.html", version=__version__,
                               page=page, section=section, source_sha=source_sha,
                               inventory_id=inventory_id, artifact_kind=artifact_kind,
                               error=None, refresh_report=refresh_report)

    @blueprint.get("/data/xml/workbench")
    def xml_workbench():
        """One entry point for legacy single-file and collection XML workflows."""
        return render_template("ml_lab_ui/xml_workbench.html", version=__version__)

    @blueprint.post("/data/xml/open")
    def xml_open():
        """Open one selected XML in the existing hierarchy explorer.

        This does not modify the file or implicitly start an extraction.
        """
        source = Path(request.form.get("source", "").strip()).expanduser()
        if (not source.is_file() or source.is_symlink()
                or source.suffix.casefold() != ".xml"):
            return redirect(url_for(request.blueprint + ".xml_workbench", error="invalid_source"))
        return redirect(xml_structure_url(str(source)))

    @blueprint.route("/data/xml/collection", methods=["GET", "POST"])
    def xml_collection_inventory():
        # GET restores source/filter choices for back-navigation, but NEVER
        # discovers or parses files. Only an explicit POST performs inspection.
        values = request.form if request.method == "POST" else request.args
        directory = values.get("directory", "").strip()
        recursive = values.get("recursive", "1") == "1"
        include_hidden = values.get("include_hidden", "") == "1"
        max_files = _form_int("max_files", 1000) if request.method == "POST" else 1000
        if request.method == "GET":
            try:
                max_files = int(values.get("max_files", 1000))
            except (ValueError, TypeError):
                max_files = 1000
            max_files = max(1, min(100000, max_files))
        include_glob = values.get("include_glob", "*.xml").strip() or "*.xml"
        exclude_glob = values.get("exclude_glob", "").strip()
        inventory = None
        comparison = None
        error = None
        compare = values.get("compare", "") == "1"
        minimum_similarity = values.get("minimum_similarity", "0.80")
        if request.method == "POST":
            try:
                if not directory:
                    raise ValueError("Select a local XML directory.")
                collection = data.inventory_xml_collection(
                    directory, recursive=recursive, include_hidden=include_hidden,
                    include_globs=(include_glob,),
                    exclude_globs=(exclude_glob,) if exclude_glob else (),
                    max_files=max_files,
                )
                inventory = collection.to_record()
                if compare:
                    comparison = data.compare_xml_collection(
                        collection, minimum_similarity=float(minimum_similarity),
                        max_sources=100,
                    ).to_record()
            except (OSError, TypeError, ValueError) as exc:
                error = f"{type(exc).__name__}: {exc}"
        return render_template(
            "ml_lab_ui/xml_collection_inventory.html", version=__version__,
            directory=directory, recursive=recursive, include_hidden=include_hidden,
            include_glob=include_glob, exclude_glob=exclude_glob, max_files=max_files,
            inventory=inventory, comparison=comparison, compare=compare,
            minimum_similarity=minimum_similarity, error=error, xml_structure_url=xml_structure_url,
        )

    @blueprint.route("/data/xml/collection/review", methods=["GET", "POST"])
    def xml_collection_review():
        """Read-only C3 review: no persistent approval, materialization, or CSV."""
        from ml_lab.data.xml_collection_review import review_xml_collection
        values = request.form if request.method == "POST" else request.args
        directory = values.get("directory", "").strip()
        recursive = values.get("recursive", "1") == "1"
        include_hidden = values.get("include_hidden", "") == "1"
        try:
            requested_max_files = int(values.get("max_files", 40))
        except (TypeError, ValueError):
            requested_max_files = 40
        max_files = min(requested_max_files, 80)
        review_limit_notice = ("Review is limited to 80 XML files per request. Narrow the include/exclude filters "
                               "if the directory contains more than 80 matching files.") if requested_max_files > 80 else None
        include_glob = values.get("include_glob", "*.xml").strip() or "*.xml"
        exclude_glob = values.get("exclude_glob", "").strip()
        fraction_raw = values.get("min_shared_fraction", "0.5")
        selected = request.form.getlist("selected") if request.method == "POST" else []
        acknowledge_variants = values.get("acknowledge_variants", "") == "1"
        mode = values.get("mode", "inspect")
        report = None
        review = None
        structure_urls = {}
        error = None
        try:
            min_shared_fraction = float(fraction_raw)
            if not 0 <= min_shared_fraction <= 1:
                raise ValueError("Minimum shared fraction must be between 0 and 1.")
            # C2 emits all pair combinations; keep synchronous HTML work bounded.
            if requested_max_files < 1:
                raise ValueError("Review must allow at least one XML file.")
            if request.method == "POST":
                if not directory:
                    raise ValueError("Select a local XML directory.")
                inventory = data.inventory_xml_collection(
                    directory, recursive=recursive, include_hidden=include_hidden,
                    include_globs=(include_glob,),
                    exclude_globs=(exclude_glob,) if exclude_glob else (),
                    max_files=max_files,
                )
                report = data.compare_xml_collection(inventory, min_shared_fraction=min_shared_fraction).to_record()
                structure_urls = {item.relative_path: xml_structure_url(item.path)
                                  for item in inventory.files if item.status == "parsed"}
                if mode in {"review", "export"}:
                    review = review_xml_collection(report, selected, acknowledge_variants=acknowledge_variants)
                    if mode == "export":
                        if review["status"] != "ready_for_recipe_design":
                            raise ValueError("Selection is blocked; resolve issues before exporting a review snapshot.")
                        response = jsonify(review)
                        response.headers["Content-Disposition"] = 'attachment; filename="xml_collection_review.json"'
                        response.headers["Cache-Control"] = "no-store"
                        return response
                elif mode != "inspect":
                    raise ValueError("Unknown XML collection review operation.")
        except (OSError, TypeError, ValueError, KeyError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        return render_template(
            "ml_lab_ui/xml_collection_review.html", version=__version__,
            directory=directory, recursive=recursive, include_hidden=include_hidden,
            include_glob=include_glob, exclude_glob=exclude_glob, max_files=max_files,
            min_shared_fraction=fraction_raw, selected=selected,
            acknowledge_variants=acknowledge_variants, report=report, review=review,
            max_visible_pairs=200, structure_urls=structure_urls,
            review_limit_notice=review_limit_notice, error=error,
        ), (400 if error else 200)

    @blueprint.route("/data/xml/collection/recipe", methods=["GET", "POST"])
    def xml_collection_recipe():
        """C4 recipe design only: never extract rows or write tabular datasets."""
        from ml_lab.data.xml_collection_recipe import (
            inspect_xml_collection_recipe, build_xml_collection_recipe,
            prepare_full_xml_collection_recipe,
        )
        directory = request.form.get("directory", "").strip()
        recursive = request.form.get("recursive", "") == "1"
        include_hidden = request.form.get("include_hidden", "") == "1"
        include_glob = request.form.get("include_glob", "*.xml").strip() or "*.xml"
        exclude_glob = request.form.get("exclude_glob", "").strip()
        max_files = min(_form_int("max_files", 40), 80)
        fraction = request.form.get("min_shared_fraction", "0.5")
        selected = request.form.getlist("selected")
        acknowledged = request.form.get("acknowledge_variants", "") == "1"
        record_root = request.form.get("record_root", "").strip()
        fields = request.form.getlist("field")
        allow_missing = request.form.get("allow_missing_fields", "") == "1"
        mode = request.form.get("mode", "inspect")
        auto_all = mode == "auto_all" or request.form.get("auto_all", "") == "1"
        catalog = recipe = None
        error = None
        try:
            if request.method == "POST":
                if not directory or not 1 <= max_files <= 80:
                    raise ValueError("Choose an XML directory and a maximum of 1–80 files.")
                shared_fraction = float(fraction)
                if not 0 <= shared_fraction <= 1:
                    raise ValueError("Minimum shared fraction must be between 0 and 1.")
                inventory = data.inventory_xml_collection(
                    directory, recursive=recursive, include_hidden=include_hidden,
                    include_globs=(include_glob,), exclude_globs=(exclude_glob,) if exclude_glob else (),
                    max_files=max_files,
                )
                report = data.compare_xml_collection(inventory, min_shared_fraction=shared_fraction)

                if auto_all:
                    catalog, recipe = prepare_full_xml_collection_recipe(
                        inventory, report, selected,
                    )
                    acknowledged = True
                    record_root = str(recipe["record_root_canonical_path"])
                    fields = list(recipe["selected_field_ids"])
                    allow_missing = True
                else:
                    catalog = inspect_xml_collection_recipe(
                        inventory, report, selected, acknowledge_variants=acknowledged,
                        record_root_canonical_path=record_root or None,
                    )
                    if mode in {"branches", "validate", "export"}:
                        paths = request.form.getlist("branch_path")
                        rules = []
                        for branch in paths:
                            strategy = request.form.get(f"strategy::{branch}", "").strip()
                            if strategy:
                                rules.append({
                                    "branch_canonical_path": branch, "strategy": strategy,
                                    "join_delimiter": request.form.get(f"join_delimiter::{branch}", ""),
                                    "aggregate_operation": request.form.get(f"aggregate_operation::{branch}", ""),
                                    "pivot_key_field_id": request.form.get(f"pivot_key_field_id::{branch}", ""),
                                    "pivot_value_field_id": request.form.get(f"pivot_value_field_id::{branch}", ""),
                                    "separate_table_name": request.form.get(f"separate_table_name::{branch}", ""),
                                })
                        recipe = build_xml_collection_recipe(
                            inventory, report, selected, record_root, fields,
                            acknowledge_variants=acknowledged, allow_missing_fields=allow_missing,
                            rules=rules,
                        )
                        if mode == "export":
                            if recipe["status"] != "ready_for_collection_preview":
                                raise ValueError("Resolve the recipe's blocking issues before exporting.")
                            response = jsonify(recipe)
                            response.headers["Content-Disposition"] = 'attachment; filename="xml_collection_recipe.json"'
                            response.headers["Cache-Control"] = "no-store"
                            return response
                    elif mode not in {"inspect", "fields"}:
                        raise ValueError("Unsupported XML recipe operation")
        except (OSError, KeyError, LookupError, TypeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        return render_template(
            "ml_lab_ui/xml_collection_recipe.html", version=__version__,
            directory=directory, recursive=recursive, include_hidden=include_hidden,
            include_glob=include_glob, exclude_glob=exclude_glob,
            max_files=max_files, min_shared_fraction=fraction,
            selected=selected, acknowledge_variants=acknowledged,
            record_root=record_root, fields=fields, allow_missing_fields=allow_missing,
            auto_all=auto_all, catalog=catalog, recipe=recipe, error=error,
            strategy_catalog=data.xml_collection_strategy_catalog(),
        ), (400 if error else 200)

    @blueprint.route("/data/xml/collection/export", methods=["GET", "POST"])
    def xml_collection_export():
        """C5 consolidated output: explicit review and confirmation, never implicit export."""
        from ml_lab.data.xml_collection_export import (
            preview_xml_collection_export, materialize_xml_collection,
        )
        recipe_text = request.form.get("recipe_json", "")
        output = request.form.get("output", str(project_root / "analysis/data/xml_collection.csv")).strip()
        mode = request.form.get("mode", "preview")
        preview = result = error = None
        try:
            if request.method == "POST":
                if len(recipe_text) > 1_000_000:
                    raise ValueError("Recipe JSON is too large")
                recipe = json.loads(recipe_text)
                if not isinstance(recipe, dict):
                    raise ValueError("Recipe JSON must be an object")
                if mode == "preview":
                    preview = preview_xml_collection_export(recipe)
                elif mode == "export":
                    if request.form.get("confirm", "") != "1":
                        raise ValueError("Confirm consolidated CSV generation explicitly")
                    result = materialize_xml_collection(
                        recipe, output, confirm=True,
                        overwrite=request.form.get("overwrite", "") == "1",
                    )
                else:
                    raise ValueError("Unsupported XML collection export action")
        except (OSError, KeyError, TypeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        return render_template(
            "ml_lab_ui/xml_collection_export.html", version=__version__,
            recipe_text=recipe_text, output=output, preview=preview,
            result=result, error=error,
            collection_provenance_url=(url_for(request.blueprint + ".xml_collection_provenance",
                                               provenance_path=result["provenance"]["path"],
                                               output=result["main_dataset"]["path"])
                                       if result else None),
            workbench_select_url=(workbench_dataset_url(result["main_dataset"]["path"])
                                  if result else None),
        ), (400 if error else 200)

    @blueprint.route("/data/xml/collection/provenance", methods=["GET", "POST"])
    def xml_collection_provenance():
        """C6: local verification, checkpoint audit, and explicitly confirmed replay."""
        from ml_lab.data.xml_collection_provenance import (
            verify_collection_event, inspect_collection_recovery,
            resume_xml_collection_export, regenerate_xml_collection,
        )
        action = request.form.get("action", "verify")
        values = request.form if request.method == "POST" else request.args
        event_path = values.get("provenance_path", "").strip()
        output = values.get("output", "").strip()
        result = error = None
        try:
            if request.method == "POST":
                if any(len(str(value)) > 4096 for value in (event_path, output)):
                    raise ValueError("Provenance/output path is too long")
                if action == "verify":
                    if not event_path:
                        raise ValueError("Select a provenance JSON file")
                    result = verify_collection_event(event_path)
                elif action == "inspect":
                    if not output:
                        raise ValueError("Select a collection CSV/TSV output path")
                    result = inspect_collection_recovery(output)
                elif action in {"resume", "regenerate"}:
                    if request.form.get("confirm") != "1":
                        raise ValueError("Explicit confirmation is required to write files")
                    if action == "resume":
                        if not output:
                            raise ValueError("Select the checkpoint output path")
                        result = resume_xml_collection_export(output, confirm=True)
                    else:
                        if not event_path or not output:
                            raise ValueError("Provide the provenance file and a new output path")
                        result = regenerate_xml_collection(event_path, output, confirm=True)
                else:
                    raise ValueError("Unrecognized C6 action")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        return render_template(
            "ml_lab_ui/xml_collection_provenance.html", version=__version__,
            action=action, event_path=event_path, output=output, result=result, error=error,
        ), (400 if error else 200)

    @blueprint.get("/data/xml/<token>")
    def xml_structure_view(token: str):
        path = verify_path(token)
        if path.suffix.casefold() != ".xml":
            abort(400)
        try:
            record = data.inspect_file(path)
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            return render_template(
                "ml_lab_ui/xml_structure.html",
                version=__version__,
                source_name=path.name,
                source_path=str(path),
                structure=None,
                structure_json=None,
                error=f"{type(exc).__name__}: {exc}",
                data_url=url_for(request.blueprint + ".data_lab"),
            ), 400
        if record.format != "xml":
            abort(400)
        if record.parse_status == "failed" or record.xml_structure is None:
            return render_template(
                "ml_lab_ui/xml_structure.html",
                version=__version__,
                source_name=path.name,
                source_path=str(path),
                structure=None,
                structure_json=None,
                error=record.error or "XML structure could not be loaded.",
                data_url=url_for(request.blueprint + ".data_lab"),
            ), 400

        structure = record.xml_structure.to_record()
        return render_template(
            "ml_lab_ui/xml_structure.html",
            version=__version__,
            source_name=path.name,
            source_path=str(path),
            structure=structure,
            structure_json=_pretty(structure),
            error=None,
            data_url=url_for(request.blueprint + ".data_lab"),
            record_fields_url=url_for(request.blueprint + ".xml_record_fields", token=token),
            collection_plan_url=url_for(request.blueprint + ".xml_collection_plan", token=token),
            preview_url=url_for(request.blueprint + ".xml_tabular_preview", token=token),
            materialize_url=url_for(request.blueprint + ".xml_tabular_materialize", token=token),
            extraction_history_url=url_for(request.blueprint + ".xml_extraction_history", token=token),
            extraction_reentry_url=url_for(request.blueprint + ".xml_extraction_reentry", token=token),
            extraction_compare_url=url_for(request.blueprint + ".xml_extraction_compare", token=token),
            default_materialize_output=str(Path(derived_output_root).expanduser().resolve() / f"{path.stem}-tabular.csv"),
        )


    @blueprint.post("/data/xml/<token>/typed-preview")
    def xml_typed_preview(token: str):
        """Explicitly requested, bounded, read-only typed XML preview (P5-15)."""
        from ml_lab.data.xml_typed_records import preview_typed_xml_records
        from ml_lab.data.xml_typed_representation import (
            XmlTypedRepresentationError, build_typed_recipe,
        )

        path = verify_path(token)
        if path.suffix.casefold() != ".xml":
            abort(400)
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "JSON object body required"}), 400
        # Bounded at the HTTP boundary as well as in the representation engine.
        variables = payload.get("variables")
        if not isinstance(variables, list) or not 1 <= len(variables) <= 12:
            return jsonify({"error": "Select between 1 and 12 typed variables"}), 400
        if any(not isinstance(v, dict) or set(v) != {"canonical_path", "representation"}
               or not isinstance(v["canonical_path"], str) or not isinstance(v["representation"], str)
               for v in variables):
            return jsonify({"error": "Malformed variable declaration"}), 400
        if not isinstance(payload.get("record_root_canonical_path"), str):
            return jsonify({"error": "Record root required"}), 400
        count = payload.get("max_records", 5)
        if type(count) is not int or not 1 <= count <= 10:
            return jsonify({"error": "max_records must be an integer from 1 to 10"}), 400
        if payload.get("linked_child_paths", []) != []:
            return jsonify({"error": "Linked-child preview is not enabled in this workspace"}), 400
        if len({v["canonical_path"] for v in variables}) != len(variables):
            return jsonify({"error": "Duplicate typed variables"}), 400
        try:
            record = data.inspect_file(path)
            if record.format != "xml" or record.parse_status == "failed" or record.xml_structure is None:
                raise ValueError(record.error or "XML structure could not be loaded")
            recipe = build_typed_recipe(
                record_root_canonical_path=payload["record_root_canonical_path"],
                variables={v["canonical_path"]: v["representation"] for v in variables},
            )
            preview = preview_typed_xml_records(
                path, record.xml_structure, recipe, max_records=count, max_nodes_per_value=2000,
            )
        except (OSError, UnicodeError, ValueError, TypeError, XmlTypedRepresentationError) as exc:
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400
        response = jsonify(preview)
        response.headers["Cache-Control"] = "no-store"
        return response

    def _typed_workspace_request(token: str, *, full: bool) -> tuple[Path, Any, dict[str, Any], int]:
        """Revalidate a typed request and inspect the source; never trust browser previews.

        A typed operation remains independent from C1--C7 recipes and dataset APIs.
        The full-document path is deliberately capped for synchronous UI execution.
        """
        from ml_lab.data.xml_typed_representation import build_typed_recipe

        path = verify_path(token)
        if path.suffix.casefold() != ".xml":
            abort(400)
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or set(payload) != {
            "record_root_canonical_path", "variables", "max_records"
        }:
            raise ValueError("Expected record root, variables and max_records only")
        variables = payload["variables"]
        if not isinstance(variables, list) or not 1 <= len(variables) <= 12:
            raise ValueError("Select between 1 and 12 typed variables")
        if any(not isinstance(v, dict) or set(v) != {"canonical_path", "representation"}
               or not isinstance(v["canonical_path"], str) or not isinstance(v["representation"], str)
               for v in variables):
            raise ValueError("Malformed typed variable declaration")
        if len({v["canonical_path"] for v in variables}) != len(variables):
            raise ValueError("Duplicate typed variable paths")
        count = payload["max_records"]
        if type(count) is not int or (not full and not 1 <= count <= 10) or (full and count != 5000):
            raise ValueError("Invalid typed record limit")
        record = data.inspect_file(path)
        if record.format != "xml" or record.parse_status == "failed" or record.xml_structure is None:
            raise ValueError(record.error or "XML structure could not be loaded")
        recipe = build_typed_recipe(
            record_root_canonical_path=payload["record_root_canonical_path"],
            variables={v["canonical_path"]: v["representation"] for v in variables},
        )
        return path, record.xml_structure, recipe, count

    @blueprint.post("/data/xml/<token>/typed-save-preview")
    def xml_typed_save_preview(token: str):
        """Explicit save of a freshly recomputed bounded preview; no dataset creation."""
        from ml_lab.data.xml_typed_records import preview_typed_xml_records
        from ml_lab.data.xml_typed_artifacts import persist_typed_preview
        from ml_lab.data.xml_typed_representation import XmlTypedRepresentationError

        try:
            path, structure, recipe, count = _typed_workspace_request(token, full=False)
            preview = preview_typed_xml_records(
                path, structure, recipe, max_records=count, max_nodes_per_value=2000,
            )
            result = persist_typed_preview(project_root, preview, recipe=recipe)
        except (OSError, UnicodeError, ValueError, TypeError, XmlTypedRepresentationError) as exc:
            response = jsonify({"error": f"{type(exc).__name__}: {exc}"})
            response.headers["Cache-Control"] = "no-store"
            return response, 400
        response = jsonify(result)
        response.headers["Cache-Control"] = "no-store"
        return response

    @blueprint.post("/data/xml/<token>/typed-materialize")
    def xml_typed_materialize(token: str):
        """Explicit and capped complete representation; never registers a dataset."""
        from ml_lab.data.xml_typed_materialize import materialize_typed_xml_records
        from ml_lab.data.xml_typed_representation import XmlTypedRepresentationError

        try:
            path, structure, recipe, _ = _typed_workspace_request(token, full=True)
            result = materialize_typed_xml_records(
                project_root, path, structure, recipe,
                max_records=5000, max_output_bytes=32 * 1024 * 1024,
                max_nodes_per_value=2000,
            )
        except (OSError, UnicodeError, ValueError, TypeError, XmlTypedRepresentationError) as exc:
            response = jsonify({"error": f"{type(exc).__name__}: {exc}"})
            response.headers["Cache-Control"] = "no-store"
            return response, 400
        response = jsonify(result)
        response.headers["Cache-Control"] = "no-store"
        return response

    @blueprint.get("/data/xml/<token>/record-fields")
    def xml_record_fields(token: str):
        path = verify_path(token)
        if path.suffix.casefold() != ".xml":
            abort(400)
        root = request.args.get("root", "").strip()
        try:
            record = data.inspect_file(path)
            if record.format != "xml" or record.parse_status == "failed" or record.xml_structure is None:
                raise ValueError(record.error or "XML structure could not be loaded.")
            selection = data.build_xml_record_selection(record.xml_structure, root)
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400
        return jsonify(selection.to_record())


    @blueprint.post("/data/xml/<token>/collection-plan")
    def xml_collection_plan(token: str):
        path = verify_path(token)
        if path.suffix.casefold() != ".xml":
            abort(400)
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "ValueError: JSON object body is required"}), 400
        root = str(payload.get("record_root_canonical_path") or "").strip()
        selected_field_ids = payload.get("selected_field_ids", [])
        rules = payload.get("rules", [])
        if not isinstance(selected_field_ids, list):
            return jsonify({"error": "ValueError: selected_field_ids must be a list"}), 400
        if not isinstance(rules, list):
            return jsonify({"error": "ValueError: rules must be a list"}), 400
        try:
            record = data.inspect_file(path)
            if record.format != "xml" or record.parse_status == "failed" or record.xml_structure is None:
                raise ValueError(record.error or "XML structure could not be loaded.")
            plan = data.build_xml_collection_plan(
                record.xml_structure,
                root,
                selected_field_ids,
                rules=rules,
            )
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400
        return jsonify(plan.to_record())


    @blueprint.post("/data/xml/<token>/preview")
    def xml_tabular_preview(token: str):
        path = verify_path(token)
        if path.suffix.casefold() != ".xml":
            abort(400)
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "ValueError: JSON object body is required"}), 400
        root = str(payload.get("record_root_canonical_path") or "").strip()
        selected_field_ids = payload.get("selected_field_ids", [])
        rules = payload.get("rules", [])
        max_rows = payload.get("max_rows", 25)
        if not isinstance(selected_field_ids, list):
            return jsonify({"error": "ValueError: selected_field_ids must be a list"}), 400
        if not isinstance(rules, list):
            return jsonify({"error": "ValueError: rules must be a list"}), 400
        try:
            max_rows = int(max_rows)
            record = data.inspect_file(path)
            if record.format != "xml" or record.parse_status == "failed" or record.xml_structure is None:
                raise ValueError(record.error or "XML structure could not be loaded.")
            preview = data.preview_xml_tabularization(
                path,
                record.xml_structure,
                root,
                selected_field_ids,
                rules=rules,
                max_rows=max_rows,
            )
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400
        return jsonify(preview.to_record())


    @blueprint.post("/data/xml/<token>/materialize")
    def xml_tabular_materialize(token: str):
        path = verify_path(token)
        if path.suffix.casefold() != ".xml":
            abort(400)
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "ValueError: JSON object body is required"}), 400
        root = str(payload.get("record_root_canonical_path") or "").strip()
        selected_field_ids = payload.get("selected_field_ids", [])
        rules = payload.get("rules", [])
        confirmed_signature = str(payload.get("confirmed_preview_signature") or "").strip()
        output = str(payload.get("output") or "").strip()
        overwrite = bool(payload.get("overwrite", False))
        if not isinstance(selected_field_ids, list):
            return jsonify({"error": "ValueError: selected_field_ids must be a list"}), 400
        if not isinstance(rules, list):
            return jsonify({"error": "ValueError: rules must be a list"}), 400
        if not output:
            output = str(Path(derived_output_root).expanduser().resolve() / f"{path.stem}-tabular.csv")
        try:
            record = data.inspect_file(path)
            if record.format != "xml" or record.parse_status == "failed" or record.xml_structure is None:
                raise ValueError(record.error or "XML structure could not be loaded.")
            result = data.materialize_xml_tabularization(
                path,
                record.xml_structure,
                root,
                selected_field_ids,
                rules=rules,
                confirmed_preview_signature=confirmed_signature,
                output=output,
                overwrite=overwrite,
                history_root=derived_output_root,
            )
        except (OSError, UnicodeError, ValueError, TypeError, FileExistsError) as exc:
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400
        response = result.to_record()
        response["transform_url"] = transform_url(result.dataset_path)
        response["data_url"] = url_for(request.blueprint + ".data_lab")
        # Same explicit, project-scoped handoff as collection export. Do not
        # mutate the active dataset or create experiments during extraction.
        response["workbench_select_url"] = workbench_dataset_url(result.dataset_path)
        return jsonify(response)


    @blueprint.get("/data/xml/<token>/extractions")
    def xml_extraction_history(token: str):
        path = verify_path(token)
        if path.suffix.casefold() != ".xml":
            abort(400)
        try:
            record = data.inspect_file(path)
            if record.format != "xml" or record.parse_status == "failed" or record.xml_structure is None:
                raise ValueError(record.error or "XML structure could not be loaded.")
            history = data.refresh_xml_extraction_history(
                record.xml_structure,
                history_root=derived_output_root,
            )
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400
        return jsonify(history)


    @blueprint.post("/data/xml/<token>/extractions/reenter")
    def xml_extraction_reentry(token: str):
        path = verify_path(token)
        if path.suffix.casefold() != ".xml":
            abort(400)
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "ValueError: JSON object body is required"}), 400
        event_id = str(payload.get("event_id") or "").strip()
        if not event_id:
            return jsonify({"error": "ValueError: event_id is required"}), 400
        try:
            record = data.inspect_file(path)
            if record.format != "xml" or record.parse_status == "failed" or record.xml_structure is None:
                raise ValueError(record.error or "XML structure could not be loaded.")
            state = data.load_xml_extraction_state(
                record.xml_structure,
                event_id,
                history_root=derived_output_root,
            )
        except (OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400
        return jsonify(state)


    @blueprint.post("/data/xml/<token>/extractions/compare")
    def xml_extraction_compare(token: str):
        path = verify_path(token)
        if path.suffix.casefold() != ".xml":
            abort(400)
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "ValueError: JSON object body is required"}), 400
        left_event_id = str(payload.get("left_event_id") or "").strip()
        right_event_id = str(payload.get("right_event_id") or "").strip()
        if not left_event_id or not right_event_id:
            return jsonify({"error": "ValueError: left_event_id and right_event_id are required"}), 400
        try:
            record = data.inspect_file(path)
            if record.format != "xml" or record.parse_status == "failed" or record.xml_structure is None:
                raise ValueError(record.error or "XML structure could not be loaded.")
            comparison = data.compare_xml_extraction_recipes(
                record.xml_structure,
                left_event_id,
                right_event_id,
                history_root=derived_output_root,
            )
        except (OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400
        return jsonify(comparison)


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

    @blueprint.get("/api/path-suggestions")
    def path_suggestions():
        query = request.args.get("q", "").strip()
        kind = request.args.get("kind", "either").strip().lower()
        if kind not in {"file", "directory", "either"}:
            return jsonify({"error": "kind must be file, directory, or either"}), 400

        extensions = {
            item if item.startswith(".") else f".{item}"
            for item in _split_names(request.args.get("extensions", ""))
        }
        extensions = {item.casefold() for item in extensions if item != "."}
        limit = min(max(request.args.get("limit", type=int) or 30, 1), 100)

        if not path_roots:
            return jsonify({"suggestions": [], "roots": []})

        expanded = Path(query).expanduser() if query else None
        if expanded is not None and expanded.is_absolute():
            candidate = expanded
        elif query:
            candidate = path_roots[0] / expanded
        else:
            candidate = path_roots[0]

        if query.endswith(("/", "\\")):
            directory = candidate
            prefix = ""
        else:
            directory = candidate.parent if query else candidate
            prefix = candidate.name if query else ""

        try:
            directory = directory.resolve()
        except (OSError, RuntimeError):
            return jsonify({"suggestions": [], "roots": [str(root) for root in path_roots]})

        if not _inside_path_root(directory) or not directory.is_dir():
            return jsonify({"suggestions": [], "roots": [str(root) for root in path_roots]})

        suggestions: list[dict[str, Any]] = []
        prefix_folded = prefix.casefold()
        try:
            entries = sorted(directory.iterdir(), key=lambda item: (not item.is_dir(), item.name.casefold()))
        except OSError:
            entries = []
        for entry in entries:
            if prefix_folded and not entry.name.casefold().startswith(prefix_folded):
                continue
            try:
                resolved_entry = entry.resolve()
                if not _inside_path_root(resolved_entry):
                    continue
                is_dir = resolved_entry.is_dir()
                is_file = resolved_entry.is_file()
            except (OSError, RuntimeError):
                continue
            if is_dir:
                suggestions.append(_path_suggestion_record(resolved_entry))
            elif is_file and kind != "directory":
                if extensions and resolved_entry.suffix.casefold() not in extensions:
                    continue
                suggestions.append(_path_suggestion_record(resolved_entry))
            if len(suggestions) >= limit:
                break

        if kind == "directory":
            suggestions = [item for item in suggestions if item["kind"] == "directory"]

        return jsonify({"suggestions": suggestions, "roots": [str(root) for root in path_roots]})

    @blueprint.route("/data/transform", methods=["GET", "POST"])
    def data_transform():
        source = request.form.get("source", "").strip()
        source_token = request.args.get("source_token", "")
        if request.method == "GET" and source_token:
            source = str(verify_path(source_token))
        recipe = None
        result = None
        error = None
        column_catalog = None
        column_catalog_error = None
        source_table = None
        source_table_error = None
        applied = False
        mode = request.form.get("mode", "preview")
        if request.method == "POST":
            try:
                if not source:
                    raise ValueError("Choose a CSV/TSV source path.")
                if mode != "columns":
                    recipe = _recipe_from_transform_form()
                    payload = {
                        "path": source,
                        "recipe": recipe,
                        "preview_rows": _form_int("preview_rows", 50),
                    }
                    if mode == "apply":
                        requested_output = request.form.get("output", "").strip()
                        if requested_output:
                            payload["output"] = requested_output
                        else:
                            source_path = Path(source).expanduser().resolve()
                            output_root = Path(derived_output_root).expanduser().resolve()
                            payload["output"] = str(output_root / f"{source_path.stem}-derived{source_path.suffix.lower()}")
                        payload["overwrite"] = request.form.get("overwrite", "") == "1"
                        result = execute_task("data.transform.apply", payload)["result"]
                        applied = True
                    else:
                        result = execute_task("data.transform.preview", payload)["result"]
            except (PayloadError, OSError, UnicodeError, ValueError, TypeError, KeyError, FileExistsError, json.JSONDecodeError) as exc:
                error = f"{type(exc).__name__}: {exc}"

        if source and (request.method == "POST" or source_token):
            try:
                column_catalog = _transform_column_catalog(source)
            except (OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
                column_catalog_error = f"{type(exc).__name__}: {exc}"
            try:
                source_table = execute_task(
                    "data.table",
                    {
                        "path": source,
                        "page": 1,
                        "page_size": min(max(_form_int("preview_rows", 50), 1), 100),
                    },
                )["result"]
            except (PayloadError, OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
                source_table_error = f"{type(exc).__name__}: {exc}"
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
            column_catalog=column_catalog,
            column_catalog_error=column_catalog_error,
            source_table=source_table,
            source_table_error=source_table_error,
            provenance_url=(provenance_url(result["derived"]["path"]) if applied and result else None),
            experiment_url=(experiment_dataset_url(result["derived"]["path"]) if applied and result else None),
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

    @blueprint.route("/data/experiment/selection", methods=["GET", "POST"])
    def experiment_dataset():
        """Explicit Data Lab -> experiment handoff; no automatic dataset switch."""
        from ml_lab.experiments import (
            DatasetSelectionError, ExperimentStore, ExperimentStoreError,
            describe_selection, select_experiment_dataset,
        )

        token = (request.form if request.method == "POST" else request.args).get("source_token", "")
        source_path = verify_path(token)
        try:
            store = ExperimentStore(project_root)
            preview = describe_selection(store, source_path)
        except (ExperimentStoreError, OSError, ValueError) as exc:
            return render_template("ml_lab_ui/experiment_dataset.html", version=__version__,
                                   error=str(exc), preview=None, experiment=None,
                                   experiments=[], source_token=token, selected=None), 400

        identifier = (request.form if request.method == "POST" else request.args).get("experiment_id", "").strip()
        selected = None
        error = None
        try:
            experiment = store.get_experiment(identifier) if identifier else None
            if experiment is not None and experiment["metadata"].get("adapter") == "ml-lab.recorded-legacy-task@1":
                raise DatasetSelectionError("Legacy task experiments cannot be changed from Data Lab")
            if request.method == "POST":
                # Pin what the user saw, including the Data Lab provenance sidecar.
                if (request.form.get("source_sha256") != preview["sha256"]
                        or request.form.get("provenance_sha256", "") !=
                        ((preview["provenance"] or {}).get("sha256") or "")):
                    raise DatasetSelectionError("Dataset or provenance changed; reload the selection form")
                expected = None
                if experiment is not None:
                    raw = request.form.get("expected_revision", "")
                    if not raw.isdecimal() or int(raw) < 1:
                        raise DatasetSelectionError("Current experiment revision is required")
                    expected = int(raw)
                features = request.form.getlist("features")
                targets = request.form.getlist("targets")
                selected = select_experiment_dataset(
                    store, source_path, experiment_id=identifier or None,
                    experiment_name=request.form.get("experiment_name", "").strip() or None,
                    feature_columns=features, target_columns=targets,
                    expected_revision=expected,
                )
                experiment = selected
        except (ExperimentStoreError, OSError, ValueError) as exc:
            error = str(exc)
            selected = None
            # Show latest revision on stale POST; an invalid ID must produce a
            # controlled form error rather than a second uncaught lookup.
            try:
                experiment = store.get_experiment(identifier) if identifier else None
            except ExperimentStoreError:
                experiment = None

        current_ref = experiment["dataset_ref"] if experiment is not None else {}
        available_columns = set(preview["column_names"])
        previous_features = current_ref.get("feature_columns", [])
        previous_targets = current_ref.get("target_columns", [])
        default_targets = [col for col in previous_targets if col in available_columns]
        default_features = ([col for col in previous_features if col in available_columns]
                            if experiment is not None else list(preview["column_names"]))
        choices = [entry for entry in store.list_experiments(limit=100)
                   if entry["metadata"].get("adapter") != "ml-lab.recorded-legacy-task@1"]
        return render_template(
            "ml_lab_ui/experiment_dataset.html", version=__version__,
            error=error, preview=preview, source_token=token,
            experiment=experiment, experiments=choices, selected=selected,
            default_features=default_features, default_targets=default_targets,
            source_url=source_url(str(source_path)),
        ), (400 if error else 200)

    @blueprint.route("/experiments/models", methods=["GET", "POST"])
    def experiment_model():
        """S3.2 typed model workspace (configuration only; no training)."""
        from ml_lab.experiments import (
            ExperimentStore, ExperimentStoreError, ModelConfigurationError,
            configure_experiment_model, describe_model_method,
            list_model_families, list_model_methods, verify_selection,
        )
        from .model_forms import (
            ModelFormError, control_placeholder, model_form_values,
            read_model_form, submitted_form_values, visible_controls,
        )

        store = ExperimentStore(project_root)
        experiments = [row for row in store.list_experiments(limit=100)
                       if row["dataset_ref"].get("schema") == "ml-lab.experiment-dataset@1"
                       and row["metadata"].get("adapter") != "ml-lab.recorded-legacy-task@1"]
        params = request.form if request.method == "POST" else request.args
        identifier = params.get("experiment_id", "").strip()
        family = params.get("family", "").strip()
        method = params.get("method", "").strip()
        error = None
        data_error = None
        experiment = None
        saved = False
        families = list_model_families()
        if family not in {row["id"] for row in families}:
            if request.method == "POST":
                error = "Unsupported model family; reload the model workspace"
            family = "classification"  # Adjusted below for unsupervised datasets.
        try:
            if identifier:
                experiment = store.get_experiment(identifier)
                if not any(row["experiment_id"] == experiment["experiment_id"] for row in experiments):
                    raise ModelConfigurationError("Selected experiment has no Data Lab dataset")
            elif experiments and request.method == "GET":
                experiment = store.get_experiment(experiments[0]["experiment_id"])
            elif request.method == "POST":
                raise ModelFormError("Select an experiment before saving a model")
        except (ExperimentStoreError, ModelFormError) as exc:
            error = str(exc)

        if experiment and not params.get("family"):
            spec = experiment["model_spec"]
            if spec.get("schema") == "ml-lab.model-configuration@1":
                family = spec["family"]
            elif not experiment["dataset_ref"]["target_columns"]:
                family = "representation"
        methods = list_model_methods(family)
        valid_methods = {row["id"] for row in methods}
        if not method or method not in valid_methods:
            if request.method == "POST":
                error = "Unsupported model method; reload the model workspace"
            stored_spec = experiment["model_spec"] if experiment else {}
            if stored_spec.get("family") == family and stored_spec.get("method") in valid_methods:
                method = stored_spec["method"]
            else:
                method = next((row["id"] for row in methods if row["id"] != "suite"), methods[0]["id"])
        method_name = next(row["name"] for row in methods if row["id"] == method)
        schema = describe_model_method(family, method) if experiment else None

        if experiment is not None:
            try:
                verify_selection(store, experiment["dataset_ref"])
                targets = len(experiment["dataset_ref"]["target_columns"])
                if family in ("classification", "regression") and targets != 1:
                    data_error = f"{family.title()} requires exactly one target column. Update the selection in Data Lab."
                elif family in ("representation", "gan", "rbm") and targets:
                    data_error = f"{family.title()} requires feature-only data. Update the selection in Data Lab."
            except ExperimentStoreError as exc:
                data_error = str(exc)

        if request.method == "POST" and experiment is not None and error is None:
            try:
                if data_error:
                    raise ModelFormError(data_error)
                expected = params.get("expected_revision", "")
                if not expected.isdecimal() or int(expected) < 1:
                    raise ModelFormError("Expected experiment revision is required")
                settings = read_model_form(request.form, schema)
                configured = configure_experiment_model(
                    store, experiment["experiment_id"], family=family, method=method,
                    settings=settings, expected_revision=int(expected),
                )
                return redirect(url_for(
                    request.blueprint + ".experiment_model",
                    experiment_id=configured["experiment_id"], family=family, method=method,
                    saved="1",
                ))
            except (ExperimentStoreError, ModelFormError, ValueError) as exc:
                error = str(exc)
                # Always expose the latest revision in the selection summary.
                experiment = store.get_experiment(experiment["experiment_id"])
        from ml_lab.experiments import list_optimization_controls, optimization_metrics
        stored = experiment["model_spec"] if experiment else {}
        if request.method == "POST":
            values = submitted_form_values(request.form)
        elif stored.get("schema") == "ml-lab.model-configuration@1" and stored.get("family") == family and stored.get("method") == method:
            values = model_form_values(stored.get("settings"))
        else:
            values = {}
        saved = (request.method == "GET" and params.get("saved") == "1"
                 and stored.get("schema") == "ml-lab.model-configuration@1"
                 and stored.get("family") == family and stored.get("method") == method)
        return render_template(
            "ml_lab_ui/experiment_model.html", version=__version__,
            experiments=experiments, experiment=experiment, families=families,
            methods=methods, method=method, family=family, method_name=method_name,
            schema=schema, controls=visible_controls(schema) if schema else {},
            values=values, placeholder=control_placeholder,
            saved=saved, data_error=data_error, error=error,
            recent_runs=(store.list_runs(experiment["experiment_id"], limit=10) if experiment else []),
            optimization_controls=(list_optimization_controls(stored["family"], stored["method"])
                                   if stored.get("schema") == "ml-lab.model-configuration@1" else []),
            optimization_metrics=(optimization_metrics(stored["family"])
                                  if stored.get("schema") == "ml-lab.model-configuration@1" else []),
        ), (400 if error else 200)

    @blueprint.get("/experiments/synthetic")
    def experiment_synthetic():
        """Synthetic-data creation remains a separate operation from model training."""
        from ml_lab.experiments import ExperimentStore, ExperimentStoreError
        store = ExperimentStore(project_root)
        experiments = [row for row in store.list_experiments(limit=100)
                       if row["dataset_ref"].get("schema") == "ml-lab.experiment-dataset@1"]
        identifier = request.args.get("experiment_id", "").strip()
        try:
            experiment = (store.get_experiment(identifier) if identifier else
                          store.get_experiment(experiments[0]["experiment_id"]) if experiments else None)
            if experiment is not None and experiment["dataset_ref"].get("schema") != "ml-lab.experiment-dataset@1":
                raise ExperimentStoreError("Select an experiment with a Data Lab dataset")
        except ExperimentStoreError:
            abort(404)
        return render_template("ml_lab_ui/experiment_synthetic.html", version=__version__,
                               experiments=experiments, experiment=experiment)

    @blueprint.post("/experiments/synthetic/run")
    def experiment_synthetic_run():
        from ml_lab.experiments import ExperimentStoreError
        identifier = request.form.get("experiment_id", "").strip()
        try:
            revision = int(request.form.get("expected_revision", ""))
            plan = {"method": request.form.get("method", "").strip(),
                    "count": int(request.form.get("count", "")),
                    "seed": int(request.form.get("seed", "")),
                    "degrees_of_freedom": float(request.form.get("degrees_of_freedom", "5"))}
            run_id = training_jobs.start_synthetic(identifier, expected_revision=revision, plan=plan)
        except (ValueError, ExperimentStoreError) as exc:
            return render_template("ml_lab_ui/experiment_run_error.html", version=__version__,
                                   error=str(exc), experiment_id=identifier), 400
        return redirect(url_for(request.blueprint + ".experiment_model_run_view", run_id=run_id))

    @blueprint.get("/experiments/runs/<run_id>/synthetic.csv")
    def experiment_synthetic_download(run_id: str):
        from ml_lab.experiments import ExperimentStore, ExperimentStoreError, SYNTHETIC_EXECUTOR
        store = ExperimentStore(project_root)
        try:
            run = store.get_run(run_id)
            if run["status"] != "succeeded" or run["environment"].get("executor") != SYNTHETIC_EXECUTOR:
                abort(404)
            artifact = next((entry for entry in store.artifacts.list_for_run(run_id)
                             if entry["role"] == "output" and entry["kind"] == "dataset"), None)
            if artifact is None or store.artifacts.verify(artifact["artifact_id"])["status"] != "valid":
                abort(404)
        except ExperimentStoreError:
            abort(404)
        return send_file(store.project_root / artifact["path"], mimetype="text/csv",
                         as_attachment=True, download_name=f"ml_lab_synthetic_{run_id[:12]}.csv")

    @blueprint.post("/experiments/models/run")
    def experiment_model_run():
        """Start a pinned configured model run without blocking the UI request."""
        from ml_lab.experiments import ExperimentStoreError

        identifier = request.form.get("experiment_id", "").strip()
        revision_text = request.form.get("expected_revision", "")
        seed_text = request.form.get("seed", "").strip()
        try:
            if not revision_text.isdecimal() or int(revision_text) < 1:
                raise ExperimentStoreError("Choose a current saved experiment revision")
            seed = int(seed_text) if seed_text else None
            run_id = training_jobs.start(identifier, expected_revision=int(revision_text), seed=seed)
        except (ValueError, ExperimentStoreError) as exc:
            return render_template("ml_lab_ui/experiment_run_error.html", version=__version__,
                                   error=str(exc), experiment_id=identifier), 400
        return redirect(url_for(request.blueprint + ".experiment_model_run_view", run_id=run_id))

    @blueprint.post("/experiments/models/optimize")
    def experiment_model_optimize():
        from ml_lab.experiments import ExperimentStoreError
        identifier = request.form.get("experiment_id", "").strip()
        try:
            revision = int(request.form.get("expected_revision", ""))
            parameters = []
            for index in range(1, 5):
                suffix = "" if index == 1 else f"_{index}"
                path = request.form.get("parameter" + suffix, "").strip()
                lower = request.form.get("lower" + suffix, "").strip()
                upper = request.form.get("upper" + suffix, "").strip()
                if not path and index > 1:
                    if lower or upper:
                        raise ExperimentStoreError("Choose a control before entering its bounds")
                    continue
                parameters.append({"path": path, "lower": float(lower), "upper": float(upper),
                                   "scale": request.form.get("scale" + suffix, "linear")})
            plan = {
                "parameters": parameters,
                "algorithm": request.form.get("algorithm", "firefly"),
                "evaluations": int(request.form.get("evaluations", "8")),
                "metric": request.form.get("metric", "").strip(),
                "goal": request.form.get("goal", "minimize"),
                "population_size": int(request.form.get("population_size", "2")),
                "iterations": int(request.form.get("iterations", "1")),
                "seed": int(request.form.get("seed", "42")),
            }
            run_id = training_jobs.start_optimization(identifier, expected_revision=revision, plan=plan)
        except (ValueError, ExperimentStoreError) as exc:
            return render_template("ml_lab_ui/experiment_run_error.html", version=__version__,
                                   error=str(exc), experiment_id=identifier), 400
        return redirect(url_for(request.blueprint + ".experiment_model_run_view", run_id=run_id))

    @blueprint.post("/experiments/runs/<run_id>/cancel")
    def experiment_model_run_cancel(run_id: str):
        from ml_lab.experiments import (ExperimentStore, ExperimentStoreError,
                                        OPTIMIZER_EXECUTOR, request_run_cancellation)
        try:
            store = ExperimentStore(project_root)
            run = store.get_run(run_id)
            if run["environment"].get("executor") not in {"ml_lab.experiments.training@1", OPTIMIZER_EXECUTOR, "ml_lab.experiments.synthetic@1"}:
                abort(404)
            request_run_cancellation(store, run_id)
        except ExperimentStoreError:
            abort(404)
        return redirect(url_for(request.blueprint + ".experiment_model_run_view", run_id=run_id))

    @blueprint.get("/experiments/runs/<run_id>")
    def experiment_model_run_view(run_id: str):
        from ml_lab.experiments import ExperimentStore, ExperimentStoreError, get_configured_run_status, OPTIMIZER_EXECUTOR
        store = ExperimentStore(project_root)
        try:
            status = get_configured_run_status(store, run_id)
            if status["run"]["environment"].get("executor") not in {"ml_lab.experiments.training@1", OPTIMIZER_EXECUTOR, "ml_lab.experiments.synthetic@1"}:
                abort(404)
            experiment = store.get_experiment(status["run"]["experiment_id"], revision=status["run"]["revision"])
        except ExperimentStoreError:
            abort(404)
        synthetic_dataset_url = None
        synthetic_select_url = None
        if status["run"]["environment"].get("executor") == "ml_lab.experiments.synthetic@1" and status["run"]["status"] == "succeeded":
            dataset_artifact = next((item for item in status["artifacts"]
                                     if item["role"] == "output" and item["kind"] == "dataset"), None)
            if dataset_artifact and store.artifacts.verify(dataset_artifact["artifact_id"])["status"] == "valid":
                synthetic_dataset_url = url_for(request.blueprint + ".experiment_synthetic_download", run_id=run_id)
                synthetic_select_url = experiment_dataset_url(str(store.project_root / dataset_artifact["path"]))
        return render_template(
            "ml_lab_ui/experiment_run.html", version=__version__, status=status,
            synthetic_dataset_url=synthetic_dataset_url, synthetic_select_url=synthetic_select_url,
            experiment=experiment,
            status_url=url_for(request.blueprint + ".experiment_model_run_status", run_id=run_id),
            cancel_url=url_for(request.blueprint + ".experiment_model_run_cancel", run_id=run_id),
            results_url=url_for(request.blueprint + ".experiment_model_run_result", run_id=run_id),
        )

    @blueprint.get("/experiments/runs/<run_id>/status")
    def experiment_model_run_status(run_id: str):
        from ml_lab.experiments import ExperimentStore, ExperimentStoreError, get_configured_run_status, OPTIMIZER_EXECUTOR
        try:
            status = get_configured_run_status(ExperimentStore(project_root), run_id)
            if status["run"]["environment"].get("executor") not in {"ml_lab.experiments.training@1", OPTIMIZER_EXECUTOR, "ml_lab.experiments.synthetic@1"}:
                abort(404)
        except ExperimentStoreError:
            abort(404)
        return jsonify(status)

    @blueprint.get("/experiments/runs/<run_id>/result")
    def experiment_model_run_result(run_id: str):
        from ml_lab.experiments import ExperimentStore, ExperimentStoreError, get_configured_run_status, OPTIMIZER_EXECUTOR
        try:
            store = ExperimentStore(project_root)
            status = get_configured_run_status(store, run_id)
            if (status["run"]["environment"].get("executor") not in {"ml_lab.experiments.training@1", OPTIMIZER_EXECUTOR, "ml_lab.experiments.synthetic@1"}
                    or status["run"]["status"] != "succeeded"):
                abort(404)
            result = next((artifact for artifact in status["artifacts"]
                           if artifact["role"] == "output" and artifact["kind"] == "result"), None)
            if result is None or store.artifacts.verify(result["artifact_id"])["status"] != "valid":
                abort(404)
        except ExperimentStoreError:
            abort(404)
        return send_file(store.project_root / result["path"], mimetype="application/json")

    @blueprint.get("/experiments/compare")
    def experiment_compare():
        """S5.1 experiment-scoped, read-only run metrics comparison."""
        from ml_lab.experiments import (ExperimentStore, ExperimentStoreError,
                                        compare_experiment_runs, OPTIMIZER_EXECUTOR)
        store = ExperimentStore(project_root)
        experiment_id = request.args.get("experiment_id", "").strip()
        selected = request.args.getlist("run_id")
        try:
            report = compare_experiment_runs(store, experiment_id,
                                             run_ids=selected if selected else None)
            candidates = [row for row in store.list_runs(experiment_id, limit=100)
                          if row["environment"].get("executor") in
                          {"ml_lab.experiments.training@1", OPTIMIZER_EXECUTOR}][:25]
        except ExperimentStoreError:
            abort(404)
        from ml_lab.experiments import build_visualization_report
        from urllib.parse import urlencode
        # Explicitly pin the chart links to the selection shown in this page,
        # even if new runs are registered before the user opens an image.
        chosen = selected if selected else [r["run_id"] for r in report["runs"]]
        params = [("experiment_id", experiment_id)] + [("run_id", rid) for rid in chosen]
        query = urlencode(params)
        visualizations = build_visualization_report(store, experiment_id,
                                                     run_ids=chosen if chosen else None)
        charts = []
        for chart in visualizations["charts"]:
            charts.append({
                "data": chart,
                "svg_url": url_for(request.blueprint + ".experiment_chart_svg", chart_id=chart["id"]) + "?" + query,
                "csv_url": url_for(request.blueprint + ".experiment_chart_csv", chart_id=chart["id"]) + "?" + query,
            })
        return render_template(
            "ml_lab_ui/experiment_compare.html", version=__version__,
            report=report, candidates=candidates, charts=charts,
            visualization_notes=visualizations["notes"],
            selected_ids=set(chosen),
            csv_url=url_for(request.blueprint + ".experiment_compare_csv") + "?" + query,
            visualization_json_url=url_for(request.blueprint + ".experiment_visualizations_json") + "?" + query,
        )

    @blueprint.get("/experiments/compare.csv")
    def experiment_compare_csv():
        """Export exactly the same verified comparison selection as the page."""
        from ml_lab.experiments import (ExperimentStore, ExperimentStoreError,
                                        compare_experiment_runs, comparison_csv)
        selected = request.args.getlist("run_id")
        try:
            report = compare_experiment_runs(
                ExperimentStore(project_root), request.args.get("experiment_id", "").strip(),
                run_ids=selected if selected else None,
            )
        except ExperimentStoreError:
            abort(404)
        return Response(comparison_csv(report), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=ml_lab_run_comparison.csv",
                                 "Cache-Control": "no-store"})

    def _selected_visualization():
        """Resolve only experiment-scoped, registered artifacts; never paths."""
        from ml_lab.experiments import (ExperimentStore, ExperimentStoreError,
                                        build_visualization_report)
        selected = request.args.getlist("run_id")
        report = build_visualization_report(
            ExperimentStore(project_root), request.args.get("experiment_id", "").strip(),
            run_ids=selected if selected else None,
        )
        return report

    @blueprint.get("/experiments/visualizations.json")
    def experiment_visualizations_json():
        """Portable data: measured values, selected IDs and artifact hashes."""
        from ml_lab.experiments import ExperimentStoreError
        try:
            report = _selected_visualization()
        except ExperimentStoreError:
            abort(404)
        response = jsonify(report)
        response.headers["Content-Disposition"] = 'attachment; filename="ml_lab_visualizations.json"'
        response.headers["Cache-Control"] = "no-store"
        return response

    def _selected_chart(chart_id: str):
        from ml_lab.experiments import ExperimentStoreError
        if len(chart_id) > 140:
            abort(404)
        try:
            report = _selected_visualization()
        except ExperimentStoreError:
            abort(404)
        chart = next((item for item in report["charts"] if item["id"] == chart_id), None)
        if chart is None:
            abort(404)
        return chart

    @blueprint.get("/experiments/charts/<chart_id>.svg")
    def experiment_chart_svg(chart_id: str):
        from ml_lab.experiments import chart_svg
        chart = _selected_chart(chart_id)
        response = Response(chart_svg(chart), mimetype="image/svg+xml")
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = "default-src 'none'; style-src 'none'; sandbox"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @blueprint.get("/experiments/charts/<chart_id>.csv")
    def experiment_chart_csv(chart_id: str):
        from ml_lab.experiments import chart_csv
        chart = _selected_chart(chart_id)
        response = Response(chart_csv(chart), mimetype="text/csv")
        response.headers["Content-Disposition"] = 'attachment; filename="ml_lab_chart.csv"'
        response.headers["Cache-Control"] = "no-store"
        return response

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
            experiment_url=(experiment_dataset_url(str(path)) if record.format == "delimited_text"
                            and record.parse_status == "parsed" and record.has_header
                            and not record.malformed_row_count else None),
            data_url=url_for(request.blueprint + ".data_lab"),
            matrix_url=(url_for(request.blueprint + ".matrix_editor", token=token)
                        if record.format == "delimited_text" and record.parse_status == "parsed"
                        and record.has_header and not record.malformed_row_count else None),
        )

    @blueprint.get("/data/source/<token>/matrix")
    def matrix_editor(token: str):
        path = verify_path(token)
        record = data.inspect_file(path)
        if (record.format != "delimited_text" or record.parse_status != "parsed"
                or record.malformed_row_count or not record.has_header):
            abort(400)
        return render_template(
            "ml_lab_ui/matrix.html", version=__version__, source=record.to_record(),
            source_dtypes=data.describe_dataset(path)["dtypes"],
            rows_url=url_for(request.blueprint + ".source_rows", token=token),
            save_url=url_for(request.blueprint + ".save_matrix_draft", token=token),
            source_url=url_for(request.blueprint + ".source_view", token=token),
        )

    @blueprint.post("/data/source/<token>/matrix/save")
    def save_matrix_draft(token: str):
        path = verify_path(token)
        if request.content_length is not None and request.content_length > 2_000_000:
            return jsonify({"error": "Matrix draft request is too large"}), 413
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Expected JSON object"}), 400
        try:
            result = data.commit_matrix_draft(
                path, output_root=derived_output_root,
                expected_sha256=payload.get("source_sha256"),
                edits=payload.get("edits"),
            )
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400
        result["source_url"] = source_url(result["path"])
        result["provenance_url"] = provenance_url(result["path"])
        result["experiment_url"] = experiment_dataset_url(result["path"])
        return jsonify(result), 201

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
