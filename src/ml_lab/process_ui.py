"""Independent local ML Lab UI process, launched through a generic supervisor.

This entry point has no PAH package dependencies and works only from an
explicit process environment. It does not expose scientific project internals
or process management APIs to the host.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Mapping


def configuration_from_environment(environment: Mapping[str, str]) -> dict[str, object]:
    """Resolve project-level persistence and instance-local generated outputs."""
    instance_id = environment.get("PAH_MODULE_INSTANCE_ID", "").strip()
    runtime_root = environment.get("PAH_MODULE_RUNTIME_DIR", "").strip()
    if not instance_id or not runtime_root:
        raise ValueError("A PAH module instance identity and runtime directory are required")

    runtime_path = Path(runtime_root).expanduser().resolve()
    if not runtime_path.is_dir():
        raise ValueError(f"Instance runtime directory does not exist: {runtime_path}")

    selected_project = environment.get("ML_LAB_PROJECT_ROOT", "").strip()
    project = (Path(selected_project).expanduser().resolve()
               if selected_project else runtime_path)
    if not project.is_dir():
        raise ValueError(f"ML Lab project directory does not exist: {project}")

    results_setting = environment.get("ML_LAB_RESULTS_ROOT", "").strip()
    results = (Path(results_setting).expanduser().resolve()
               if results_setting and selected_project else project / "analysis")
    # Instance IDs contain colons and other runtime identifiers. Hash rather
    # than using their raw bytes in a path component or trusting separators.
    instance_folder = hashlib.sha256(instance_id.encode("utf-8")).hexdigest()[:24]
    instance_outputs = results / "instances" / instance_folder

    return {
        "ML_LAB_EXPERIMENT_PROJECT_ROOT": str(project),
        "ML_LAB_PROFILE_OUTPUT_ROOT": str(instance_outputs / "data" / "profile_runs"),
        "ML_LAB_DERIVED_OUTPUT_ROOT": str(instance_outputs / "data" / "derived"),
        "ML_LAB_PATH_AUTOCOMPLETE_ROOTS": [str(project)],
        "ML_LAB_WORKBENCH_COOKIE_SCOPE": instance_id,
    }


def main() -> int:
    environment = os.environ
    port_string = environment.get("PAH_MODULE_PORT", "")
    try:
        port = int(port_string)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("PAH_MODULE_PORT must be an integer") from exc
    if not 1 <= port <= 65535:
        raise RuntimeError("PAH_MODULE_PORT must be between 1 and 65535")

    from ml_lab.ui import create_app

    app = create_app(config=configuration_from_environment(environment))
    # PAH's instance UI only advertises loopback addresses. Disable the Flask
    # development reloader: it would spawn an unmanaged second process.
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
