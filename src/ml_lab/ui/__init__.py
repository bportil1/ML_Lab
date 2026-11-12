"""Optional first-party UI for ML Lab.

Importing :mod:`ml_lab.ui` is safe in headless/base installations. Flask is
loaded only when an application or Blueprint is explicitly requested.
"""

from __future__ import annotations

from .service import capability_groups, capability_records, get_capability


def create_ui_blueprint(
    name: str = "ml_lab_ui",
    *,
    enable_experimental: bool = False,
    profile_output_root: str = "analysis/data/profile_runs",
    derived_output_root: str = "analysis/data/derived",
    experiment_project_root: str | None = None,
    path_autocomplete_roots=None,
):
    from .blueprint import create_ui_blueprint as _create_ui_blueprint

    blueprint = _create_ui_blueprint(
        name=name,
        enable_experimental=enable_experimental,
        profile_output_root=profile_output_root,
        derived_output_root=derived_output_root,
        experiment_project_root=experiment_project_root,
        path_autocomplete_roots=path_autocomplete_roots,
    )
    # Keep workbench API wiring independent of the large, frequently edited
    # legacy blueprint. Registration happens before the host mounts it.
    from pathlib import Path
    from .workbench_routes import install_workbench_routes

    root = Path(experiment_project_root if experiment_project_root is not None else Path.cwd())
    install_workbench_routes(blueprint, name=name, project_root=root.expanduser().resolve())
    return blueprint


def create_app(*, enable_experimental: bool = False, config: dict | None = None):
    from .app import create_app as _create_app

    return _create_app(enable_experimental=enable_experimental, config=config)


__all__ = [
    "capability_groups",
    "capability_records",
    "create_app",
    "create_ui_blueprint",
    "get_capability",
]
