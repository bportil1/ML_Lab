from __future__ import annotations

from typing import Any


def create_app(*, enable_experimental: bool = False, config: dict[str, Any] | None = None):
    """Create a standalone Flask application containing the same UI PAH can mount."""
    try:
        from flask import Flask
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("ML Lab UI requires: pip install 'ml-lab[ui]'") from exc

    from .blueprint import create_ui_blueprint

    app = Flask("ml_lab.ui")
    app.config.update({
        "JSON_SORT_KEYS": False,
        "ML_LAB_PROFILE_OUTPUT_ROOT": "ml_lab_results/data/profile_runs",
    })
    if config:
        app.config.update(config)
    app.register_blueprint(create_ui_blueprint(enable_experimental=enable_experimental, profile_output_root=app.config["ML_LAB_PROFILE_OUTPUT_ROOT"]), url_prefix="/")
    return app


def run_server(
    *,
    host: str = "127.0.0.1",
    port: int = 5055,
    debug: bool = False,
    enable_experimental: bool = False,
) -> None:
    app = create_app(enable_experimental=enable_experimental)
    app.run(host=host, port=port, debug=debug)
