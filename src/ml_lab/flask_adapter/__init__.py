"""Optional Flask/PAH adapter for ML Lab.

Importing this package does not require Flask. Flask is imported only when a
Blueprint is requested, keeping the ML engine usable in headless environments.
"""

from __future__ import annotations

from .service import PayloadError, execute_task


def create_blueprint(name: str = "ml_lab", *, enable_experimental: bool = False):
    try:
        from .blueprint import create_blueprint as _create_blueprint
    except RuntimeError:
        raise
    return _create_blueprint(name=name, enable_experimental=enable_experimental)


__all__ = ["PayloadError", "create_blueprint", "execute_task"]
