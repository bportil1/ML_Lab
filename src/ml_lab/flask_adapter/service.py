"""Backward-compatible import surface for the host-neutral ML Lab application service.

The implementation moved to :mod:`ml_lab.application` in 0.13.0 so CLI, REST,
and optional UI callers share one non-Flask execution boundary.
"""

from ml_lab.application import PayloadError, execute_task

__all__ = ["PayloadError", "execute_task"]
