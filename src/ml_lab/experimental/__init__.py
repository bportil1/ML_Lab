"""Incubator for research modules that are intentionally outside ML Lab's stable API.

Stable ML Lab modules may be imported by experiments. Stable modules must not
import from ``ml_lab.experimental``. Experimental implementations are loaded
lazily from manifest import paths only when explicitly requested.
"""

from .manifest import ExperimentalManifest, ExperimentalStatus
from .registry import (
    get_manifest,
    list_experiments,
    load_experiment,
    register_experiment,
    run_experiment,
    unregister_experiment,
)

__all__ = [
    "ExperimentalManifest",
    "ExperimentalStatus",
    "get_manifest",
    "list_experiments",
    "load_experiment",
    "register_experiment",
    "run_experiment",
    "unregister_experiment",
]
