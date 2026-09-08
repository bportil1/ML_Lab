from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

from .catalog import BUILTIN_MANIFESTS
from .manifest import ExperimentalManifest

_MANIFESTS: dict[str, ExperimentalManifest] = {manifest.id: manifest for manifest in BUILTIN_MANIFESTS}


def register_experiment(manifest: ExperimentalManifest, *, replace: bool = False) -> ExperimentalManifest:
    """Register metadata without importing the experiment implementation."""
    manifest.validate()
    if manifest.id in _MANIFESTS and not replace:
        raise ValueError(f"experimental module {manifest.id!r} is already registered")
    _MANIFESTS[manifest.id] = manifest
    return manifest


def unregister_experiment(experiment_id: str) -> None:
    """Remove a dynamic registration. Primarily useful to hosts and tests."""
    _MANIFESTS.pop(experiment_id, None)


def list_experiments(
    *,
    status: str | None = None,
    capability: str | None = None,
) -> list[ExperimentalManifest]:
    manifests = list(_MANIFESTS.values())
    if status is not None:
        manifests = [manifest for manifest in manifests if manifest.status == status]
    if capability is not None:
        manifests = [manifest for manifest in manifests if capability in manifest.capabilities]
    return sorted(manifests, key=lambda manifest: manifest.id)


def get_manifest(experiment_id: str) -> ExperimentalManifest:
    try:
        return _MANIFESTS[experiment_id]
    except KeyError as exc:
        raise KeyError(f"unknown experimental module: {experiment_id}") from exc


def load_experiment(experiment_id: str) -> ModuleType:
    """Import an experiment only when the caller explicitly requests it."""
    manifest = get_manifest(experiment_id)
    return import_module(manifest.module_path)


def run_experiment(experiment_id: str, **kwargs: Any) -> Any:
    manifest = get_manifest(experiment_id)
    module = load_experiment(experiment_id)
    try:
        entrypoint = getattr(module, manifest.entrypoint)
    except AttributeError as exc:
        raise AttributeError(
            f"experimental module {experiment_id!r} does not expose entrypoint {manifest.entrypoint!r}"
        ) from exc
    if not callable(entrypoint):
        raise TypeError(
            f"experimental entrypoint {manifest.entrypoint!r} for {experiment_id!r} is not callable"
        )
    return entrypoint(**kwargs)
