from __future__ import annotations

import sys

import pytest

from ml_lab.experimental import (
    ExperimentalManifest,
    get_manifest,
    list_experiments,
    load_experiment,
    register_experiment,
    run_experiment,
    unregister_experiment,
)


def test_experimental_manifest_registry_is_lazy(tmp_path, monkeypatch):
    module_name = "ml_lab_test_lazy_experiment"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        "IMPORTED = True\n"
        "def run(value=0):\n"
        "    return {'value': value + 1}\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(module_name, None)

    manifest = ExperimentalManifest(
        id="lazy_example",
        name="Lazy Example",
        module_path=module_name,
        capabilities=("demo", "testing"),
    )
    register_experiment(manifest, replace=True)
    try:
        assert module_name not in sys.modules
        assert get_manifest("lazy_example") == manifest
        assert [item.id for item in list_experiments(capability="demo")] == ["lazy_example"]
        assert module_name not in sys.modules

        module = load_experiment("lazy_example")
        assert module.IMPORTED is True
        assert run_experiment("lazy_example", value=4) == {"value": 5}
    finally:
        unregister_experiment("lazy_example")
        sys.modules.pop(module_name, None)


def test_experimental_manifest_validation_and_duplicate_protection():
    with pytest.raises(ValueError):
        register_experiment(ExperimentalManifest(id="Bad-ID", name="bad", module_path="x"))

    manifest = ExperimentalManifest(id="duplicate_example", name="Duplicate", module_path="json")
    register_experiment(manifest, replace=True)
    try:
        with pytest.raises(ValueError):
            register_experiment(manifest)
    finally:
        unregister_experiment("duplicate_example")
