from __future__ import annotations

from pathlib import Path
import sys

import pytest

from ml_lab.pah_integration import MLLabRuntimeAdapter
from ml_lab.process_ui import configuration_from_environment


def test_adapter_opt_in_uses_only_generic_process_contract(tmp_path: Path):
    adapter = MLLabRuntimeAdapter()
    spec = adapter.process_launch_spec(context={
        "project_root": tmp_path,
        "results_root": tmp_path / "ml_lab_results",
    })
    assert spec == {
        "command": [sys.executable, "-m", "ml_lab.process_ui"],
        "env": {
            "ML_LAB_PROJECT_ROOT": str(tmp_path.resolve()),
            "ML_LAB_RESULTS_ROOT": str((tmp_path / "analysis").resolve()),
        },
        "requested_port": 0,
        "ui_path": "/",
    }
    assert "cwd" not in spec


def test_unbound_launch_does_not_use_host_cwd_for_scientific_outputs():
    spec = MLLabRuntimeAdapter().process_launch_spec(context=None)
    assert spec["env"] == {}
    assert spec["command"][1:] == ["-m", "ml_lab.process_ui"]


def test_sibling_instances_have_independent_default_outputs_and_cookie_scopes(tmp_path: Path):
    runtime = tmp_path / "runtime"
    project = tmp_path / "project"
    runtime.mkdir()
    project.mkdir()
    baseline = {
        "PAH_MODULE_RUNTIME_DIR": str(runtime),
        "ML_LAB_PROJECT_ROOT": str(project),
        "ML_LAB_RESULTS_ROOT": str(project / "analysis"),
    }
    first = configuration_from_environment({**baseline, "PAH_MODULE_INSTANCE_ID": "host:ml_lab:one"})
    second = configuration_from_environment({**baseline, "PAH_MODULE_INSTANCE_ID": "host:ml_lab:two"})
    assert first["ML_LAB_EXPERIMENT_PROJECT_ROOT"] == second["ML_LAB_EXPERIMENT_PROJECT_ROOT"]
    assert first["ML_LAB_DERIVED_OUTPUT_ROOT"] != second["ML_LAB_DERIVED_OUTPUT_ROOT"]
    assert first["ML_LAB_PROFILE_OUTPUT_ROOT"] != second["ML_LAB_PROFILE_OUTPUT_ROOT"]
    assert first["ML_LAB_WORKBENCH_COOKIE_SCOPE"] != second["ML_LAB_WORKBENCH_COOKIE_SCOPE"]
    for configured in (first, second):
        assert Path(configured["ML_LAB_DERIVED_OUTPUT_ROOT"]).is_relative_to(project / "analysis")


def test_without_project_outputs_live_inside_runtime(tmp_path: Path):
    config = configuration_from_environment({
        "PAH_MODULE_RUNTIME_DIR": str(tmp_path),
        "PAH_MODULE_INSTANCE_ID": "host:ml_lab:one",
    })
    assert config["ML_LAB_EXPERIMENT_PROJECT_ROOT"] == str(tmp_path)
    assert Path(config["ML_LAB_PROFILE_OUTPUT_ROOT"]).is_relative_to(tmp_path)


@pytest.mark.parametrize("environment", [
    {},
    {"PAH_MODULE_INSTANCE_ID": "ok"},
    {"PAH_MODULE_INSTANCE_ID": "ok", "PAH_MODULE_RUNTIME_DIR": "/does-not-exist-for-ml-lab-tests"},
])
def test_missing_instance_contract_rejected(environment):
    with pytest.raises(ValueError):
        configuration_from_environment(environment)


def test_workbench_cookies_are_distinct_for_same_project_across_ports(tmp_path: Path):
    pytest.importorskip("flask")
    from ml_lab.ui import create_app

    first = create_app(config={
        "TESTING": True,
        "ML_LAB_EXPERIMENT_PROJECT_ROOT": str(tmp_path),
        "ML_LAB_WORKBENCH_COOKIE_SCOPE": "a:ml_lab:one",
    })
    second = create_app(config={
        "TESTING": True,
        "ML_LAB_EXPERIMENT_PROJECT_ROOT": str(tmp_path),
        "ML_LAB_WORKBENCH_COOKIE_SCOPE": "a:ml_lab:two",
    })
    with first.test_client() as one, second.test_client() as two:
        one.get("/api/workbench/context")
        two.get("/api/workbench/context")
        headers_a = one.get("/api/workbench/context").headers.getlist("Set-Cookie")
        headers_b = two.get("/api/workbench/context").headers.getlist("Set-Cookie")
    assert any("ml_lab_workbench_" in header for header in headers_a)
    assert any("ml_lab_workbench_" in header for header in headers_b)
    assert headers_a != headers_b


def test_child_launch_binds_only_loopback_without_reloader(tmp_path: Path, monkeypatch):
    from ml_lab import process_ui, ui

    launch = {
        "PAH_MODULE_RUNTIME_DIR": str(tmp_path),
        "PAH_MODULE_INSTANCE_ID": "host:ml_lab:one",
        "PAH_MODULE_PORT": "39321",
    }
    for key, value in launch.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("ML_LAB_PROJECT_ROOT", raising=False)
    monkeypatch.delenv("ML_LAB_RESULTS_ROOT", raising=False)

    called = []

    class FakeApp:
        def run(self, **kwargs):
            called.append(kwargs)

    def fake_create_app(*, config):
        assert config["ML_LAB_WORKBENCH_COOKIE_SCOPE"] == launch["PAH_MODULE_INSTANCE_ID"]
        return FakeApp()

    monkeypatch.setattr(ui, "create_app", fake_create_app)
    assert process_ui.main() == 0
    assert called == [{"host": "127.0.0.1", "port": 39321,
                       "debug": False, "use_reloader": False}]
