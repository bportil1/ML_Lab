"""P5-13 navigation contracts work in the headless/base installation."""
from pathlib import Path

from ml_lab.ui.workspace import active_tool, project_scope, tool_records


def test_scientific_tools_are_free_navigation_and_have_explicit_entrypoints():
    tools = tool_records()
    assert [tool["id"] for tool in tools] == [
        "data", "models", "representation", "ebm", "optimization",
        "generative", "xml", "results",
    ]
    assert len({tool["id"] for tool in tools}) == len(tools)
    assert all(tool["endpoint"] for tool in tools)
    assert all("requires" not in tool and "completed" not in tool for tool in tools)
    assert next(tool for tool in tools if tool["id"] == "ebm")["query"] == {"family": "rbm"}
    assert next(tool for tool in tools if tool["id"] == "representation")["query"] == {"family": "representation"}
    assert next(tool for tool in tools if tool["id"] == "optimization")["anchor"] == "model-optimization"


def test_project_scopes_are_stable_but_isolated(tmp_path: Path):
    a = tmp_path / "alpha"
    b = tmp_path / "beta"
    a.mkdir()
    b.mkdir()
    token = project_scope(a, "ml_lab_ui")
    assert token == project_scope(a, "ml_lab_ui")
    assert token != project_scope(b, "ml_lab_ui")
    assert token != project_scope(a, "hosted_ml_lab")
    assert str(a) not in token
    assert len(token) == 24


def test_active_tool_fallback_preserves_existing_workflows():
    assert active_tool("experiment_model", "rbm") == "ebm"
    assert active_tool("experiment_model", "representation") == "representation"
    assert active_tool("experiment_model") == "models"
    assert active_tool("workspace_results") == "results"
    assert active_tool("data_transform") == "data"
    assert active_tool("xml_collection_inventory") == "xml"
    assert active_tool("index") == ""


def test_ui_uses_existing_routes_and_no_new_science_implementations():
    base = Path(__file__).resolve().parents[1] / "src" / "ml_lab" / "ui"
    html = (base / "templates" / "ml_lab_ui" / "base.html").read_text()
    script = (base / "static" / "ml_lab_workspace.js").read_text()
    model = (base / "templates" / "ml_lab_ui" / "experiment_model.html").read_text()
    assert 'data-workspace-tool="{{ tool.id }}"' in html
    assert 'data-workspace-scope="{{ workspace_scope }}"' in html
    assert '"ml-lab.workspace.v1." + root.dataset.workspaceScope' in script
    assert 'localStorage.setItem(key, JSON.stringify(state))' in script
    assert 'data-workspace-current-run-id' in (base / "templates" / "ml_lab_ui" / "experiment_run.html").read_text()
    assert 'id="model-optimization"' in model
    assert '1 · Data selected' not in model
    assert '2 · Model configuration' not in model
