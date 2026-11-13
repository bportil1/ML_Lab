"""ML-UW-16B: Data remains a primary tab; its three children are nested below all five."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader

UI = Path(__file__).resolve().parents[1] / "src/ml_lab/ui"


def test_five_primary_tabs_three_separate_data_children():
    source = (UI / "workbench_navigation.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    top = {node.targets[0].id: ast.literal_eval(node.value)
           for node in tree.body if isinstance(node, ast.Assign)
           and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
           and node.targets[0].id in {"SECTION_SPEC", "DATA_SUBSECTIONS"}}
    assert [key for key, _, _ in top["SECTION_SPEC"]] == [
        "data", "profiling", "representation", "comparison", "tools"]
    assert [label for _, label, _ in top["DATA_SUBSECTIONS"]] == [
        "Tabular Data", "XML Data", "Post-Collection Utilities"]
    assert {key for key, _, _ in top["DATA_SUBSECTIONS"]}.isdisjoint(
        {key for key, _, _ in top["SECTION_SPEC"]})
    template = (UI / "templates/ml_lab_ui/base.html").read_text(encoding="utf-8")
    assert template.index("{% endfor %}\n        {% else %}") < template.index('class="mlwb-secondary-nav mlwb-data-subsections"')
    assert 'aria-label="Data subsections"' in template
    assert '{% if data_subsections %}' in template
    stylesheet = (UI / "static/ml_lab_workbench.css").read_text(encoding="utf-8")
    assert ".mlwb-data-subsections a" in stylesheet
    assert "color: var(--pah-muted)" in stylesheet
    Environment(loader=FileSystemLoader(UI / "templates")).get_template("ml_lab_ui/base.html")


def test_data_parent_child_route_classification(tmp_path):
    pytest.importorskip("flask")
    from ml_lab.ui.workbench_navigation import section_for_request, data_subsection_for_request
    cases = (
        ("app.data_lab", "tabular"),
        ("app.source_view", "tabular"),
        ("app.xml_workbench", "xml"),
        ("app.xml_collection_inventory", "xml"),
        ("app.xml_collection_recipe", "xml"),
        ("app.xml_collection_export", "xml"),
        ("app.data_transform", "postcollection"),
        ("app.data_provenance", "postcollection"),
        ("app.workbench_postcollection", "postcollection"),
        ("app.experiment_dataset", "postcollection"),
    )
    for route, child in cases:
        assert section_for_request(route, None) == "data"
        assert data_subsection_for_request(route) == child
    for route in ("app.index", "app.workspace", "app.workbench_representation",
                  "app.workbench_feature_profiling", "app.workbench_general_ml_tools"):
        assert section_for_request(route, None) != "data"


def test_nested_sidebar_only_on_data_routes(tmp_path):
    flask = pytest.importorskip("flask")
    from ml_lab.ui import create_ui_blueprint
    app = flask.Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(create_ui_blueprint(name="mluw16b", experiment_project_root=str(tmp_path)),
                           url_prefix="/pah/ml")
    client = app.test_client()
    for route, active_child in (
        ("/data", "Tabular Data"),
        ("/data/xml/workbench", "XML Data"),
        ("/workbench/postcollection", "Post-Collection Utilities"),
        ("/data/xml/collection", "XML Data"),
    ):
        response = client.get("/pah/ml" + route)
        assert response.status_code == 200, route
        html = response.get_data(as_text=True)
        assert html.count('class="mlwb-primary-link') == 5
        assert html.count('class="mlwb-secondary-nav mlwb-data-subsections"') == 1
        primary = html.index('class="mlwb-primary-link')
        general = html.index('>General ML Tools</a>', primary)
        nested = html.index('class="mlwb-secondary-nav mlwb-data-subsections"')
        assert general < nested
        for label in ("Tabular Data", "XML Data", "Post-Collection Utilities"):
            assert html.index('>' + label + '</a>', nested) > nested
        assert f'>{active_child}</a>' in html[nested:]
        assert '/pah/ml/data/xml/workbench' in html
        assert '/pah/ml/workbench/postcollection' in html
    for route in ("/", "/workbench/feature-profiling", "/workbench/representation",
                  "/workbench/comparison", "/workbench/general-ml-tools"):
        response = client.get('/pah/ml' + route)
        assert response.status_code == 200, route
        html = response.get_data(as_text=True)
        assert html.count('class="mlwb-primary-link') == 5
        assert 'class="mlwb-secondary-nav mlwb-data-subsections"' not in html
