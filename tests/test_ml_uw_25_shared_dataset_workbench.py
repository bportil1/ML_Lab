"""ML-UW-25: generated datasets are current workbench data, not another wizard step."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader

from ml_lab.data.xml_collection import inventory_xml_collection
from ml_lab.data.xml_collection_recipe import build_xml_collection_recipe
from ml_lab.data.xml_similarity import compare_xml_collection
from ml_lab.experiments import (
    ExperimentStore, create_workbench_instance, select_workbench_dataset,
    WorkbenchConflict,
)

UI = Path(__file__).resolve().parents[1] / "src/ml_lab/ui"


def test_no_legacy_entry_or_browser_experiment_override():
    routes = (UI / "blueprint.py").read_text(encoding="utf-8")
    base = (UI / "templates/ml_lab_ui/base.html").read_text(encoding="utf-8")
    assert 'return redirect(url_for(request.blueprint + ".data_lab"))' in routes
    assert 'render_template("ml_lab_ui/workspace.html"' not in routes
    assert 'render_template("ml_lab_ui/index.html"' not in routes
    assert "filename='ml_lab_workspace.js'" not in base
    assert "filename='ml_lab_workbench_context.js'" in base
    assert 'mlwb-primary-nav' in base


def test_derived_selection_remains_instance_scoped_and_source_verified(tmp_path):
    source = tmp_path / "analysis/data/derived/combined.csv"
    source.parent.mkdir(parents=True)
    source.write_text("value,score\n1,5\n2,7\n", encoding="utf-8")
    store = ExperimentStore(tmp_path)
    first = create_workbench_instance(store)
    second = create_workbench_instance(store)
    selected = select_workbench_dataset(store, first["instance_id"], source,
                                        expected_generation=0)
    assert selected["active_dataset"]["rows"] == 2
    assert selected["active_dataset"]["path"] == "analysis/data/derived/combined.csv"
    assert second["status"] == "empty"
    with pytest.raises(WorkbenchConflict):
        select_workbench_dataset(store, first["instance_id"], source,
                                 expected_generation=0)
    assert not store.list_runs(selected["experiment_id"])


def test_updated_templates_compile_with_activation_status():
    env = Environment(loader=FileSystemLoader(UI / "templates"), autoescape=True)
    for template in ("base.html", "xml_structure.html", "xml_collection_export.html",
                     "xml_collection_provenance.html", "transform.html",
                     "experiment_dataset.html"):
        env.get_template("ml_lab_ui/" + template)
    route = (UI / "blueprint.py").read_text()
    assert "workbench_activated=activated" in route
    assert "workbench_activation_error=activation_error" in route
    assert "starting_generation" in route
    assert "new_manifest" in route


def _client(root: Path):
    flask = pytest.importorskip("flask")
    from ml_lab.ui import create_ui_blueprint
    app = flask.Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(create_ui_blueprint(name="uw25", experiment_project_root=str(root)),
                           url_prefix="/tools/ml-lab")
    return app.test_client()


def test_xml_collection_export_auto_activates_across_all_five_sections(tmp_path):
    first = tmp_path / "a.xml"
    second = tmp_path / "b.xml"
    first.write_text('<rows><item><value>2</value></item></rows>', encoding="utf-8")
    second.write_text('<rows><item><value>3</value></item></rows>', encoding="utf-8")
    inventory = inventory_xml_collection(tmp_path)
    recipe = build_xml_collection_recipe(
        inventory, compare_xml_collection(inventory), ['a.xml', 'b.xml'],
        '/rows/item', ['text:/rows/item/value'],
    )
    # Keep an older saved dataset to prove General ML Tools does not default
    # to a different experiment just because it appears in the catalog.
    prior_source = tmp_path / 'prior.csv'
    prior_source.write_text('feature\n7\n8\n', encoding='utf-8')
    from ml_lab.experiments import select_experiment_dataset
    prior = select_experiment_dataset(ExperimentStore(tmp_path), prior_source)
    client = _client(tmp_path)
    start = client.get('/tools/ml-lab/')
    assert start.status_code == 302
    assert start.headers['Location'] == '/tools/ml-lab/data'
    assert b'Machine Learning Lab' not in client.get(start.headers['Location']).data
    empty = client.get('/tools/ml-lab/api/workbench/context').get_json()
    assert empty['status'] == 'empty'
    output = tmp_path / 'analysis/data/derived/combined.csv'
    form = {'recipe_json': json.dumps(recipe), 'mode': 'preview', 'output': str(output)}
    pre = client.post('/tools/ml-lab/data/xml/collection/export', data=form)
    assert pre.status_code == 200
    assert client.get('/tools/ml-lab/api/workbench/context').get_json()['generation'] == 0
    assert not output.exists()
    written = client.post('/tools/ml-lab/data/xml/collection/export',
                          data={**form, 'mode': 'export', 'confirm': '1'})
    assert written.status_code == 200
    assert b'Active dataset updated' in written.data
    ctx = client.get('/tools/ml-lab/api/workbench/context').get_json()
    assert ctx['status'] == 'ready'
    assert ctx['active_dataset']['rows'] == 2
    assert 'Comments' not in ctx['active_dataset']['feature_columns']
    assert ctx['generation'] == 1
    for section in ('/data', '/workbench/feature-profiling',
                    '/workbench/representation', '/workbench/comparison',
                    '/workbench/general-ml-tools'):
        page = client.get('/tools/ml-lab' + section)
        assert page.status_code == 200, section
        assert b'No verified active dataset' not in page.data, section
    assert not ExperimentStore(tmp_path).list_runs(ctx['experiment_id'])
    assert ctx['experiment_id'] != prior['experiment_id']
    model_page = client.get('/tools/ml-lab/experiments/models')
    assert model_page.status_code == 200
    assert f'value="{ctx["experiment_id"]}" selected'.encode() in model_page.data
    synthetic_page = client.get('/tools/ml-lab/experiments/synthetic')
    assert synthetic_page.status_code == 200
    assert f'value="{ctx["experiment_id"]}" selected'.encode() in synthetic_page.data


def test_outside_project_materialization_does_not_switch_existing_dataset(tmp_path):
    project = tmp_path / 'project'
    project.mkdir()
    source = project / 'input.xml'
    source.write_text('<catalog><row><value>10</value></row></catalog>', encoding='utf-8')
    client = _client(project)
    initial = client.get('/tools/ml-lab/api/workbench/context').get_json()
    opened = client.post('/tools/ml-lab/data/xml/open', data={'source': str(source)})
    assert opened.status_code == 302
    import html, re
    markup = client.get(opened.headers['Location']).get_data(as_text=True)
    def attribute(key):
        return html.unescape(re.search(rf'{key}="([^"]+)"', markup).group(1))
    fields_url = attribute('data-record-fields-url')
    preview_url = attribute('data-preview-url')
    materialize_url = attribute('data-materialize-url')
    fields = client.get(fields_url, query_string={'root': '/catalog/row'}).get_json()['fields']
    field_ids = [field['field_id'] for field in fields if field['relative_path'] == 'value']
    preview = client.post(preview_url, json={'record_root_canonical_path': '/catalog/row',
                                            'selected_field_ids': field_ids, 'rules': []})
    payload = {'record_root_canonical_path': '/catalog/row',
               'selected_field_ids': field_ids, 'rules': [],
               'confirmed_preview_signature': preview.get_json()['preview_signature']}
    outside = client.post(materialize_url, json={**payload, 'output': str(tmp_path / 'other.csv')})
    assert outside.status_code == 200
    assert outside.get_json()['workbench_activated'] is False
    assert 'outside' in (outside.get_json()['workbench_activation_error'] or '').lower()
    after = client.get('/tools/ml-lab/api/workbench/context').get_json()
    assert after['generation'] == initial['generation'] == 0
    assert after['status'] == 'empty'
