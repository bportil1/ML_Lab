"""ML-UW-11: XML collection return paths and explicit single-file dataset handoff.

The optional Flask tests perform real local XML parsing and extraction while
headless checks protect the template/JS contract and cross-mode dataset safety.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader

from ml_lab.data.xml_collection import inventory_xml_collection
from ml_lab.data.xml_collection_recipe import build_xml_collection_recipe
from ml_lab.data.xml_collection_export import materialize_xml_collection
from ml_lab.data.xml_similarity import compare_xml_collection
from ml_lab.experiments import ExperimentStore, create_workbench_instance, select_workbench_dataset


TEMPLATES = Path(__file__).resolve().parents[1] / 'src/ml_lab/ui/templates'
UI_SRC = Path(__file__).resolve().parents[1] / 'src/ml_lab/ui'


def test_single_and_collection_extractions_remain_eligible_for_verified_selection(tmp_path):
    """The new UI handoff must not replace existing extraction/provenance engines."""
    xml_a = tmp_path / 'a.xml'
    xml_b = tmp_path / 'b.xml'
    xml_a.write_text('<rows><item><value>2</value></item></rows>')
    xml_b.write_text('<rows><item><value>3</value></item></rows>')
    inventory = inventory_xml_collection(tmp_path)
    recipe = build_xml_collection_recipe(inventory, compare_xml_collection(inventory),
                                         ['a.xml', 'b.xml'], '/rows/item', ['text:/rows/item/value'])
    output = tmp_path / 'analysis/data/combined.csv'
    materialize_xml_collection(recipe, output, confirm=True)
    store = ExperimentStore(tmp_path)
    instance = create_workbench_instance(store)
    selected = select_workbench_dataset(store, instance['instance_id'], output,
                                        expected_generation=instance['generation'])
    assert selected['active_dataset']['rows'] == 2
    assert selected['active_dataset']['provenance']['event_id'].startswith('event:sha256:')


def test_xml_ui_templates_and_javascript_have_explicit_nonautomatic_handoff():
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=True)
    for name in ('xml_structure', 'xml_collection_inventory', 'xml_collection_review',
                 'xml_collection_recipe', 'xml_collection_export'):
        env.get_template(f'ml_lab_ui/{name}.html')
    structure = (TEMPLATES / 'ml_lab_ui/xml_structure.html').read_text()
    collection = (TEMPLATES / 'ml_lab_ui/xml_collection_inventory.html').read_text()
    review = (TEMPLATES / 'ml_lab_ui/xml_collection_review.html').read_text()
    js = (UI_SRC / 'static/ml_lab_xml_structure.js').read_text()
    route = (UI_SRC / 'blueprint.py').read_text()
    assert 'data-xml-materialize-workbench' in structure
    assert 'workbench_select_url' in js
    assert 'response["workbench_select_url"] = workbench_dataset_url(result.dataset_path)' in route
    assert 'experiment_dataset_url(path) is None' in route
    assert 'Prepare full-group extraction recipe' in collection
    assert 'Prepare full-group extraction recipe' in review
    assert 'minimum_similarity=min_shared_fraction' in review
    assert 'request.method == "POST"' in route


def _ui_client(root):
    flask = pytest.importorskip('flask')
    from ml_lab.ui import create_ui_blueprint
    app = flask.Flask(__name__)
    app.config['TESTING'] = True
    app.register_blueprint(create_ui_blueprint(name='ml_uw_11',
                                               experiment_project_root=str(root)),
                           url_prefix='/pah/ml-lab')
    return app.test_client()


def test_collection_review_back_to_inventory_preserves_filters_without_inspection(tmp_path, monkeypatch):
    client = _ui_client(tmp_path)
    import ml_lab.data as data

    def fail_if_scanned(*args, **kwargs):
        raise AssertionError('GET must not scan XML files')

    monkeypatch.setattr(data, 'inventory_xml_collection', fail_if_scanned)
    values = {'directory': str(tmp_path), 'recursive': '0', 'include_hidden': '1',
              'include_glob': 'station-*.xml', 'exclude_glob': '*.old.xml',
              'max_files': '37', 'compare': '1', 'minimum_similarity': '0.65'}
    page = client.get('/pah/ml-lab/data/xml/collection', query_string=values)
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    assert 'station-*.xml' in html and '*.old.xml' in html
    assert 'value="37"' in html
    assert 'data/xml/' in html
    page = client.get('/pah/ml-lab/data/xml/collection/review', query_string={
        **values, 'min_shared_fraction': '0.65',
    })
    assert page.status_code == 200
    assert '← XML collection inventory (retain filters)' in page.get_data(as_text=True)
    assert 'station-*.xml' in page.get_data(as_text=True)


def test_single_xml_materialization_explicit_project_scoped_workbench_link(tmp_path):
    import re
    import html

    project = tmp_path / 'project'
    project.mkdir()
    source = project / 'input.xml'
    source.write_text('<catalog><row><value>A</value></row><row><value>B</value></row></catalog>')
    client = _ui_client(project)
    opened = client.post('/pah/ml-lab/data/xml/open', data={'source': str(source)})
    assert opened.status_code == 302
    explorer = client.get(opened.headers['Location'])
    assert explorer.status_code == 200
    markup = explorer.get_data(as_text=True)
    def attr(field):
        return html.unescape(re.search(rf'{field}="([^"]+)"', markup).group(1))
    fields_url = attr('data-record-fields-url')
    preview_url = attr('data-preview-url')
    materialize_url = attr('data-materialize-url')
    fields = client.get(fields_url, query_string={'root': '/catalog/row'}).get_json()['fields']
    selected = [field['field_id'] for field in fields if field['relative_path'] == 'value']
    assert selected
    preview = client.post(preview_url, json={
        'record_root_canonical_path': '/catalog/row', 'selected_field_ids': selected,
        'rules': [], 'max_rows': 10,
    })
    assert preview.status_code == 200
    payload = {'record_root_canonical_path': '/catalog/row',
               'selected_field_ids': selected, 'rules': [], 'overwrite': False,
               'confirmed_preview_signature': preview.get_json()['preview_signature']}
    project_output = project/'analysis/data/derived/result.csv'
    inside = client.post(materialize_url, json={**payload, 'output': str(project_output)})
    assert inside.status_code == 200
    link = inside.get_json()['workbench_select_url']
    assert link.startswith('/pah/ml-lab/workbench/dataset/select-file?')
    assert client.get(link).status_code == 200
    assert client.get('/pah/ml-lab/api/workbench/context').get_json()['status'] == 'empty'
    assert not ExperimentStore(project).list_experiments()

    outside_output = tmp_path / 'outside.csv'
    outside = client.post(materialize_url, json={**payload, 'output': str(outside_output)})
    assert outside.status_code == 200
    assert outside.get_json()['workbench_select_url'] is None
    assert outside.get_json()['transform_url'].startswith('/pah/ml-lab/')
