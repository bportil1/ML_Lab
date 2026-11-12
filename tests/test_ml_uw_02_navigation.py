"""ML-UW-02: regression coverage for five-section workbench navigation."""
from __future__ import annotations

import pytest

flask = pytest.importorskip('flask')
from flask import Flask

from ml_lab.ui import create_ui_blueprint
from ml_lab.ui.workbench_navigation import section_for_request


def _client(tmp_path, prefix=''):
    app = Flask(__name__)
    app.config.update(TESTING=True)
    app.register_blueprint(
        create_ui_blueprint(name='ml_uw02_test', experiment_project_root=str(tmp_path)),
        url_prefix=prefix or '/',
    )
    return app.test_client()


@pytest.mark.parametrize('route, title', [
    ('/workbench/feature-profiling', 'Feature Profiling'),
    ('/workbench/representation', 'Representation'),
    ('/workbench/comparison', 'Comparison'),
    ('/workbench/general-ml-tools', 'General ML Tools'),
])
def test_new_sections_render_and_mark_planned_honestly(tmp_path, route, title):
    client = _client(tmp_path)
    response = client.get(route)
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert title in html
    assert html.count('class="mlwb-primary-link') == 5
    assert '<nav class="module-nav' not in html
    assert 'any order' not in html.lower()
    assert 'ml_lab_workbench.css' in html
    assert 'ml_lab_workbench_context.js' in html
    assert '<main class="shell mlwb-main"' in html
    assert 'aria-current="page"' in html


def test_stable_routes_and_prefix(tmp_path):
    client = _client(tmp_path, prefix='/tools/ml-lab')
    home = client.get('/tools/ml-lab/')
    assert home.status_code == 200
    html = home.get_data(as_text=True)
    for suffix in ('data', 'workbench/feature-profiling', 'workbench/representation',
                   'workbench/comparison', 'workbench/general-ml-tools'):
        assert '/tools/ml-lab/' + suffix in html
    assert '/tools/ml-lab/api/workbench/context' in html
    for route in ('/tools/ml-lab/data', '/tools/ml-lab/data/compare',
                  '/tools/ml-lab/data/transform', '/tools/ml-lab/data/provenance',
                  '/tools/ml-lab/experiments/models',
                  '/tools/ml-lab/data/xml/collection'):
        assert client.get(route).status_code == 200, route


def test_active_section_maps_both_legacy_and_new_routes():
    examples = [
        ('my_plugin.data_lab', None, 'data'),
        ('my_plugin.xml_collection_inventory', None, 'data'),
        ('my_plugin.profile_view', None, 'profiling'),
        ('my_plugin.workbench_representation', None, 'representation'),
        ('my_plugin.task', {'capability_id': 'representation'}, 'representation'),
        ('my_plugin.task', {'capability_id': 'rbm'}, 'tools'),
        ('my_plugin.data_compare', None, 'comparison'),
        ('my_plugin.experiment_compare', None, 'comparison'),
        ('my_plugin.experiment_model', None, 'tools'),
        ('my_plugin.workspace_results', None, 'comparison'),
    ]
    for endpoint, args, expected in examples:
        assert section_for_request(endpoint, args) == expected
    assert section_for_request('my_plugin.experiment_model', None, 'representation') == 'representation'


def test_context_status_api_is_scoped_and_read_only(tmp_path):
    client = _client(tmp_path, prefix='/ml')
    page = client.get('/ml/workbench/representation')
    assert b'Active dataset: checking' in page.data
    first = client.get('/ml/api/workbench/context')
    assert first.status_code == 200
    assert first.json['status'] == 'empty'
    # Reading another section retains the exact same workbench instance.
    assert client.get('/ml/workbench/comparison').status_code == 200
    second = client.get('/ml/api/workbench/context')
    assert second.json['instance_id'] == first.json['instance_id']
    assert second.json['generation'] == first.json['generation'] == 0
