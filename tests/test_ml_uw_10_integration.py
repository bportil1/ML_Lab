"""ML-UW-10: explicit saved-experiment handoff and five-tab state integrity.

Headless checks run without the optional Flask UI. Browser route checks run
only when Flask is installed and protect PAH mount-path compatibility.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest
from jinja2 import Environment, FileSystemLoader

from ml_lab.experiments import (
    DatasetSelectionError, ExperimentStore, WorkbenchConflict,
    activate_workbench_experiment, create_workbench_instance,
    get_workbench_instance, select_workbench_dataset,
)
from ml_lab.experiments.feature_roles import feature_roles, set_feature_role
from ml_lab.representation.autoencoder_workbench import DEFAULTS, save_autoencoder_configuration
from ml_lab.representation.comparison_workbench import compare_representations


def _source(root: Path, name='first.csv', offset=0):
    source = root / name
    source.write_text('x,y,z\n' + ''.join(
        f'{i + offset},{2*i + offset},{(i*i)%17 + offset}\n' for i in range(1, 25)
    ), encoding='utf-8')
    return source


def _configured_form(context):
    return {
        **DEFAULTS, 'experiment_id': context['experiment_id'],
        'expected_generation': str(context['generation']),
        'expected_revision': str(context['experiment_revision']),
        'source_sha256': context['active_dataset']['sha256'],
        'latent_dim': '2', 'hidden_dims': '8,4', 'epochs': '2', 'batch_size': '8',
    }


def test_complete_data_profile_representation_comparison_handoff_and_restore(tmp_path):
    """Both the live revision and archived dataset can be revisited losslessly."""
    first, second = _source(tmp_path), _source(tmp_path, 'second.csv', 100)
    original = first.read_bytes()
    store = ExperimentStore(tmp_path)
    empty = create_workbench_instance(store)
    first_context = select_workbench_dataset(
        store, empty['instance_id'], first, expected_generation=0,
        feature_columns=['x', 'y', 'z'],
    )
    assert first_context['status'] == 'ready'
    assert first_context['generation'] == 1
    sha = first_context['active_dataset']['sha256']
    old_revision = first_context['experiment_revision']

    set_feature_role(store, empty['instance_id'], column='z', role='discrete',
                     source_sha256=sha, expected_generation=1)
    assert feature_roles(store, first_context) == {'z': 'discrete'}
    from ml_lab.data.feature_explorer import explore_features
    report = explore_features(str(first), source_sha256=sha, overrides={'z': 'discrete'},
                              selected='z', left='x', right='y')
    assert report['distribution']['kind'] == 'categories'

    values = pd.read_csv(first)[['x', 'y', 'z']]
    compared = compare_representations(
        values, methods=['pca', 'truncated_svd'], components=2, scaling='standard',
        store=store, selection=first_context['active_dataset'],
        experiment_id=first_context['experiment_id'], include_recorded_neural=True,
    )
    assert compared['source']['feature_names'] == ['x', 'y', 'z']
    assert compared['source']['sha256'] == sha
    assert len(compared['results']) == 2
    assert not store.list_runs(first_context['experiment_id'])

    configured = save_autoencoder_configuration(store, first_context, _configured_form(first_context))
    assert configured['experiment_id'] == first_context['experiment_id']
    assert configured['experiment_revision'] == old_revision + 1
    assert configured['generation'] == 2
    assert store.get_experiment(first_context['experiment_id'], revision=old_revision)['model_spec'] == {}
    assert feature_roles(store, configured) == {'z': 'discrete'}

    switched = select_workbench_dataset(
        store, empty['instance_id'], second, expected_generation=2,
        feature_columns=['x', 'y', 'z'],
    )
    assert switched['status'] == 'ready' and switched['generation'] == 3
    assert feature_roles(store, switched) == {}
    restored = activate_workbench_experiment(
        store, empty['instance_id'], first_context['experiment_id'],
        expected_generation=3, revision=configured['experiment_revision'],
    )
    assert restored['status'] == 'ready' and restored['generation'] == 4
    assert restored['active_dataset']['sha256'] == sha
    assert feature_roles(store, restored) == {'z': 'discrete'}
    assert len(store.list_experiments()) == 2
    assert not store.list_runs(first_context['experiment_id'])
    assert original == first.read_bytes()
    with sqlite3.connect(store.db_path) as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == 3


def test_saved_revisions_remain_verified_and_project_private(tmp_path):
    left, right = tmp_path/'left', tmp_path/'right'
    left.mkdir(); right.mkdir()
    source = _source(left)
    store = ExperimentStore(left)
    context = create_workbench_instance(store)
    chosen = select_workbench_dataset(store, context['instance_id'], source, expected_generation=0)
    right_store = ExperimentStore(right)
    other = create_workbench_instance(right_store)
    with pytest.raises(ValueError):
        activate_workbench_experiment(right_store, other['instance_id'], chosen['experiment_id'],
                                      expected_generation=0, revision=chosen['experiment_revision'])
    with pytest.raises(WorkbenchConflict):
        activate_workbench_experiment(store, chosen['instance_id'], chosen['experiment_id'],
                                      expected_generation=0, revision=chosen['experiment_revision'])
    source.write_text(source.read_text() + '30,60,2\n', encoding='utf-8')
    stale = get_workbench_instance(store, chosen['instance_id'])
    assert stale['status'] == 'stale' and stale['active_dataset'] is None
    with pytest.raises(DatasetSelectionError):
        activate_workbench_experiment(store, chosen['instance_id'], chosen['experiment_id'],
                                      expected_generation=1, revision=chosen['experiment_revision'])
    assert get_workbench_instance(store, chosen['instance_id'])['generation'] == 1


def test_new_templates_compile_and_explain_explicit_handoff():
    base = Path(__file__).resolve().parents[1]/'src/ml_lab/ui/templates'
    env = Environment(loader=FileSystemLoader(base), autoescape=True)
    for name in ('base.html', 'workbench_experiment_activate.html',
                 'workbench_general_ml_tools.html', 'workbench_section.html'):
        env.get_template('ml_lab_ui/' + name)
    activate = (base/'ml_lab_ui/workbench_experiment_activate.html').read_text()
    catalog = (base/'ml_lab_ui/workbench_general_ml_tools.html').read_text()
    assert 'method="post"' in activate
    assert 'expected_generation' in activate and 'revision' in activate
    assert 'workbench_activate_experiment' in catalog
    assert 'Use in workbench' in catalog
    assert 'workbench_context_api' in (base/'ml_lab_ui/base.html').read_text()


def test_handoff_preview_escapes_untrusted_labels_without_flask():
    from types import SimpleNamespace
    from urllib.parse import urlencode

    base = Path(__file__).resolve().parents[1]/'src/ml_lab/ui/templates'
    env = Environment(loader=FileSystemLoader(base), autoescape=True)

    def url_for(endpoint, **params):
        suffix = '?' + urlencode(params) if params else ''
        return '/pah/ml-lab/' + endpoint.rsplit('.', 1)[-1] + suffix

    context = {'generation': 7, 'experiment_id': None, 'experiment_revision': None}
    experiment = {
        'name': 'Model <script>do_not_run()</script>',
        'experiment_id': '0123456789abcdef0123456789abcdef', 'revision': 2,
        'dataset_ref': {'path': 'private/<img src=x>', 'rows': 20, 'columns': 3,
                        'feature_columns': ['a', 'b'], 'target_columns': []},
    }
    html = env.get_template('ml_lab_ui/workbench_experiment_activate.html').render(
        request=SimpleNamespace(blueprint='ml'), url_for=url_for, version='test',
        workspace_scope='test', workspace_active_tool='data', workspace_tools=[],
        context=context, experiment=experiment, verified=True, error=None, activated=False,
    )
    assert '<script>do_not_run()' not in html and '&lt;script&gt;' in html
    assert '<img src=x>' not in html and '&lt;img src=x&gt;' in html
    assert 'name="expected_generation" value="7"' in html
    assert 'name="revision" value="2"' in html
    assert 'method="post"' in html
    assert '/pah/ml-lab/workbench_activate_experiment' in html


def test_mounted_five_section_navigation_and_explicit_saved_revision(tmp_path):
    flask = pytest.importorskip('flask')
    from ml_lab.ui import create_ui_blueprint

    source = _source(tmp_path)
    store = ExperimentStore(tmp_path)
    experiment = store.create_experiment('Verified training source', dataset_ref={})
    # Create the valid dataset selection through the existing public service.
    app = flask.Flask(__name__)
    app.config['TESTING'] = True
    app.register_blueprint(create_ui_blueprint(name='research', experiment_project_root=str(tmp_path)),
                           url_prefix='/pah/ml-lab')
    client = app.test_client()
    initial = client.get('/pah/ml-lab/api/workbench/context').get_json()
    assert initial['status'] == 'empty' and initial['generation'] == 0
    # Do not activate this browser's workbench: pin a separate saved revision.
    archive = create_workbench_instance(store)
    saved = select_workbench_dataset(store, archive['instance_id'], source, expected_generation=0)
    location = f'/pah/ml-lab/workbench/experiment/activate?experiment_id={saved["experiment_id"]}&revision=1'
    preview = client.get(location)
    assert preview.status_code == 200
    assert b'Activate this saved revision' in preview.data
    assert store.list_runs(saved['experiment_id']) == []
    assert client.get('/pah/ml-lab/api/workbench/context').get_json()['generation'] == 0
    html = client.get('/pah/ml-lab/workbench/general-ml-tools').get_data(as_text=True)
    assert '/pah/ml-lab/workbench/experiment/activate?' in html
    assert '/pah/ml-lab/api/workbench/context' in html

    data = {'experiment_id': saved['experiment_id'], 'revision': '1', 'expected_generation': '0'}
    accepted = client.post('/pah/ml-lab/workbench/experiment/activate', data=data)
    assert accepted.status_code == 200
    assert b'Existing experiment activated' in accepted.data
    active = client.get('/pah/ml-lab/api/workbench/context').get_json()
    assert active['status'] == 'ready' and active['generation'] == 1
    assert active['experiment_id'] == saved['experiment_id']
    assert client.post('/pah/ml-lab/workbench/experiment/activate', data=data).status_code == 409
    repeated = client.post('/pah/ml-lab/workbench/experiment/activate', data={**data, 'expected_generation':'1'})
    assert repeated.status_code == 200  # Idempotent: no extra generation
    assert client.get('/pah/ml-lab/api/workbench/context').get_json()['generation'] == 1
    for page in ('data', 'workbench/feature-profiling', 'workbench/representation',
                 'workbench/comparison', 'workbench/general-ml-tools'):
        response = client.get('/pah/ml-lab/' + page)
        assert response.status_code == 200, page
        assert response.get_data(as_text=True).count('class="mlwb-primary-link') == 5
    profile = client.get('/pah/ml-lab/workbench/feature-profiling?inspect=1')
    assert profile.status_code == 200 and b'Per-feature metadata' in profile.data
    represented = client.post('/pah/ml-lab/workbench/representation', data={
        'expected_generation':'1', 'source_sha256':saved['active_dataset']['sha256'],
        'method':'pca', 'components':'2', 'axis_x':'1', 'axis_y':'2',
        'scaling':'standard', 'reconstruct':'1',
    })
    assert represented.status_code == 200 and b'Latent scatter preview' in represented.data
    compared = client.post('/pah/ml-lab/workbench/comparison', data={
        'expected_generation':'1', 'expected_revision':'1',
        'source_sha256':saved['active_dataset']['sha256'],
        'methods':['pca','truncated_svd'], 'scaling':'standard', 'components':'2',
    })
    assert compared.status_code == 200 and b'Side-by-side evaluation' in compared.data
    assert client.get('/pah/ml-lab/api/workbench/context').get_json()['generation'] == 1
    # Same saved revision in a separate browser creates a distinct workbench.
    other = app.test_client()
    assert other.get('/pah/ml-lab/api/workbench/context').get_json()['status'] == 'empty'
    assert store.list_runs(saved['experiment_id']) == []
    assert len(store.list_experiments()) == 2  # archived test fixture + saved selection
