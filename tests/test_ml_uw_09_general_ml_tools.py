"""ML-UW-09 general-model launchers must not replace existing engines."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader

from ml_lab.ui.general_ml_catalog import general_ml_catalog


def _experiment(name, family='', method='', *, source=True, revision=1):
    return {
        'experiment_id': name,
        'name': name,
        'revision': revision,
        'dataset_ref': {'schema': 'ml-lab.experiment-dataset@1'} if source else {},
        'model_spec': ({'schema': 'ml-lab.model-configuration@1', 'family': family,
                        'method': method} if family else {}),
    }


def test_categories_share_existing_sources_and_do_not_claim_unsupported_optimization():
    rows = [
        _experiment('rbm', 'rbm', 'bernoulli'),
        _experiment('gan', 'gan', 'gan'),
        _experiment('supervised', 'regression', 'decision_tree', revision=7),
        _experiment('unconfigured'),
        _experiment('bad-source', 'regression', 'decision_tree', source=False),
    ]
    catalog = general_ml_catalog(rows)
    assert list(catalog) == ['models', 'ebm', 'optimization', 'generative']
    assert [row['name'] for row in catalog['models']] == ['rbm', 'gan', 'supervised', 'unconfigured']
    assert [row['name'] for row in catalog['ebm']] == ['rbm']
    # RBM/GAN currently have no supported optimization objective registry.
    assert [row['name'] for row in catalog['optimization']] == ['supervised']
    assert [row['name'] for row in catalog['generative']] == ['rbm', 'gan', 'supervised', 'unconfigured']
    assert catalog['optimization'][0]['revision'] == 7
    assert catalog['models'][-1]['configured'] is False


def test_bounded_catalog_retains_one_visible_generator_among_many_sources():
    rows = [_experiment(f'dataset-{i}') for i in range(14)]
    rows.append(_experiment('gan-later', 'gan', 'gan'))
    report = general_ml_catalog(rows, per_tool=4)
    assert len(report['models']) == len(report['generative']) == 4
    assert report['models'][0]['name'] == 'dataset-0'
    assert 'gan-later' in [entry['name'] for entry in report['generative']]
    for bad in (0, -1, 21, True):
        with pytest.raises(ValueError):
            general_ml_catalog(rows, per_tool=bad)


def test_unsupported_or_unconfigured_models_never_get_optimization_links():
    rows = [_experiment('bad-method', 'regression', 'madeup'),
            _experiment('missing-method', 'regression'),
            _experiment('unconfigured')]
    report = general_ml_catalog(rows)
    assert not report['optimization']
    assert not report['ebm']
    assert len(report['generative']) == 3


def test_project_catalog_is_metadata_only_and_does_not_run_jobs(tmp_path):
    from ml_lab.experiments import ExperimentStore
    first = tmp_path/'one'
    second = tmp_path/'two'
    first.mkdir()
    second.mkdir()
    left, right = ExperimentStore(first), ExperimentStore(second)
    dataset = {'schema': 'ml-lab.experiment-dataset@1', 'path': 'data.csv'}
    config = {'schema': 'ml-lab.model-configuration@1', 'family': 'regression', 'method': 'decision_tree'}
    row = left.create_experiment('only local', dataset_ref=dataset, model_spec=config)
    right.create_experiment('other project', dataset_ref=dataset)
    report = general_ml_catalog(left.list_experiments())
    assert [record['experiment_id'] for record in report['optimization']] == [row['experiment_id']]
    assert 'other project' not in repr(report)
    assert not left.list_runs(row['experiment_id'])
    with sqlite3.connect(left.db_path) as conn:
        assert conn.execute('PRAGMA user_version').fetchone()[0] == 3


def test_general_ml_template_parses_and_never_binds_post_actions():
    ui = Path(__file__).resolve().parents[1] / 'src/ml_lab/ui'
    Environment(loader=FileSystemLoader(ui/'templates')).get_template('ml_lab_ui/workbench_general_ml_tools.html')
    template = (ui/'templates/ml_lab_ui/workbench_general_ml_tools.html').read_text()
    assert 'method="post"' not in template.lower()
    assert '<nav class="module-nav' not in template
    for group in ('models', 'ebm', 'optimization', 'generative'):
        assert f'id="{group}"' in template


def test_grouped_hub_under_mount_prefix_reads_only_project_experiments(tmp_path):
    flask = pytest.importorskip('flask')
    from ml_lab.experiments import ExperimentStore
    from ml_lab.ui import create_ui_blueprint
    app = flask.Flask(__name__)
    app.config['TESTING'] = True
    app.register_blueprint(create_ui_blueprint(name='ml_uw09', experiment_project_root=str(tmp_path)),
                           url_prefix='/pah/ml-lab')
    store = ExperimentStore(tmp_path)
    sample = store.create_experiment('source <script>alert(1)</script>', dataset_ref={
        'schema': 'ml-lab.experiment-dataset@1', 'path': 'data.csv'})
    response = app.test_client().get('/pah/ml-lab/workbench/general-ml-tools')
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'source &lt;script&gt;' in html and '<script>alert(1)</script>' not in html
    assert html.count('class="mlwb-primary-link') == 7
    for group in ('models', 'ebm', 'optimization', 'generative'):
        assert f'id="{group}"' in html
        assert f'/pah/ml-lab/workbench/general-ml-tools#{group}' in html
    assert f'/pah/ml-lab/experiments/models?experiment_id={sample["experiment_id"]}' in html
    assert f'/pah/ml-lab/experiments/synthetic?experiment_id={sample["experiment_id"]}' in html
    assert not store.list_runs(sample['experiment_id'])


def test_general_ml_template_renders_without_flask_for_configured_and_empty_sections():
    from types import SimpleNamespace
    from urllib.parse import urlencode
    ui = Path(__file__).resolve().parents[1] / 'src/ml_lab/ui'
    env = Environment(loader=FileSystemLoader(ui/'templates'), autoescape=True)
    def url_for(endpoint, **params):
        anchor = params.pop('_anchor', None)
        path = '/ml/' + endpoint.rsplit('.', 1)[-1]
        return path + ('?' + urlencode(params) if params else '') + ('#' + anchor if anchor else '')
    html = env.get_template('ml_lab_ui/workbench_general_ml_tools.html').render(
        request=SimpleNamespace(blueprint='ml'), version='0.0',
        url_for=url_for, groups=general_ml_catalog([
            _experiment('source <img>', 'regression', 'decision_tree'),
            _experiment('RBM saved', 'rbm', 'bernoulli'),
            _experiment('unnamed'),
        ]), error=None, catalog_limit=200, workspace_scope='test', workspace_active_tool='tools',
        workspace_tools=[],
    )
    assert 'source &lt;img&gt;' in html and 'source <img>' not in html
    assert '/ml/experiment_model?experiment_id=source+%3Cimg%3E&amp;family=regression' in html
    assert '/ml/experiment_model?experiment_id=unnamed' in html
    assert 'id="optimization"' in html and 'id="generative"' in html
    assert 'method="post"' not in html.lower()
