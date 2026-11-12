"""ML-UW-14: locally executable final workbench acceptance contracts.

These tests do not fetch network resources, run model training, or create
artifacts outside pytest's isolated temporary project. Flask is optional.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ml_lab.data.feature_explorer import explore_cached_views, explore_features
from ml_lab.experiments import (
    ExperimentStore, activate_workbench_experiment, create_workbench_instance,
    select_workbench_dataset,
)
from ml_lab.experiments.feature_profile_cache import (
    feature_baseline_key, load_feature_report, save_feature_report,
)
from ml_lab.experiments.feature_roles import feature_roles, set_feature_role
from ml_lab.representation.autoencoder_workbench import DEFAULTS, save_autoencoder_configuration
from ml_lab.representation.comparison_workbench import compare_representations
from ml_lab.representation.sklearn_catalog import method_catalog


TEMPLATES = Path(__file__).resolve().parents[1] / 'src/ml_lab/ui/templates/ml_lab_ui'


def _source(project: Path):
    source = project / 'specimen.csv'
    lines = ['x,y,z'] + [f'{n},{2 * n + (n % 3)},{(n * n) % 13}' for n in range(1, 65)]
    source.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return source


def test_every_ui_template_parses_without_optional_flask():
    env = Environment(loader=FileSystemLoader(TEMPLATES.parent),
                      autoescape=select_autoescape(['html']))
    names = sorted(path.name for path in TEMPLATES.glob('*.html'))
    assert len(names) >= 20  # Covers both the workbench and preserved legacy UI.
    for filename in names:
        env.get_template('ml_lab_ui/' + filename)


def test_five_section_scientific_contract_is_preserved():
    navigation = (TEMPLATES.parent.parent / 'workbench_navigation.py').read_text(encoding='utf-8')
    labels = ('Data', 'Feature Profiling', 'Representation', 'Comparison', 'General ML Tools')
    positions = [navigation.index(f'"{label}"') for label in labels]
    assert positions == sorted(positions)
    assert '"Planned"' in navigation  # Held-out benchmarking must not claim availability.
    methods = {item['id']: item for item in method_catalog()}
    assert {'pca', 'truncated_svd', 'fast_ica', 'nmf', 'kernel_pca', 'isomap', 'tsne'} <= set(methods)
    assert all(not methods[key]['can_inverse_transform'] for key in ('isomap', 'tsne'))
    assert not methods['tsne']['can_transform_new_data']


def test_data_profile_representation_comparison_and_saved_revision_smoke(tmp_path):
    """A shared verified source is consumed without implicit training or mutation."""
    project = tmp_path / 'project'
    other = tmp_path / 'other'
    project.mkdir()
    other.mkdir()
    source = _source(project)
    original = source.read_bytes()
    store = ExperimentStore(project)
    initial = create_workbench_instance(store)
    selected = select_workbench_dataset(
        store, initial['instance_id'], source, expected_generation=0,
        feature_columns=['x', 'y', 'z'],
    )
    sha = selected['active_dataset']['sha256']

    # Profiling display roles remain presentation metadata, not training inputs.
    set_feature_role(store, selected['instance_id'], column='z', role='discrete',
                     source_sha256=sha, expected_generation=selected['generation'])
    overrides = feature_roles(store, selected)
    assert overrides == {'z': 'discrete'}
    baseline = explore_features(str(source), source_sha256=sha, overrides=overrides,
                                selected='x', left='x', right='y')
    key = feature_baseline_key(selected, overrides=overrides)
    assert save_feature_report(store, selected, key, baseline)
    replay = explore_cached_views(str(source), source_sha256=sha,
                                  baseline=load_feature_report(store, selected, key),
                                  selected='z', left='x', right='y')
    fresh = explore_features(str(source), source_sha256=sha, overrides=overrides,
                             selected='z', left='x', right='y')
    assert replay == fresh

    frame = pd.read_csv(source)[['x', 'y', 'z']]
    comparison = compare_representations(frame, methods=['pca', 'truncated_svd'],
                                         components=2, scaling='standard',
                                         store=store, selection=selected['active_dataset'],
                                         experiment_id=selected['experiment_id'])
    assert comparison['source']['sha256'] == sha
    assert comparison['source']['feature_names'] == ['x', 'y', 'z']
    assert {row['method'] for row in comparison['results']} == {'pca', 'truncated_svd'}
    assert not store.list_runs(selected['experiment_id'])

    form = {
        **DEFAULTS, 'experiment_id': selected['experiment_id'],
        'expected_revision': str(selected['experiment_revision']),
        'expected_generation': str(selected['generation']),
        'source_sha256': sha,
        'latent_dim': '2', 'hidden_dims': '8,4', 'epochs': '2', 'batch_size': '8',
    }
    configured = save_autoencoder_configuration(store, selected, form)
    assert configured['experiment_revision'] == selected['experiment_revision'] + 1
    assert store.get_experiment(selected['experiment_id'], revision=selected['experiment_revision'])['model_spec'] == {}
    assert not store.list_runs(selected['experiment_id'])  # Saving does not train.

    # Re-selecting a verified archived revision never duplicates the experiment.
    activated = activate_workbench_experiment(
        store, selected['instance_id'], selected['experiment_id'],
        expected_generation=configured['generation'],
        revision=selected['experiment_revision'],
    )
    assert activated['active_dataset']['sha256'] == sha
    assert len(store.list_experiments()) == 1
    assert feature_roles(store, activated) == {'z': 'discrete'}
    assert not ExperimentStore(other).list_experiments()
    assert source.read_bytes() == original
    with sqlite3.connect(store.db_path) as con:
        assert con.execute('PRAGMA user_version').fetchone()[0] == 3


def test_mounted_page_navigation_is_read_only_and_prefix_safe(tmp_path):
    flask = pytest.importorskip('flask')
    from ml_lab.ui import create_ui_blueprint

    source = _source(tmp_path)
    store = ExperimentStore(tmp_path)
    app = flask.Flask(__name__)
    app.config.update(TESTING=True)
    app.register_blueprint(
        create_ui_blueprint(name='mluw14', experiment_project_root=str(tmp_path)),
        url_prefix='/pah/labs/ml-lab',
    )
    client = app.test_client()
    before = source.read_bytes()
    paths = (
        '/data', '/workbench/feature-profiling', '/workbench/representation',
        '/workbench/comparison', '/workbench/general-ml-tools',
    )
    context = client.get('/pah/labs/ml-lab/api/workbench/context').get_json()
    for path in paths:
        response = client.get('/pah/labs/ml-lab' + path)
        assert response.status_code == 200, path
        html = response.get_data(as_text=True)
        assert html.count('class="mlwb-primary-link') == 5, path
        assert '/pah/labs/ml-lab/api/workbench/context' in html
    tools = client.get('/pah/labs/ml-lab/workbench/general-ml-tools').get_data(as_text=True)
    for anchor in ('#models', '#ebm', '#optimization', '#generative'):
        assert '/pah/labs/ml-lab/workbench/general-ml-tools' + anchor in tools
    later = client.get('/pah/labs/ml-lab/api/workbench/context').get_json()
    assert later['instance_id'] == context['instance_id']
    assert later['generation'] == context['generation'] == 0
    assert store.list_experiments() == []
    assert source.read_bytes() == before
