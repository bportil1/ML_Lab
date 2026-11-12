"""ML-UW-06: fit-only representation preview and capability enforcement."""
from __future__ import annotations

import numpy as np
import pytest

from ml_lab.representation.sklearn_catalog import METHODS, explore_representation, method_catalog
from ml_lab.experiments import ExperimentStore, select_workbench_dataset
from ml_lab.experiments.workbench import get_workbench_instance


def _matrix():
    rng = np.random.default_rng(42)
    return rng.normal(size=(36, 5))


def test_method_registry_is_explicit_and_inverse_actions_are_declared():
    catalog = method_catalog()
    assert {method['id'] for method in catalog} == {method.id for method in METHODS}
    assert {m['id'] for m in catalog if not m['can_inverse_transform']} == {'isomap', 'tsne'}
    assert {m['id'] for m in catalog if not m['can_transform_new_data']} == {'tsne'}
    assert all(m['max_rows'] <= 5000 for m in catalog)


@pytest.mark.parametrize('method', ['pca', 'truncated_svd', 'fast_ica', 'nmf', 'kernel_pca', 'isomap', 'tsne'])
def test_method_fits_with_only_declared_capabilities(method):
    X = _matrix()
    scaling = 'minmax' if method == 'nmf' else 'standard'
    # t-SNE remains a visualization-only fit; it must never be assigned an inverse.
    report = explore_representation(X, method=method, components=2,
                                    scaling=scaling, reconstruct=method not in {'isomap', 'tsne'})
    assert report['latent_shape'] == [36, 2]
    assert report['schema'] == 'ml-lab.representation-explorer@1'
    assert report['scatter'] and len(report['scatter']['points']) == 36
    assert len(report['preview']) == 12
    assert not report['saved_model']
    if report['capabilities']['inverse_transform']:
        assert report['metrics']['mse'] >= 0
    else:
        assert report['metrics'] == {}


@pytest.mark.parametrize('method', ['tsne', 'isomap'])
def test_unavailable_inverse_is_rejected_server_side(method):
    with pytest.raises(ValueError, match='does not support inverse'):
        explore_representation(_matrix(), method=method, reconstruct=True)


def test_bad_inputs_and_constraints_have_explicit_errors():
    X = _matrix()
    with pytest.raises(ValueError, match='Unknown representation'):
        explore_representation(X, method='not-real')
    with pytest.raises(ValueError, match='NMF supports'):
        explore_representation(X, method='nmf', scaling='standard')
    with pytest.raises(ValueError, match='nonnegative'):
        explore_representation(X, method='nmf', scaling='none')
    with pytest.raises(ValueError, match='finite'):
        explore_representation(np.array([[np.nan, 1], [2, 3]]))
    with pytest.raises(ValueError, match='Plot axes'):
        explore_representation(X, axis_x=0, axis_y=0)
    with pytest.raises(ValueError, match='limited'):
        explore_representation(np.zeros((501, 5)), method='tsne')
    with pytest.raises(ValueError, match='rank/dimension'):
        explore_representation(X, components=6)
    with pytest.raises(ValueError, match='at most three'):
        explore_representation(X, method='tsne', components=4)
    single = explore_representation(X, components=1)
    assert single['scatter'] is None


def test_existing_pca_metric_parity():
    from ml_lab.representation import pca
    from ml_lab.representation.config import PCARepresentationConfig
    original = pca(_matrix(), config=PCARepresentationConfig(n_components=2))
    report = explore_representation(_matrix(), components=2, reconstruct=True)
    assert report['metrics']['mse'] == pytest.approx(original.metrics['mse'])
    assert report['metrics']['explained_variance_ratio_sum'] == pytest.approx(original.metrics['explained_variance_ratio_sum'])


def test_mounted_ui_active_dataset_and_stale_form(tmp_path):
    flask = pytest.importorskip('flask')
    from ml_lab.ui import create_ui_blueprint
    path = tmp_path / 'features.csv'
    path.write_text('a,b,c,target\n1,2,4,A\n2,3,3,B\n3,4,2,A\n4,5,1,B\n', encoding='utf-8')
    app = flask.Flask(__name__)
    app.config['TESTING'] = True
    app.register_blueprint(create_ui_blueprint(name='r06', experiment_project_root=tmp_path), url_prefix='/ml')
    client = app.test_client()
    assert b'No verified active dataset' in client.get('/ml/workbench/representation').data
    context = client.get('/ml/api/workbench/context').get_json()
    store = ExperimentStore(tmp_path)
    selected = select_workbench_dataset(store, context['instance_id'], path, expected_generation=0,
                                        feature_columns=['a', 'b', 'c'])
    before = path.read_bytes()
    fields = {'method': 'pca', 'components': '2', 'scaling': 'standard',
              'axis_x': '1', 'axis_y': '2', 'reconstruct': '1',
              'expected_generation': '1', 'source_sha256': selected['active_dataset']['sha256']}
    page = client.get('/ml/workbench/representation')
    assert page.status_code == 200 and b'Fit selected representation' in page.data
    response = client.post('/ml/workbench/representation', data=fields)
    assert response.status_code == 200 and b'Latent scatter preview' in response.data
    assert b'Input-space reconstruction diagnostics' in response.data
    assert path.read_bytes() == before
    assert get_workbench_instance(store, context['instance_id'])['generation'] == 1
    assert client.post('/ml/workbench/representation', data={**fields, 'method': 'tsne'}).status_code == 400
    assert client.post('/ml/workbench/representation', data={**fields, 'expected_generation': '0'}).status_code == 409
    path.write_text(path.read_text() + '5,6,0,A\n', encoding='utf-8')
    stale = client.post('/ml/workbench/representation', data=fields)
    assert stale.status_code == 400
    assert b'Select or repair a dataset' in stale.data


def test_mounted_ui_prefix_and_targets_must_be_explicit(tmp_path):
    flask = pytest.importorskip('flask')
    from ml_lab.ui import create_ui_blueprint
    path = tmp_path / 'features.csv'
    path.write_text('x,y,label\n1,2,A\n2,3,B\n3,4,A\n4,5,B\n', encoding='utf-8')
    app = flask.Flask(__name__)
    app.config['TESTING'] = True
    app.register_blueprint(create_ui_blueprint(name='target06', experiment_project_root=tmp_path), url_prefix='/plugins/ml-lab')
    client = app.test_client()
    ctx = client.get('/plugins/ml-lab/api/workbench/context').get_json()
    store = ExperimentStore(tmp_path)
    selected = select_workbench_dataset(store, ctx['instance_id'], path, expected_generation=0,
                                        feature_columns=['x', 'y'], target_columns=['label'])
    result = client.post('/plugins/ml-lab/workbench/representation', data={
        'expected_generation': '1', 'source_sha256': selected['active_dataset']['sha256'],
        'method': 'pca', 'components': '2', 'axis_x': '1', 'axis_y': '2',
    })
    assert result.status_code == 400 and b'without targets' in result.data


def test_representation_jinja_template_compiles():
    from jinja2 import Environment, FileSystemLoader
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / 'src' / 'ml_lab' / 'ui' / 'templates'
    Environment(loader=FileSystemLoader(root)).get_template('ml_lab_ui/workbench_representation.html')
