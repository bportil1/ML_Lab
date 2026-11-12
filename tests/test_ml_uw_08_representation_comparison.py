"""ML-UW-08: source-matched comparisons without model retraining."""
from __future__ import annotations

import json
import numpy as np
import pytest

from ml_lab.representation.comparison_workbench import (
    MAX_COMPARE_ROWS, compare_representations, comparison_csv, comparison_svg,
    comparison_latex, geometry_metrics,
)


def _data():
    rng = np.random.default_rng(13)
    data = rng.normal(size=(36, 4))
    data[:, 3] = 0.5 * data[:, 0] + 0.3 * data[:, 1] + rng.normal(size=36) * .05
    return data


def test_methods_share_input_and_geometry_and_preserve_noninverse_blanks():
    result = compare_representations(_data(), methods=['pca', 'truncated_svd', 'tsne'], components=2,
                                     scaling='standard', include_recorded_neural=False)
    assert result['schema'] == 'ml-lab.representation-comparison@1'
    assert result['source']['rows'] == 36
    assert result['conditions']['evaluation'].startswith('in-sample')
    assert [item['method'] for item in result['results']] == ['pca', 'truncated_svd', 'tsne']
    for entry in result['results']:
        assert entry['geometry']['neighbors'] == 5
        assert 0 <= entry['geometry']['trustworthiness'] <= 1
        assert len(entry['scatter']['points']) == 36
    assert result['results'][0]['metrics']['mse'] >= 0
    assert result['results'][2]['metrics'] == {}
    csv_content = comparison_csv(result)
    assert 'trustworthiness' in csv_content
    assert 't-SNE' in csv_content
    assert 'tsne' not in result['failures']
    assert '<svg' in comparison_svg(result) and 't-SNE' in comparison_svg(result)
    assert r'\begin{tabular}' in comparison_latex(result)
    assert 'N/A' in comparison_latex(result)


def test_incompatible_nmf_shows_explicit_failure_and_one_dimension_scatter_is_blank():
    report = compare_representations(_data(), methods=['pca', 'nmf'], scaling='standard', components=1)
    assert len(report['results']) == 1
    assert report['results'][0]['scatter'] is None
    assert len(report['failures']) == 1
    assert 'NMF supports' in report['failures'][0]['reason']


@pytest.mark.parametrize('methods,components,scale', [
    (['pca'], 2, 'minmax'), (['pca','pca'],2,'minmax'),
    (['pca','not_real'],2,'minmax'), (['pca','tsne'],9,'minmax'),
    (['pca','tsne'],2,'notscaling'),
])
def test_untrusted_settings_rejected(methods, components, scale):
    with pytest.raises(ValueError):
        compare_representations(_data(), methods=methods, components=components, scaling=scale)


def test_bounded_input_and_constant_geometry_handling():
    with pytest.raises(ValueError, match='8–'):
        compare_representations(np.ones((MAX_COMPARE_ROWS + 1, 3)), methods=['pca','truncated_svd'])
    x = np.arange(32).reshape(16,2)
    geo = geometry_metrics(x, np.ones((16,2)))
    assert geo['distance_rank_spearman'] is None
    assert 0 <= geo['trustworthiness'] <= 1


def _saved_fixture(tmp_path):
    from ml_lab.experiments import ExperimentStore, select_workbench_dataset
    from ml_lab.experiments.workbench import create_workbench_instance
    from ml_lab.experiments.models import configure_experiment_model
    matrix = _data()
    path = tmp_path / 'data.csv'
    path.write_text('a,b,c,d\n' + ''.join(','.join(str(v) for v in row) + '\n' for row in matrix))
    store = ExperimentStore(tmp_path)
    instance = create_workbench_instance(store)
    context = select_workbench_dataset(store, instance['instance_id'], path,
                                       expected_generation=0, feature_columns=['a','b','c','d'])
    revision = configure_experiment_model(store, context['experiment_id'], family='representation',
                                          method='mlp_autoencoder', settings={
                                              'model_config': {'latent_dim': 2},
                                              'training_config': {'scaling': 'minmax'},
                                          }, expected_revision=context['experiment_revision'])
    run = store.create_run(revision['experiment_id'], revision=revision['revision'],
                           environment={'executor': 'ml_lab.experiments.training@1'})
    store.update_run_status(run['run_id'], 'running')
    store.update_run_status(run['run_id'], 'succeeded')
    # Model output is deliberately fabricated for testing artifact validation
    # and input-space metric recomputation; no user data or trainer is called.
    result = {'task':'representation', 'result':{
        'method':'mlp_autoencoder', 'feature_names':['a','b','c','d'],
        'metadata':{'scaling':'minmax'},
        'metrics':{'mse':-99},  # Must be ignored in favor of recomputation.
        'latent': matrix[:,:2].tolist(), 'reconstruction': matrix.tolist(),
    }}
    artifact_path = tmp_path / 'result.json'
    artifact_path.write_text(json.dumps(result))
    artifact = store.artifacts.register(run['run_id'], artifact_path, role='output', kind='result',
                                        media_type='application/json')
    return matrix, store, context['active_dataset'], run, artifact, artifact_path


def test_saved_neural_artifact_matches_and_recalculates_source_metrics(tmp_path):
    matrix, store, selection, run, artifact, path = _saved_fixture(tmp_path)
    report = compare_representations(matrix, methods=['pca','truncated_svd'], components=2,
                                     scaling='minmax', store=store, selection=selection,
                                     experiment_id=run['experiment_id'])
    recorded = [item for item in report['results'] if item['kind'] == 'verified_saved_neural']
    assert len(recorded) == 1
    assert recorded[0]['run_id'] == run['run_id']
    assert recorded[0]['metrics']['mse'] == pytest.approx(0.0)
    assert recorded[0]['geometry']['trustworthiness'] >= 0
    # Fingerprint mismatch must exclude, even if JSON remains parseable.
    path.write_text(path.read_text() + '\n')
    rejected = compare_representations(matrix, methods=['pca','truncated_svd'], components=2,
                                       scaling='minmax', store=store, selection=selection,
                                       experiment_id=run['experiment_id'])
    assert not any(r['kind'] == 'verified_saved_neural' for r in rejected['results'])
    assert rejected['excluded_runs']


def test_saved_neural_requires_exact_matching_comparison_settings(tmp_path):
    matrix, store, selection, run, artifact, path = _saved_fixture(tmp_path)
    mismatched = compare_representations(matrix, methods=['pca','truncated_svd'], components=2,
                                          scaling='standard', store=store, selection=selection,
                                          experiment_id=run['experiment_id'])
    assert not any(r['kind'] == 'verified_saved_neural' for r in mismatched['results'])
    assert 'scaling differs' in mismatched['excluded_runs'][0]['reason']
    other = dict(selection, sha256='different')
    mismatched = compare_representations(matrix, methods=['pca','truncated_svd'], components=2,
                                          scaling='minmax', store=store, selection=other,
                                          experiment_id=run['experiment_id'])
    assert mismatched['excluded_runs'][0]['reason'] == 'Different source or selected features'


def test_jinja_comparison_template_compiles():
    from jinja2 import Environment, FileSystemLoader
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / 'src/ml_lab/ui/templates'
    Environment(loader=FileSystemLoader(root)).get_template('ml_lab_ui/workbench_comparison.html')


def test_mounted_comparison_route_and_stale_selection(tmp_path):
    flask = pytest.importorskip('flask')
    from ml_lab.ui import create_ui_blueprint
    from ml_lab.experiments import ExperimentStore, select_workbench_dataset
    path = tmp_path/'data.csv'
    matrix = _data()
    path.write_text('a,b,c,d\n' + ''.join(','.join(str(v) for v in row) + '\n' for row in matrix))
    app = flask.Flask(__name__)
    app.config['TESTING'] = True
    app.register_blueprint(create_ui_blueprint(name='uw08', experiment_project_root=tmp_path), url_prefix='/ml')
    client = app.test_client()
    initial = client.get('/ml/workbench/comparison')
    assert initial.status_code == 200 and b'No verified active dataset' in initial.data
    ctx = client.get('/ml/api/workbench/context').get_json()
    store = ExperimentStore(tmp_path)
    context = select_workbench_dataset(store, ctx['instance_id'], path,
                                       expected_generation=ctx['generation'], feature_columns=['a','b','c','d'])
    form = {'expected_generation':str(context['generation']),
            'expected_revision':str(context['experiment_revision']),
            'source_sha256':context['active_dataset']['sha256'],
            'methods':['pca', 'tsne'], 'scaling':'standard', 'components':'2'}
    assert b'Compute comparison' in client.get('/ml/workbench/comparison').data
    rendered = client.post('/ml/workbench/comparison', data=form)
    assert rendered.status_code == 200 and b'Side-by-side evaluation' in rendered.data
    assert b'Latent-space preview' in rendered.data
    exported = client.post('/ml/workbench/comparison', data={**form, 'export':'csv'})
    assert exported.status_code == 200 and 'text/csv' in exported.content_type
    assert b'PCA' in exported.data
    for format, token in [('svg', b'<svg'), ('tex', b'\\begin{tabular}')]:
        response = client.post('/ml/workbench/comparison', data={**form, 'export':format})
        assert response.status_code == 200 and token in response.data
    assert client.post('/ml/workbench/comparison', data={**form,'expected_generation':'0'}).status_code == 409
    path.write_text(path.read_text() + '1,2,3,4\n')
    assert client.post('/ml/workbench/comparison', data=form).status_code == 400
