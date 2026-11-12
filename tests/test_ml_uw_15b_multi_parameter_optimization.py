"""ML-UW-15B: joint numeric controls, log geometry, and legacy compatibility."""
import json

import numpy as np
import pytest

from ml_lab.optimization import (FireflyConfig, FunctionEvaluator, OptimizationRuntime,
                                 ParameterSpec, SearchSpace, optimize)
from ml_lab.experiments import (
    ExperimentStore, ExperimentStoreError, configure_experiment_model,
    execute_optimization_run, queue_optimization_run, select_experiment_dataset,
    validate_optimization_plan,
)
from test_experiment_optimization import PLAN, prepared


def test_log_movement_coordinate_round_trip_and_legacy_linear_reproducibility(tmp_path):
    space = SearchSpace([ParameterSpec('lr', 'continuous_log', 1e-6, 1e-2),
                         ParameterSpec('width', 'integer_log', 2, 128),
                         ParameterSpec('x', 'continuous', -2, 2)])
    value = np.asarray([1e-4, 16, 0.5])
    movement = space.to_movement_coordinates(value)
    assert movement[0] == pytest.approx(-4)
    assert movement[1] == pytest.approx(np.log10(16))
    assert space.from_movement_coordinates(movement).tolist() == pytest.approx(value.tolist())
    assert space.movement_bound_pairs[0] == pytest.approx((-6, -2))
    def objective(p, **_):
        return (np.log10(p['lr']) + 3)**2 + (p['width'] - 16)**2 + p['x']**2
    results = [optimize(space, FunctionEvaluator(space, objective), algorithm='firefly',
                        config=FireflyConfig(population_size=2, iterations=1, seed=13),
                        runtime=OptimizationRuntime(device='cpu', candidate_parallelism='serial'),
                        output_path=str(tmp_path / str(index))) for index in range(2)]
    assert results[0].best_position == results[1].best_position
    assert 1e-6 <= results[0].best_parameters['lr'] <= 1e-2
    assert 2 <= results[0].best_parameters['width'] <= 128


def test_multidimensional_plan_rejects_duplicate_and_unsafe_controls(tmp_path):
    store, exp, _ = prepared(tmp_path)
    valid = {**PLAN, 'parameters': [{'path': 'config.n_components', 'lower': 1, 'upper': 2}]}
    single = validate_optimization_plan(exp, valid)
    assert single['parameter'] == 'config.n_components'
    assert single['parameters'][0]['kind'] == 'integer'
    base = {'algorithm': 'random_search', 'metric': 'rmse', 'goal': 'minimize', 'seed': 42,
            'evaluations': 2}
    for items in (
        [{'path': 'config.n_components', 'lower': 1, 'upper': 2}] * 2,
        [{'path': 'config.n_components', 'lower': 1, 'upper': 2, 'scale': 'abs'}],
        [{'path': 'config.random_state', 'lower': 1, 'upper': 2}],
        [{'path': 'config.n_components', 'lower': 0, 'upper': 2, 'scale': 'log'}],
        [{'path': 'config.n_components', 'lower': True, 'upper': 2}],
        [{'path': 'config.n_components', 'lower': 1, 'upper': 2}] * 5,
    ):
        with pytest.raises(ExperimentStoreError):
            validate_optimization_plan(exp, {**base, 'parameters': items})
    assert not store.list_runs(exp['experiment_id'])


def _classification(tmp_path):
    project = tmp_path/'model_project'
    project.mkdir()
    source = project/'samples.csv'
    source.write_text('x,z,label\n' + '\n'.join(
        f'{i},{i%5},{i%2}' for i in range(1, 41)) + '\n', encoding='utf-8')
    store = ExperimentStore(project)
    selected = select_experiment_dataset(store, source, feature_columns=['x', 'z'], target_columns=['label'])
    exp = configure_experiment_model(store, selected['experiment_id'], family='classification', method='ridge',
                                     settings={'config': {'cv_folds': 2}, 'test_size': 0.25},
                                     expected_revision=selected['revision'])
    return store, exp


@pytest.mark.parametrize('algorithm', ['random_search', 'firefly'])
def test_joint_search_is_project_local_and_does_not_promote_model(tmp_path, algorithm):
    store, exp = _classification(tmp_path)
    plan = {'algorithm': algorithm, 'metric': 'accuracy', 'goal': 'maximize', 'seed': 11,
            'parameters': [
                {'path': 'config.cv_folds', 'lower': 2, 'upper': 3},
                {'path': 'task_options.test_size', 'lower': 0.2, 'upper': 0.3, 'scale': 'linear'},
            ], 'evaluations': 2, 'population_size': 2, 'iterations': 0}
    queued = queue_optimization_run(store, exp['experiment_id'], plan, expected_revision=exp['revision'])
    frozen = queued['environment']['plan']
    assert [p['path'] for p in frozen['parameters']] == ['config.cv_folds', 'task_options.test_size']
    assert frozen['parameters'][0]['kind'] == 'integer'
    result = execute_optimization_run(store, queued['run_id'])
    assert result['status'] == 'succeeded'
    assert result['evaluations'] == 2
    assert set(result['best_parameters']) == {'config.cv_folds', 'task_options.test_size'}
    assert store.get_experiment(exp['experiment_id'])['revision'] == exp['revision']
    artifacts = store.artifacts.list_for_run(result['run_id'])
    assert sum(a['kind'] == 'trial' for a in artifacts) == 2
    assert all(store.artifacts.verify(a['artifact_id'])['status'] == 'valid' for a in artifacts)
    result_file = next(a for a in artifacts if a['kind'] == 'result')
    result_json = json.loads((store.project_root/result_file['path']).read_text())
    assert result_json['search_parameters'] == ['config.cv_folds', 'task_options.test_size']


def test_frozen_multidimensional_plan_revalidates_on_execution(tmp_path):
    store, exp = _classification(tmp_path)
    request = {'algorithm': 'random_search', 'metric': 'accuracy', 'goal': 'maximize', 'seed': 9,
               'parameters': [{'path': 'config.cv_folds', 'lower': 2, 'upper': 3},
                              {'path': 'task_options.test_size', 'lower': 0.2, 'upper': 0.3}],
               'evaluations': 2}
    frozen = validate_optimization_plan(exp, request)
    assert validate_optimization_plan(exp, frozen) == frozen
    assert not store.list_runs(exp['experiment_id'])


def test_conflicting_frozen_single_parameter_contract(tmp_path):
    store, exp, _ = prepared(tmp_path)
    plan = {**PLAN, 'parameters': [{'path': 'config.n_components', 'lower': 1, 'upper': 2}]}
    valid = validate_optimization_plan(exp, plan)
    assert validate_optimization_plan(exp, valid) == valid
    for key, mismatch in [('parameter', 'config.random_state'), ('upper', 3),
                          ('scale', 'log')]:
        with pytest.raises(ExperimentStoreError):
            validate_optimization_plan(exp, {**valid, key: mismatch})


def test_optimizer_template_multi_controls_compile():
    from jinja2 import Environment, FileSystemLoader
    from pathlib import Path
    folder = Path(__file__).parents[1]/'src/ml_lab/ui/templates'
    env = Environment(loader=FileSystemLoader(folder))
    template = env.get_template('ml_lab_ui/experiment_model.html')
    raw = (folder/'ml_lab_ui/experiment_model.html').read_text()
    assert 'parameter_{{ index }}' in raw
    assert 'name="scale_{{ index }}"' in raw
    assert 'name="algorithm"' in raw
    assert template is not None
