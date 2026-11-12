"""ML-UW-15A: algorithm-independent random-search baseline and run records."""
import csv
import json
from pathlib import Path

import pytest

from ml_lab.optimization import (FireflyConfig, FunctionEvaluator, OptimizationRuntime,
                                 ParameterSpec, RandomSearchConfig, SearchSpace,
                                 available_optimizers, optimize)
from ml_lab.experiments import (ExperimentStoreError, optimize_configured_experiment,
                                queue_optimization_run, validate_optimization_plan)
from ml_lab.experiments.comparison import compare_experiment_runs
from test_experiment_optimization import PLAN, prepared


def test_random_search_seeded_mixed_space_and_artifacts(tmp_path):
    space = SearchSpace([ParameterSpec('x', 'continuous_log', 1e-4, 1e-2),
                         ParameterSpec('y', 'integer', 1, 5),
                         ParameterSpec('activation', 'categorical', choices=('a', 'b'))],
                        fixed_parameters={'fixed': 3})

    def f(params, **_):
        assert params['fixed'] == 3
        return {'fitness': params['x'] + params['y'], 'objective_terms': {'sum': params['x'] + params['y']}}

    runs = []
    for folder in ('first', 'second'):
        runs.append(optimize(space, FunctionEvaluator(space, f), algorithm='random_search',
                             config=RandomSearchConfig(evaluations=7, seed=23),
                             runtime=OptimizationRuntime(device='cpu', candidate_parallelism='serial'),
                             output_path=str(tmp_path / folder)))
    assert 'random_search' in available_optimizers()
    assert runs[0].algorithm == 'random_search'
    assert runs[0].best_position == runs[1].best_position
    assert runs[0].best_parameters == runs[1].best_parameters
    assert runs[0].evaluations == 7
    assert {p.name for p in (tmp_path/'first').iterdir()} == {
        'optimization_search_space.json', 'optimization_history.csv',
        'random_search_evaluations.json', 'random_search_history.json',
        'optimization_result.json'}
    with (tmp_path/'first'/'optimization_history.csv').open() as fp:
        assert len(list(csv.DictReader(fp))) == 7
    payload = json.loads((tmp_path/'first'/'optimization_result.json').read_text())
    assert payload['algorithm'] == 'random_search'
    assert payload['search_parameters'] == ['x', 'y', 'activation']


def test_random_search_failed_candidates_and_stop(tmp_path):
    space = SearchSpace([ParameterSpec('x', 'continuous', 0, 1)])
    def failed(params, **_):
        raise ValueError('failure from evaluator')
    with pytest.raises(RuntimeError, match='every candidate evaluation returned an error'):
        optimize(space, FunctionEvaluator(space, failed), algorithm='random_search',
                 config=RandomSearchConfig(evaluations=2),
                 runtime=OptimizationRuntime(candidate_parallelism='serial'), output_path=str(tmp_path))
    payload = json.loads((tmp_path/'optimization_result.json').read_text())
    assert payload['metadata']['optimization_failed'] is True
    calls = []
    def one(params, **_):
        calls.append(1)
        return 1.
    from ml_lab.optimization.firefly import OptimizationStopped
    with pytest.raises(OptimizationStopped):
        optimize(space, FunctionEvaluator(space, one), algorithm='random_search',
                 config=RandomSearchConfig(evaluations=8), should_stop=lambda: bool(calls),
                 runtime=OptimizationRuntime(candidate_parallelism='serial'))
    assert len(calls) == 1


def test_random_search_experiment_run_and_comparison(tmp_path):
    store, exp, source = prepared(tmp_path)
    plan = {**PLAN, 'algorithm': 'random_search', 'evaluations': 3}
    queued = queue_optimization_run(store, exp['experiment_id'], plan, expected_revision=exp['revision'])
    assert queued['environment']['plan']['evaluations'] == 3
    from ml_lab.experiments import execute_optimization_run
    result = execute_optimization_run(store, queued['run_id'])
    assert result['evaluations'] == 3
    artifacts = store.artifacts.list_for_run(result['run_id'])
    assert sum(a['kind'] == 'trial' for a in artifacts) == 3
    assert all(store.artifacts.verify(a['artifact_id'])['status'] == 'valid' for a in artifacts)
    report = compare_experiment_runs(store, exp['experiment_id'], run_ids=[result['run_id']])
    assert report['runs'][0]['optimization']['algorithm'] == 'random_search'
    assert source.exists()
    assert store.get_experiment(exp['experiment_id'])['revision'] == exp['revision']


def test_random_search_plan_rejects_bad_budget_and_algorithm(tmp_path):
    store, exp, _ = prepared(tmp_path)
    for bad in ({'algorithm': 'random_search', 'evaluations': 33},
                {'algorithm': 'random_search', 'evaluations': True},
                {'algorithm': 'anneal'}, {'algorithm': 'random_search', 'evaluations': 0}):
        with pytest.raises(ExperimentStoreError):
            validate_optimization_plan(exp, {**PLAN, **bad})
    assert validate_optimization_plan(exp, PLAN)['algorithm'] == 'firefly'
    assert not store.list_runs(exp['experiment_id'])
