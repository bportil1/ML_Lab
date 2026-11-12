"""ML-UW-07 regression: typed neural configuration and project-owned checkpoints."""
from __future__ import annotations

import copy
import pytest

from ml_lab.experiments import ExperimentStore, select_workbench_dataset
from ml_lab.experiments.workbench import WorkbenchConflict, get_workbench_instance
from ml_lab.representation.autoencoder_workbench import (
    DEFAULTS, form_values_from_spec, parse_autoencoder_settings, save_autoencoder_configuration,
)


def _context(tmp_path):
    path = tmp_path / 'vectors.csv'
    path.write_text('x,y,z\n1,2,3\n2,1,4\n3,4,5\n4,2,6\n5,6,7\n6,3,8\n7,8,9\n8,5,10\n', encoding='utf-8')
    store = ExperimentStore(tmp_path)
    from ml_lab.experiments.workbench import create_workbench_instance
    first = create_workbench_instance(store)
    selected = select_workbench_dataset(store, first['instance_id'], path,
                                        expected_generation=first['generation'],
                                        feature_columns=['x','y','z'])
    return path, store, selected


def _fields(context, **updates):
    values = dict(DEFAULTS)
    values.update(experiment_id=context['experiment_id'],
                  expected_generation=str(context['generation']),
                  expected_revision=str(context['experiment_revision']),
                  source_sha256=context['active_dataset']['sha256'])
    values.update(updates)
    return values


def test_typed_configs_reuse_existing_model_validation():
    method, settings = parse_autoencoder_settings(DEFAULTS, n_features=3)
    assert method == 'mlp_autoencoder'
    assert settings['model_config']['hidden_dims'] == (64, 32) or settings['model_config']['hidden_dims'] == [64, 32]
    assert settings['training_config']['restore_best'] is True
    method, settings = parse_autoencoder_settings({**DEFAULTS, 'method': 'transformer_autoencoder',
                                                   'checkpoint_selection': 'last'}, n_features=3)
    assert method == 'transformer_autoencoder'
    assert settings['model_config']['nhead'] == 4
    assert settings['training_config']['restore_best'] is False
    assert 'checkpoint_path' not in settings['training_config']


@pytest.mark.parametrize('changes', [
    {'hidden_dims': '64,-1'}, {'hidden_dims':'2,2,2,2,2'}, {'epochs': '0'},
    {'learning_rate': 'nan'}, {'checkpoint_selection': 'my_file.pt'},
    {'model_dim': '65','method': 'transformer_autoencoder'},
    {'method':'transformer_autoencoder','max_tokens':'1'},
    {'unknown_key':'x'}, {'training_config.checkpoint_path':'/tmp/evil'},
    {'device':'not_supported'}, {'deterministic':'yes'},
])
def test_rejects_unsupported_or_unsafe_controls(changes):
    with pytest.raises(ValueError):
        parse_autoencoder_settings({**DEFAULTS, **changes}, n_features=3)


def test_immutable_config_and_workbench_rebind(tmp_path):
    path, store, context = _context(tmp_path)
    before = path.read_bytes()
    old_revision = context['experiment_revision']
    fields = _fields(context, hidden_dims='24,12', epochs='3', checkpoint_selection='last')
    updated = save_autoencoder_configuration(store, context, fields)
    assert updated['generation'] == context['generation'] + 1
    assert updated['experiment_revision'] == old_revision + 1
    assert path.read_bytes() == before
    pinned = store.get_experiment(context['experiment_id'], revision=old_revision)
    assert pinned['model_spec'] == {}
    config = store.get_experiment(updated['experiment_id'], revision=updated['experiment_revision'])
    assert config['model_spec']['settings']['training_config']['restore_best'] is False
    assert form_values_from_spec(config['model_spec'])['checkpoint_selection'] == 'last'
    with pytest.raises(WorkbenchConflict):
        save_autoencoder_configuration(store, context, fields)
    # A semantically identical save creates no duplicate revision or generation.
    identical = save_autoencoder_configuration(store, updated, _fields(updated, hidden_dims='24,12', epochs='3', checkpoint_selection='last'))
    assert identical['generation'] == updated['generation']
    assert identical['experiment_revision'] == updated['experiment_revision']
    assert get_workbench_instance(store, updated['instance_id'])['status'] == 'ready'


def test_stale_file_and_target_selection_do_not_save(tmp_path):
    path, store, context = _context(tmp_path)
    wrong = _fields(context, source_sha256='f'*64)
    with pytest.raises(WorkbenchConflict):
        save_autoencoder_configuration(store, context, wrong)
    path.write_text(path.read_text() + '9,9,9\n', encoding='utf-8')
    stale = get_workbench_instance(store, context['instance_id'])
    assert stale['status'] == 'stale'
    with pytest.raises(ValueError, match='Select or repair'):
        save_autoencoder_configuration(store, stale, _fields(context))


def test_template_compiles():
    from pathlib import Path
    from jinja2 import Environment, FileSystemLoader
    path = Path(__file__).resolve().parents[1] / 'src/ml_lab/ui/templates'
    Environment(loader=FileSystemLoader(path)).get_template('ml_lab_ui/workbench_autoencoder.html')


def test_checkpoint_is_saved_only_inside_owning_project(tmp_path):
    torch = pytest.importorskip('torch')
    from ml_lab.experiments.models import configure_experiment_model
    from ml_lab.experiments.training import run_configured_experiment
    path, store, context = _context(tmp_path)
    configured = configure_experiment_model(store, context['experiment_id'],
                       expected_revision=context['experiment_revision'],
                       family='representation', method='mlp_autoencoder',
                       settings={'model_config': {'hidden_dims': (6, 3), 'latent_dim': 2},
                                 'training_config': {'epochs': 2, 'batch_size': 4,
                                                     'device': 'cpu', 'validation_fraction': .25}})
    result = run_configured_experiment(store, configured['experiment_id'], revision=configured['revision'])
    run = store.get_run(result['run_id'])
    assert run['status'] == 'succeeded'
    registered = store.artifacts.list_for_run(run['run_id'])
    checkpoints = [entry for entry in registered if entry['kind'] == 'checkpoint']
    assert len(checkpoints) == 1
    checkpoint = checkpoints[0]
    assert checkpoint['path'].startswith('analysis/ml_lab/training_runs/' + run['run_id'] + '/')
    assert store.artifacts.verify(checkpoint['artifact_id'])['status'] == 'valid'
    assert checkpoint['metadata']['selection'] == 'best_epoch'
    assert not (tmp_path / 'best_model.pt').exists()
    from ml_lab.ui.workbench_autoencoder_routes import _recent_reconstruction
    # Flask is optional: extracting a verified run report does not require it,
    # and should read the real execute_task envelope (task/result).
    extracted = _recent_reconstruction(store, context['active_dataset'], run)
    assert extracted is not None
    assert extracted['metrics']['mse'] >= 0
    assert extracted['preview'] is not None
    assert extracted['preview']['feature'] == 'x'
    assert len(extracted['preview']['rows']) == 8
    assert extracted['chart'] is not None
