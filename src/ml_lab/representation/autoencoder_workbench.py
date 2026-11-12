"""ML-UW-07: typed, revision-safe neural configuration on the active workbench.

This adapter *only* configures an existing experiment. Training, validation,
checkpoint writing and result storage remain owned by ML_Lab's existing model
engine and configured-run executor. No Flask or torch import is needed here.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from ml_lab.experiments.models import configure_experiment_model
from ml_lab.experiments.store import ExperimentStore
from ml_lab.experiments.workbench import (
    WorkbenchConflict, activate_workbench_experiment, get_workbench_instance,
)

METHODS = ('mlp_autoencoder', 'transformer_autoencoder')
SCALINGS = ('none', 'standard', 'minmax', 'robust')
ACTIVATIONS = ('relu', 'gelu', 'tanh', 'leaky_relu')
OUTPUTS = ('none', 'sigmoid', 'tanh')
DEVICES = ('auto', 'cpu', 'cuda', 'mps')

DEFAULTS = {
    'method': 'mlp_autoencoder', 'hidden_dims': '64,32', 'latent_dim': '8',
    'activation': 'relu', 'output_activation': 'none', 'dropout': '0',
    'token_width': '1', 'model_dim': '64', 'nhead': '4', 'encoder_layers': '2',
    'decoder_layers': '2', 'feedforward_dim': '128', 'max_tokens': '1024',
    'transformer_activation': 'gelu', 'norm_first': '0',
    'epochs': '100', 'batch_size': '64', 'learning_rate': '0.001',
    'weight_decay': '0', 'validation_fraction': '0.1', 'patience': '15',
    'min_delta': '0.000001', 'random_state': '42', 'device': 'auto',
    'scaling': 'standard', 'checkpoint_selection': 'best', 'deterministic': '1',
}
CONTROL_KEYS = frozenset(DEFAULTS)
FORM_KEYS = CONTROL_KEYS | frozenset({
    'expected_generation', 'expected_revision', 'source_sha256', 'experiment_id',
})


def _integer(raw: str, name: str, low: int, high: int) -> int:
    if not raw.isdecimal():
        raise ValueError(f'{name} must be an integer from {low} to {high}')
    value = int(raw)
    if not low <= value <= high:
        raise ValueError(f'{name} must be from {low} to {high}')
    return value


def _number(raw: str, name: str, low: float, high: float, *, exclusive_low=False) -> float:
    try:
        value = float(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{name} must be numeric') from exc
    if not math.isfinite(value) or not (low < value if exclusive_low else low <= value) or value > high:
        raise ValueError(f'{name} must be finite and between {low} and {high}')
    return value


def _choice(raw: str, name: str, choices: tuple[str, ...]) -> str:
    if raw not in choices:
        raise ValueError(f'Unsupported {name}: {raw!r}')
    return raw


def _bool(raw: str, name: str) -> bool:
    return _choice(raw, name, ('0', '1')) == '1'


def parse_autoencoder_settings(fields: Mapping[str, str], *, n_features: int) -> tuple[str, dict[str, Any]]:
    """Parse the documented form without accepting arbitrary engine parameters."""
    if not isinstance(fields, Mapping) or set(fields) - FORM_KEYS:
        raise ValueError('Unknown autoencoder configuration fields')
    if type(n_features) is not int or n_features < 1:
        raise ValueError('A selected nonempty feature set is required')
    values = {name: fields.get(name, default) for name, default in DEFAULTS.items()}
    if any(not isinstance(value, str) for value in values.values()):
        raise ValueError('Configuration fields must be text')
    method = _choice(values['method'], 'autoencoder method', METHODS)
    latent = _integer(values['latent_dim'], 'latent_dim', 1, 512)
    dropout = _number(values['dropout'], 'dropout', 0, 0.8)
    output = _choice(values['output_activation'], 'output_activation', OUTPUTS)
    if method == 'mlp_autoencoder':
        raw_dims = values['hidden_dims'].split(',')
        if not 1 <= len(raw_dims) <= 4:
            raise ValueError('Specify between one and four MLP hidden layers')
        dims = tuple(_integer(part.strip(), 'hidden layer width', 1, 512) for part in raw_dims)
        model_config: dict[str, Any] = {
            'hidden_dims': dims, 'latent_dim': latent,
            'activation': _choice(values['activation'], 'activation', ACTIVATIONS),
            'output_activation': output, 'dropout': dropout,
        }
    else:
        model_dim = _integer(values['model_dim'], 'model_dim', 4, 512)
        nhead = _integer(values['nhead'], 'nhead', 1, 16)
        token_width = _integer(values['token_width'], 'token_width', 1, 128)
        max_tokens = _integer(values['max_tokens'], 'max_tokens', 1, 2048)
        if model_dim % nhead:
            raise ValueError('model_dim must be divisible by nhead')
        if (n_features + token_width - 1) // token_width > max_tokens:
            raise ValueError('Selected features exceed max_tokens for the configured token width')
        model_config = {
            'token_width': token_width, 'model_dim': model_dim, 'latent_dim': latent,
            'nhead': nhead, 'encoder_layers': _integer(values['encoder_layers'], 'encoder_layers', 1, 8),
            'decoder_layers': _integer(values['decoder_layers'], 'decoder_layers', 1, 8),
            'feedforward_dim': _integer(values['feedforward_dim'], 'feedforward_dim', 4, 2048),
            'dropout': dropout,
            'transformer_activation': _choice(values['transformer_activation'], 'transformer_activation', ('relu', 'gelu')),
            'output_activation': output, 'max_tokens': max_tokens,
            'norm_first': _bool(values['norm_first'], 'norm_first'),
        }
    training_config = {
        'epochs': _integer(values['epochs'], 'epochs', 1, 1000),
        'batch_size': _integer(values['batch_size'], 'batch_size', 1, 4096),
        'learning_rate': _number(values['learning_rate'], 'learning_rate', 0, 1.0, exclusive_low=True),
        'weight_decay': _number(values['weight_decay'], 'weight_decay', 0, 1.0),
        'validation_fraction': _number(values['validation_fraction'], 'validation_fraction', 0, 0.9),
        'patience': _integer(values['patience'], 'patience', 1, 1000),
        'min_delta': _number(values['min_delta'], 'min_delta', 0, 1.0),
        'random_state': _integer(values['random_state'], 'random_state', 0, 2**32 - 1),
        'device': _choice(values['device'], 'device', DEVICES),
        'scaling': _choice(values['scaling'], 'scaling', SCALINGS),
        'deterministic': _bool(values['deterministic'], 'deterministic'),
        'restore_best': _choice(values['checkpoint_selection'], 'checkpoint_selection', ('best', 'last')) == 'best',
    }
    # Validate using the *authoritative* experiment model/config service.
    from ml_lab.experiments.models import validate_model_configuration
    validated = validate_model_configuration(
        'representation', method, {'model_config': model_config, 'training_config': training_config},
        n_features=n_features,
    )
    return method, {'model_config': validated['model_config'], 'training_config': validated['training_config']}


def form_values_from_spec(spec: dict[str, Any]) -> dict[str, str]:
    result = dict(DEFAULTS)
    if spec.get('family') != 'representation' or spec.get('method') not in METHODS:
        return result
    result['method'] = spec['method']
    settings = spec.get('settings', {})
    for key, val in settings.get('model_config', {}).items():
        if key == 'hidden_dims':
            result[key] = ','.join(str(item) for item in val)
        elif key in result:
            result[key] = str(int(val)) if isinstance(val, bool) else str(val)
    for key, val in settings.get('training_config', {}).items():
        if key == 'restore_best':
            result['checkpoint_selection'] = 'best' if val else 'last'
        elif key in result:
            result[key] = str(int(val)) if isinstance(val, bool) else str(val)
    return result


def save_autoencoder_configuration(store: ExperimentStore, context: dict[str, Any], fields: Mapping[str, str]) -> dict[str, Any]:
    """Immutable model revision + explicit workbench rebind, both race-checked."""
    if context['status'] != 'ready' or not context['active_dataset']:
        raise ValueError('Select or repair the active dataset in Data first')
    if (fields.get('expected_generation', '') != str(context['generation'])
            or fields.get('expected_revision', '') != str(context['experiment_revision'])
            or fields.get('source_sha256') != context['active_dataset']['sha256']
            or fields.get('experiment_id') != context['experiment_id']):
        raise WorkbenchConflict('Dataset, selection, or experiment revision changed; reload')
    if context['active_dataset']['target_columns']:
        raise ValueError('Autoencoders require a feature-only dataset selection')
    current_context = get_workbench_instance(store, context['instance_id'])
    if (current_context['generation'] != context['generation']
            or current_context['active_dataset'] != context['active_dataset']):
        raise WorkbenchConflict('Workbench changed; reload')
    experiment = store.get_experiment(context['experiment_id'])
    if experiment['revision'] != context['experiment_revision']:
        raise WorkbenchConflict('Experiment revision changed; reactivate the latest revision')
    method, settings = parse_autoencoder_settings(fields, n_features=len(context['active_dataset']['feature_columns']))
    configured = configure_experiment_model(
        store, context['experiment_id'], family='representation', method=method,
        settings=settings, expected_revision=context['experiment_revision'],
    )
    if configured['revision'] == context['experiment_revision']:
        return get_workbench_instance(store, context['instance_id'])
    return activate_workbench_experiment(
        store, context['instance_id'], context['experiment_id'],
        expected_generation=context['generation'], revision=configured['revision'],
    )
