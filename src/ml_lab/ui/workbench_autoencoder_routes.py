"""ML-UW-07 neural workbench: configure in-place, train through existing runs."""
from __future__ import annotations

import json
import math
from pathlib import Path

from ml_lab import __version__


def _recent_reconstruction(store, selection, run):
    """Return bounded, verified diagnostics for a completed owned run only."""
    if run['status'] != 'succeeded' or run['environment'].get('executor') != 'ml_lab.experiments.training@1':
        return None
    candidates = [item for item in store.artifacts.list_for_run(run['run_id'])
                  if item['role'] == 'output' and item['kind'] == 'result']
    if not candidates:
        return None
    artifact = candidates[0]
    if artifact['size_bytes'] > 3 * 1024 * 1024 or store.artifacts.verify(artifact['artifact_id'])['status'] != 'valid':
        return None
    path = store.project_root / artifact['path']
    # The registry verifies the project-local, non-symlinked file above.
    envelope = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(envelope, dict) or envelope.get('task') != 'representation':
        return None
    record = envelope.get('result')
    if not isinstance(record, dict):
        return None
    if record.get('method') not in ('mlp_autoencoder', 'transformer_autoencoder'):
        return None
    metrics = {k: float(v) for k, v in record.get('metrics', {}).items()
               if isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v)}
    history = []
    for item in record.get('history', [])[:1001]:
        if not isinstance(item, dict):
            continue
        epoch = item.get('epoch')
        if not isinstance(epoch, int) or epoch < 1:
            continue
        line = {'epoch': epoch}
        for key in ('train_mse', 'validation_mse'):
            val = item.get(key)
            if isinstance(val, (float, int)) and not isinstance(val, bool) and math.isfinite(val):
                line[key] = float(val)
        history.append(line)
    if len(history) > 120:
        stride = math.ceil((len(history) - 1) / 119)
        history = history[::stride] + ([history[-1]] if history[::stride][-1] != history[-1] else [])
    chart = None
    points = [entry for entry in history if 'train_mse' in entry]
    if len(points) >= 2:
        # SVG-ready coordinates, never executable page content.
        max_value = max([p['train_mse'] for p in points] + [p['validation_mse'] for p in history if 'validation_mse' in p])
        max_value = max(max_value, 1e-12)
        min_epoch, max_epoch = points[0]['epoch'], points[-1]['epoch']
        denom = max(max_epoch - min_epoch, 1)
        series = {}
        for key in ('train_mse', 'validation_mse'):
            series[key] = ' '.join(f"{36 + 522 * (p['epoch'] - min_epoch) / denom:.1f},{276 - 238 * p[key] / max_value:.1f}"
                                   for p in history if key in p)
        chart = {'series': series, 'max_value': max_value, 'last_epoch': max_epoch}
    preview = None
    reconstruction = record.get('reconstruction')
    if (isinstance(reconstruction, list) and selection['rows'] <= 500
            and len(selection['feature_columns']) <= 32
            and len(reconstruction) == selection['rows'] and reconstruction):
        from ml_lab.data import read_dataset_frame
        from ml_lab.experiments.datasets import verify_selection
        source = store.project_root / selection['path']
        if source.stat().st_size <= 8 * 1024 * 1024:
            frame = read_dataset_frame(source)
            values = frame.loc[:, selection['feature_columns']]
            shown = []
            for index in range(min(12, len(frame))):
                predicted = reconstruction[index]
                if not isinstance(predicted, list) or len(predicted) != len(selection['feature_columns']):
                    return {'metrics': metrics, 'history': history, 'chart': chart, 'preview': None}
                observed = float(values.iloc[index, 0])
                estimated = predicted[0]
                if not isinstance(estimated, (float, int)) or not math.isfinite(float(estimated)):
                    return {'metrics': metrics, 'history': history, 'chart': chart, 'preview': None}
                shown.append({'row': index + 1, 'original': observed, 'reconstructed': float(estimated)})
            # Recheck verified source after reading; don't display mismatched rows.
            verify_selection(store, selection)
            preview = {'feature': selection['feature_columns'][0], 'rows': shown}
    return {'metrics': metrics, 'history': history, 'chart': chart, 'preview': preview}


def install_autoencoder_routes(blueprint: Blueprint, *, project_root: Path, workbench_context) -> None:
    from flask import redirect, render_template, request, url_for
    from ml_lab.experiments import ExperimentStore, ExperimentStoreError
    from ml_lab.experiments.workbench import WorkbenchConflict
    from ml_lab.representation.autoencoder_workbench import (
        DEFAULTS, METHODS, form_values_from_spec, save_autoencoder_configuration,
    )

    @blueprint.route('/workbench/representation/autoencoder', methods=['GET', 'POST'])
    def workbench_autoencoder():
        store = ExperimentStore(project_root)
        context = workbench_context(store)
        error = None
        status = 200
        saved = request.args.get('saved') == '1'
        if request.method == 'POST':
            try:
                context = save_autoencoder_configuration(store, context, request.form)
                return redirect(url_for(request.blueprint + '.workbench_autoencoder', saved='1'))
            except WorkbenchConflict as exc:
                error, status = str(exc), 409
            except (ValueError, ExperimentStoreError) as exc:
                error, status = str(exc), 400
        model_spec = {}
        recent = []
        diagnostics = None
        if context['status'] == 'ready':
            experiment = store.get_experiment(context['experiment_id'], revision=context['experiment_revision'])
            model_spec = experiment['model_spec']
            for run in store.list_runs(context['experiment_id'], limit=30):
                if run['revision'] == context['experiment_revision'] and run['environment'].get('executor') == 'ml_lab.experiments.training@1':
                    recent.append(run)
                if len(recent) == 5:
                    break
            for run in recent:
                if run['status'] == 'succeeded':
                    try:
                        diagnostics = _recent_reconstruction(store, context['active_dataset'], run)
                    except (OSError, ValueError, KeyError, TypeError):
                        diagnostics = None
                    if diagnostics:
                        break
        fields = form_values_from_spec(model_spec)
        if request.method == 'POST' and error:
            fields.update({key: str(request.form[key]) for key in DEFAULTS if key in request.form})
        configured = (model_spec.get('family') == 'representation'
                      and model_spec.get('method') in METHODS)
        return render_template(
            'ml_lab_ui/workbench_autoencoder.html', version=__version__, context=context,
            fields=fields, configured=configured, saved=saved, error=error,
            recent=recent, diagnostics=diagnostics, methods=METHODS,
        ), status
