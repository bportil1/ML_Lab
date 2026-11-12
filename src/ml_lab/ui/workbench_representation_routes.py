"""ML-UW-06 representation view, using only the verified active dataset.

The view is an ephemeral scientific explorer. It does not publish an experiment,
change feature roles, alter the source, or persist a fitted sklearn estimator.
"""
from __future__ import annotations

from pathlib import Path

from flask import Blueprint, render_template, request

from ml_lab import __version__


def install_representation_routes(blueprint: Blueprint, *, project_root: Path, workbench_context) -> None:
    from ml_lab.experiments import ExperimentStore
    from ml_lab.experiments.datasets import _project_file
    from ml_lab.experiments.workbench import get_workbench_instance, WorkbenchConflict
    from ml_lab.representation.sklearn_catalog import (
        MAX_FEATURES, MAX_ROWS, METHODS, SCALING, explore_representation, method_spec,
    )

    @blueprint.route('/workbench/representation', methods=['GET', 'POST'])
    def workbench_representation():
        store = ExperimentStore(project_root)
        context = workbench_context(store)
        selection = context['active_dataset']
        form = request.form
        method = form.get('method', 'pca')
        scaling = form.get('scaling', 'standard')
        raw_components = form.get('components', '2')
        raw_x = form.get('axis_x', '1')
        raw_y = form.get('axis_y', '2')
        reconstruct = form.get('reconstruct') == '1'
        report = None
        error = None
        status = 200
        if request.method == 'POST':
            try:
                if context['status'] != 'ready':
                    raise ValueError('Select or repair a dataset in Data before fitting a representation')
                if (not form.get('expected_generation', '').isdecimal()
                        or int(form['expected_generation']) != context['generation']
                        or form.get('source_sha256') != selection['sha256']):
                    raise WorkbenchConflict('Dataset or workbench changed since the form was opened; reload')
                spec = method_spec(method)
                if (not raw_components.isdecimal() or not raw_x.isdecimal() or not raw_y.isdecimal()):
                    raise ValueError('Components and plot axes must be positive integers')
                if form.get('reconstruct') not in {None, '1'}:
                    raise ValueError('Invalid reconstruction setting')
                if reconstruct and not spec.can_inverse_transform:
                    raise ValueError(f'{spec.title} does not support inverse transformation')
                if selection['target_columns']:
                    raise ValueError('Representation requires a dataset selection without targets; change the selection in Data')
                if (selection['rows'] > spec.max_rows or len(selection['feature_columns']) > MAX_FEATURES
                        or selection['rows'] < 2):
                    raise ValueError(f'{spec.title} supports 2–{spec.max_rows} rows and at most {MAX_FEATURES} selected features; choose a smaller dataset explicitly')
                source, _ = _project_file(store, selection['path'])
                if source.stat().st_size > 32 * 1024 * 1024:
                    raise ValueError('Representation explorer accepts source files up to 32 MiB; create a smaller derived dataset explicitly')
                # The selected revision is already fingerprinted in workbench_context.
                # Read through Data Lab's existing, provenance-aware parser.
                from ml_lab.data import read_dataset_frame
                import numpy as np
                import pandas as pd
                frame = read_dataset_frame(source)
                selected = frame.loc[:, selection['feature_columns']]
                try:
                    values = selected.apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)
                except (ValueError, TypeError) as exc:
                    raise ValueError('Selected feature columns must be numeric; choose numeric columns in Data') from exc
                if not np.isfinite(values).all():
                    raise ValueError('Selected features must be finite and complete; clean missing values in Data first')
                if values.shape != (selection['rows'], len(selection['feature_columns'])):
                    raise WorkbenchConflict('Dataset shape changed during representation; reload')
                report = explore_representation(
                    pd.DataFrame(values, columns=selection['feature_columns']),
                    method=method, components=int(raw_components), scaling=scaling,
                    reconstruct=reconstruct, axis_x=int(raw_x) - 1, axis_y=int(raw_y) - 1,
                )
                current = get_workbench_instance(store, context['instance_id'])
                if (current['status'] != 'ready' or current['generation'] != context['generation']
                        or current['experiment_id'] != context['experiment_id']
                        or current['experiment_revision'] != context['experiment_revision']
                        or current['active_dataset'] != selection):
                    raise WorkbenchConflict('Dataset or workbench changed during representation; results discarded')
            except WorkbenchConflict as exc:
                error = str(exc)
                report = None
                status = 409
            except (OSError, KeyError, ValueError) as exc:
                error = str(exc)
                report = None
                status = 400
        return render_template(
            'ml_lab_ui/workbench_representation.html', version=__version__,
            context=context, methods=METHODS, scalings=SCALING, report=report,
            method=method, scaling=scaling, raw_components=raw_components,
            raw_x=raw_x, raw_y=raw_y, reconstruct=reconstruct,
            max_rows=MAX_ROWS, max_features=MAX_FEATURES, error=error,
        ), status
