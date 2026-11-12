"""ML-UW-08 comparison UI; all results source-bound and display-only."""
from __future__ import annotations

from pathlib import Path

from flask import Blueprint, Response, render_template, request
from ml_lab import __version__


def install_comparison_routes(blueprint: Blueprint, *, project_root: Path, workbench_context) -> None:
    from ml_lab.data import read_dataset_frame
    from ml_lab.experiments import ExperimentStore
    from ml_lab.experiments.datasets import _project_file
    from ml_lab.experiments.workbench import WorkbenchConflict, get_workbench_instance
    from ml_lab.representation.comparison_workbench import (
        MAX_COMPARE_ROWS, MAX_METHODS, compare_representations,
        comparison_csv, comparison_latex, comparison_svg,
    )
    from ml_lab.representation.sklearn_catalog import MAX_FEATURES, METHODS, SCALING

    @blueprint.route('/workbench/comparison', methods=['GET', 'POST'])
    def workbench_comparison():
        store = ExperimentStore(project_root)
        context = workbench_context(store)
        selection = context['active_dataset']
        report, error, status = None, None, 200
        methods = request.form.getlist('methods') if request.method == 'POST' else ['pca', 'truncated_svd']
        scaling = request.form.get('scaling', 'minmax')
        raw_components = request.form.get('components', '2')
        include_neural = request.form.get('include_recorded_neural') == '1'
        if request.method == 'POST':
            try:
                if context['status'] != 'ready' or selection is None:
                    raise ValueError('Select or repair a dataset in Data before comparing representations')
                if (not request.form.get('expected_generation', '').isdecimal()
                        or int(request.form['expected_generation']) != context['generation']
                        or request.form.get('source_sha256') != selection['sha256']
                        or not request.form.get('expected_revision', '').isdecimal()
                        or int(request.form['expected_revision']) != context['experiment_revision']):
                    raise WorkbenchConflict('Dataset or experiment changed since the form was opened; reload')
                if selection['target_columns']:
                    raise ValueError('Representation comparison requires feature-only selection (no targets)')
                if (selection['rows'] < 8 or selection['rows'] > MAX_COMPARE_ROWS
                        or len(selection['feature_columns']) > MAX_FEATURES):
                    raise ValueError(f'Comparison requires 8–{MAX_COMPARE_ROWS} rows and at most {MAX_FEATURES} features')
                if not raw_components.isdecimal():
                    raise ValueError('Latent dimensions must be a positive integer')
                if request.form.get('include_recorded_neural') not in (None, '1'):
                    raise ValueError('Invalid neural comparison flag')
                if request.form.get('export') not in (None, 'csv', 'svg', 'tex'):
                    raise ValueError('Unsupported comparison export')
                source, _ = _project_file(store, selection['path'])
                if source.stat().st_size > 32 * 1024 * 1024:
                    raise ValueError('Comparison accepts files up to 32 MiB; create a smaller dataset explicitly')
                import numpy as np
                import pandas as pd
                frame = read_dataset_frame(source)
                try:
                    selected = frame.loc[:, selection['feature_columns']].apply(pd.to_numeric, errors='raise')
                    values = selected.to_numpy(dtype=float)
                except (ValueError, TypeError, KeyError) as exc:
                    raise ValueError('Selected feature columns must be numeric') from exc
                if not np.isfinite(values).all():
                    raise ValueError('Selected features must be finite and complete; clean data first')
                if values.shape != (selection['rows'], len(selection['feature_columns'])):
                    raise WorkbenchConflict('Dataset shape changed during comparison; reload')
                report = compare_representations(
                    selected, methods=methods, components=int(raw_components), scaling=scaling,
                    store=store, selection=selection, experiment_id=context['experiment_id'],
                    include_recorded_neural=include_neural,
                )
                # A long-running multi-estimator fit can overlap another
                # tab's dataset/revision change: drop these stale results.
                current = get_workbench_instance(store, context['instance_id'])
                if (current['status'] != 'ready' or current['generation'] != context['generation']
                        or current['experiment_id'] != context['experiment_id']
                        or current['experiment_revision'] != context['experiment_revision']
                        or current['active_dataset'] != selection):
                    raise WorkbenchConflict('Dataset or experiment changed during comparison; results discarded')
                export = request.form.get('export')
                if export in ('csv', 'svg', 'tex'):
                    payload, mimetype = {
                        'csv': (comparison_csv(report), 'text/csv'),
                        'svg': (comparison_svg(report), 'image/svg+xml'),
                        'tex': (comparison_latex(report), 'text/plain; charset=utf-8'),
                    }[export]
                    return Response(payload, mimetype=mimetype,
                        headers={'Content-Disposition': f'attachment; filename="ml_lab_representation_comparison.{export}"',
                                 'Cache-Control': 'no-store'})
            except WorkbenchConflict as exc:
                report, error, status = None, str(exc), 409
            except (OSError, KeyError, ValueError) as exc:
                report, error, status = None, str(exc), 400
        return render_template(
            'ml_lab_ui/workbench_comparison.html', version=__version__,
            context=context, methods=METHODS, scalings=SCALING,
            chosen_methods=methods, scaling=scaling, raw_components=raw_components,
            include_neural=include_neural, max_methods=MAX_METHODS,
            max_rows=MAX_COMPARE_ROWS, max_features=MAX_FEATURES,
            report=report, error=error,
        ), status
