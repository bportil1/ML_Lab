"""ML-UW-04: feature exploration for a verified active workbench dataset.

No changes to the legacy UI blueprint or its profiling routes. The new path
works in standalone Flask and any PAH-mounted Blueprint with a project root.
"""
from __future__ import annotations

from pathlib import Path

from flask import Blueprint, jsonify, redirect, render_template, request, url_for

from ml_lab import __version__


def install_feature_routes(blueprint: Blueprint, *, project_root: Path, workbench_context) -> None:
    from ml_lab.experiments import ExperimentStore
    from ml_lab.experiments.datasets import _project_file
    from ml_lab.experiments.feature_roles import feature_roles, set_feature_role
    from ml_lab.experiments.workbench import get_workbench_instance
    from ml_lab.experiments.workbench import WorkbenchConflict, WorkbenchInstanceError
    from ml_lab.data.feature_explorer import explore_features, explore_cached_views
    from ml_lab.data.feature_dependency import dependency_matrix, METRIC_LABELS, MATRIX_LIMITS
    from ml_lab.experiments.feature_profile_cache import (
        feature_report_key, feature_baseline_key, load_feature_report, save_feature_report,
    )

    def build_report(store, context):
        if context['status'] != 'ready':
            raise WorkbenchInstanceError('Select or repair a dataset in Data before profiling')
        selected = context['active_dataset']
        source, _ = _project_file(store, selected['path'])
        overrides = feature_roles(store, context)
        names = selected['column_names']
        choice = request.values.get('selected')
        left = request.values.get('left')
        right = request.values.get('right')
        choice = choice if choice in names else (names[0] if names else None)
        left = left if left in names else (names[0] if names else None)
        right = right if right in names and right != left else next((n for n in names if n != left), None)
        metric = request.values.get('metric', 'normalized_mutual_information')
        if metric not in METRIC_LABELS:
            raise ValueError('Unknown dependency metric')
        try:
            limit = int(request.values.get('matrix_limit', '12'))
        except ValueError as exc:
            raise ValueError('Invalid matrix size') from exc
        if limit not in MATRIX_LIMITS:
            raise ValueError('Matrix size must be 8, 12 or 20')
        key = feature_report_key(context, selected=choice, left=left, right=right, overrides=overrides)
        refreshed = request.values.get('refresh') == '1'
        baseline_key = feature_baseline_key(context, overrides=overrides)
        report = None if refreshed else load_feature_report(store, context, key)
        cache_status = 'hit' if report is not None else 'computed'
        if report is None:
            # The baseline holds only computed summary/relationship JSON, not
            # sample rows. Each new plot reads a bounded deterministic sample.
            baseline = None if refreshed else load_feature_report(store, context, baseline_key)
            if baseline is not None and isinstance(baseline.get('columns'), list):
                report = explore_cached_views(
                    str(source), source_sha256=selected['sha256'], baseline=baseline,
                    selected=choice, left=left, right=right,
                )
                cache_status = 'reused'
            else:
                report = explore_features(
                    str(source), source_sha256=selected['sha256'], overrides=overrides,
                    selected=choice, left=left, right=right,
                )
            # Hash/provenance verification before cache access and again after
            # sampling: never store a snapshot of changing source bytes.
            confirmed = get_workbench_instance(store, context['instance_id'])
            if (confirmed['status'] != 'ready'
                    or confirmed['generation'] != context['generation']
                    or confirmed['experiment_id'] != context['experiment_id']
                    or confirmed['experiment_revision'] != context['experiment_revision']
                    or confirmed['active_dataset']['sha256'] != selected['sha256']
                    or confirmed['active_dataset']['path'] != selected['path']):
                raise WorkbenchConflict('Dataset or workbench changed during profiling; reload')
            stored = save_feature_report(
                store, context, key, report, preserve_cache_key=baseline_key,
            )
            # Only record a new baseline when this request actually computed
            # the full A2 profile. An explicit refresh rebuilds it.
            if cache_status == 'computed':
                save_feature_report(
                    store, context, baseline_key, report, preserve_cache_key=baseline_key,
                )
        else:
            stored = True
        # Filter/sort/presentation runs entirely on cached A2 numbers. Matrix
        # metric/size changes never redo statistical tests or scan the CSV.
        report['matrix'] = dependency_matrix(
            report['relationships'], report['columns'], metric=metric, limit=limit,
        )
        report['cache'] = {'status': cache_status, 'stored': stored}
        return report

    @blueprint.get('/workbench/feature-profiling')
    def workbench_feature_profiling():
        context = workbench_context()
        report = None
        error = None
        if context['status'] == 'ready' and request.args.get('inspect') == '1':
            try:
                report = build_report(ExperimentStore(project_root), context)
            except (OSError, ValueError) as exc:
                error = str(exc)
        return render_template(
            'ml_lab_ui/workbench_feature_profiling.html', version=__version__,
            context=context, report=report, error=error,
        ), (409 if error else 200)

    @blueprint.get('/workbench/feature-profiling/json')
    def workbench_feature_profiling_json():
        context = workbench_context()
        try:
            return jsonify(build_report(ExperimentStore(project_root), context))
        except (OSError, ValueError) as exc:
            return jsonify({'error': str(exc)}), 409

    @blueprint.post('/workbench/feature-profiling/role')
    def workbench_feature_role():
        context = workbench_context()
        generation = request.form.get('expected_generation', '')
        try:
            if not generation.isdecimal():
                raise WorkbenchConflict('Invalid generation; reload feature profiling')
            set_feature_role(
                ExperimentStore(project_root), context['instance_id'],
                column=request.form.get('column', ''),
                role=request.form.get('role', ''),
                source_sha256=request.form.get('source_sha256', ''),
                expected_generation=int(generation),
            )
        except (OSError, ValueError) as exc:
            return jsonify({'error': str(exc)}), 409 if isinstance(exc, WorkbenchConflict) else 400
        return redirect(url_for(request.blueprint + '.workbench_feature_profiling',
                                inspect='1', selected=request.form.get('column', '')))
