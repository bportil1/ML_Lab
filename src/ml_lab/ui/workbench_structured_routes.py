"""Instance-scoped, mount-aware structured-uncertainty UI.

Explicit project-local read only; no source data in cookies, no automatic
inference/training, no cross-project databases or new schema migrations.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from flask import Blueprint, g, jsonify, request, render_template, url_for
from ml_lab import __version__
from ml_lab.structured import StructuredContractError
from .structured_workbench import KINDS, read_source, analyze
from .structured_graph_view import make_shared_graph_payload

def _shared_graph_available() -> bool:
    """Older PAH_Shared installations simply do not offer the opt-in view."""
    try:
        from importlib.resources import files
        root = files('pah_shared').joinpath('static')
        return all(root.joinpath(name).is_file() for name in
                   ('interactive_graph_view.mjs', 'interactive_graph_view.css'))
    except (ImportError, ModuleNotFoundError, OSError, TypeError):
        return False


SECTIONS = (('data', 'Data'), ('profiling', 'Feature Profiling'),
            ('representation', 'Representation'), ('comparison', 'Comparison'),
            ('tools', 'General ML Tools'))


def install_structured_routes(blueprint: Blueprint, *, name: str, project_root: Path):
    cookie = 'ml_lab_structured_' + hashlib.sha256(f'{name}:{project_root}'.encode()).hexdigest()[:20]

    @blueprint.after_request
    def selection_cookie(response):
        selection = getattr(g, 'ml_lab_structured_selection', None)
        if selection is not None:
            path = url_for(request.blueprint + '.index')
            if selection is False:
                response.delete_cookie(cookie, path=path)
            else:
                response.set_cookie(cookie, json.dumps(selection, separators=(',', ':')),
                                    path=path, httponly=True, samesite='Lax')
        return response

    @blueprint.get('/workbench/structured/graph')
    def workbench_structured_graph():
        """Explicit read-only source view; fresh SHA-verified selection every call."""
        try:
            raw = request.cookies.get(cookie)
            if not raw or len(raw) > 1024:
                raise StructuredContractError('select a verified structured artifact first')
            selection = json.loads(raw)
            if not isinstance(selection, dict) or set(selection) != {'path', 'kind', 'sha256'}:
                raise StructuredContractError('invalid selection: select the artifact again')
            source, _ = read_source(project_root, selection['path'], selection['kind'], selection['sha256'])
            return jsonify(make_shared_graph_payload(source, fingerprint=selection['sha256']))
        except (StructuredContractError, ValueError, TypeError, OSError, RecursionError) as exc:
            return jsonify({'error': str(exc)}), 409

    @blueprint.route('/workbench/structured', methods=['GET', 'POST'])
    def workbench_structured():
        if request.method == 'POST' and (request.content_length is None or request.content_length > 180000):
            return jsonify({'error': 'form exceeds 180 KB interactive cap'}), 413
        submitted = request.form if request.method == 'POST' else request.args
        section = submitted.get('section', 'data')
        if section not in dict(SECTIONS):
            return jsonify({'error': 'invalid workbench section'}), 400
        action = request.form.get('action', '') if request.method == 'POST' else ''
        source = report = artifact = metadata = selection = None
        error, model_text, status = None, '', 200
        try:
            if action == 'clear':
                g.ml_lab_structured_selection = False
            else:
                if action == 'select':
                    source, metadata = read_source(project_root, request.form.get('path', ''),
                                                   request.form.get('kind', 'structured'))
                    selection = {key: metadata[key] for key in ('path', 'kind', 'sha256')}
                    g.ml_lab_structured_selection = selection
                else:
                    raw = request.cookies.get(cookie)
                    if raw:
                        selection = json.loads(raw)
                        if (not isinstance(selection, dict) or
                                set(selection) != {'path', 'kind', 'sha256'}):
                            raise StructuredContractError('invalid selection: clear it and select again')
                        source, metadata = read_source(project_root, selection['path'],
                                                       selection['kind'], selection['sha256'])
                if action not in ('', 'select'):
                    if source is None:
                        raise StructuredContractError('select a verified local structured artifact first')
                    report, artifact = analyze(source, section, action, request.form)
                    if request.form.get('download') in ('report', 'artifact'):
                        output = artifact if request.form['download'] == 'artifact' and artifact is not None else report
                        response = jsonify(output)
                        response.headers['Content-Disposition'] = 'attachment; filename="structured-analysis.json"'
                        return response
                    if action in ('draft', 'fit'):
                        model_text = json.dumps(artifact, indent=2, ensure_ascii=False)
        except (StructuredContractError, ValueError, TypeError, OSError, RecursionError) as exc:
            error, status = str(exc), 400
            if 'changed' in error or 'invalid selection' in error:
                source, metadata = None, None
        if not model_text and request.method == 'POST':
            model_text = request.form.get('model_json', '')[:120000]
        chart = []
        if isinstance(report, dict) and isinstance(report.get('candidates'), list):
            for item in report['candidates']:
                retention = item['retention']
                chart.append({'label': item['label'],
                              'nodes': round(retention['node_identity_fraction'] * 100, 1),
                              'edges': (round(retention['edge_identity_fraction'] * 100, 1)
                                        if retention['edge_identity_fraction'] is not None else None),
                              'loss': item['predictive_holdout']['holdout_log_loss_nats'],
                              'status': item['predictive_holdout']['status']})
        return render_template('ml_lab_ui/workbench_structured.html', version=__version__,
                               section=section, sections=SECTIONS, selection=selection,
                               source=metadata, kinds=KINDS, report=report, chart=chart,
                               report_json=(json.dumps(report, indent=2, ensure_ascii=False)
                                            if report is not None else ''),
                               error=error, values=submitted, model_text=model_text,
                               shared_graph_available=_shared_graph_available()), status
