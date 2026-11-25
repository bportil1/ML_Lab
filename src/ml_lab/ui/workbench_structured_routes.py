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
from .structured_workbench import LOCAL_FILE_KINDS, read_source, analyze, _strict_json
from .structured_source_intake import browse_project_files
from .structured_graph_view import make_shared_graph_payload
from .structured_hia_view import hierarchy_overview, inspect_workspace_scope
from .structured_sandbox import analyze_sandbox
from .structured_network_view import association_view
from ml_lab.structured import analyze_observed_network
from ml_lab.structured import source_sha256, atomic_node_statistics, analyze_neighbor_relationships, analyze_hierarchical_branch

def _shared_graph_available() -> bool:
    """Older PAH_Shared installations simply do not offer the opt-in view."""
    try:
        from importlib.resources import files
        root = files('pah_shared').joinpath('static')
        return all(root.joinpath(name).is_file() for name in
                   ('interactive_graph_view.mjs', 'interactive_graph_view.css'))
    except (ImportError, ModuleNotFoundError, OSError, TypeError):
        return False


def _shared_graph_asset_revision() -> str | None:
    """Fingerprint the *installed* PAH_Shared renderer used by this process.

    A fixed /pah-shared URL previously allowed Firefox's cached ES module to
    survive a same-version package refresh. Version the import by real bytes,
    not by a package version or Git revision from another environment.
    """
    try:
        from importlib.resources import files
        root = files('pah_shared').joinpath('static')
        content = root.joinpath('interactive_graph_view.mjs').read_bytes()
        return hashlib.sha256(content).hexdigest()[:20]
    except (ImportError, ModuleNotFoundError, OSError, TypeError):
        return None


def _structured_controller_revision() -> str:
    """Invalidate local script caching when any graph controller changes."""
    base = Path(__file__).resolve().parent / 'static'
    digest = hashlib.sha256()
    for name in ('ml_lab_structured_graph.js', 'ml_lab_hia_workspace.js',
                 'ml_lab_hia_atomic.js', 'ml_lab_hia_neighbors.js', 'ml_lab_hia_branch.js', 'ml_lab_structured_sandbox.js', 'ml_lab_structured_network.js'):
        digest.update(name.encode('utf-8'))
        digest.update(base.joinpath(name).read_bytes())
    return digest.hexdigest()[:20]


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
            payload = make_shared_graph_payload(source, fingerprint=selection['sha256'])
            payload['binding_contract'] = {
                'schema': 'ml-lab.observation-binding-plan@1',
                'source_digest': source_sha256(source),
                'observation_unit': source.observation_unit,
                'observed_columns': sorted(source.observations[0]['values']) if source.observations else [],
            }
            return jsonify(payload)
        except (StructuredContractError, ValueError, TypeError, OSError, RecursionError) as exc:
            return jsonify({'error': str(exc)}), 409

    @blueprint.get('/workbench/structured/hierarchy')
    def workbench_structured_hierarchy():
        """HIA-03 level inventory from a freshly verified project-local source."""
        try:
            raw = request.cookies.get(cookie)
            selection = _strict_json(raw) if raw and len(raw) <= 1024 else None
            if not isinstance(selection, dict) or set(selection) != {'path', 'kind', 'sha256'}:
                raise StructuredContractError('select a verified structured artifact first')
            source, _ = read_source(project_root, selection['path'], selection['kind'], selection['sha256'])
            return jsonify(hierarchy_overview(source, source_revision='sha256:' + selection['sha256']))
        except (StructuredContractError, ValueError, TypeError, OSError, RecursionError) as exc:
            return jsonify({'error': str(exc)}), 409

    @blueprint.post('/workbench/structured/hierarchy/inspect')
    def workbench_structured_hierarchy_inspect():
        """Explicit SHA-verified structural selection; no writes or estimators."""
        if request.content_length is None or request.content_length > 8192:
            return jsonify({'error': 'HIA scope request exceeds 8 KB'}), 413
        if not request.is_json or request.headers.get('X-ML-Lab-HIA') != 'explicit-inspection':
            return jsonify({'error': 'explicit same-origin JSON inspection required'}), 400
        try:
            raw = request.cookies.get(cookie)
            selection = _strict_json(raw) if raw and len(raw) <= 1024 else None
            if not isinstance(selection, dict) or set(selection) != {'path', 'kind', 'sha256'}:
                raise StructuredContractError('select a verified structured artifact first')
            data = _strict_json(request.get_data(as_text=True))
            if not isinstance(data, dict) or set(data) != {'source_revision', 'scope', 'offset'}:
                raise StructuredContractError('HIA request requires source_revision, scope, offset')
            revision = 'sha256:' + selection['sha256']
            if data['source_revision'] != revision:
                raise StructuredContractError('HIA source revision mismatch: reopen the source')
            source, _ = read_source(project_root, selection['path'], selection['kind'], selection['sha256'])
            return jsonify(inspect_workspace_scope(source, source_revision=revision,
                                                   scope_input=data['scope'], offset=data['offset']))
        except (StructuredContractError, ValueError, TypeError, OSError, RecursionError) as exc:
            return jsonify({'error': str(exc)}), 409

    @blueprint.post('/workbench/structured/hierarchy/atomic')
    def workbench_structured_hierarchy_atomic():
        """Opt-in node-to-column SU-05 binding; source-pinned, read-only stats."""
        if request.content_length is None or request.content_length > 8192:
            return jsonify({'error': 'atomic statistics request exceeds 8 KB'}), 413
        if (not request.is_json or
                request.headers.get('X-ML-Lab-HIA') != 'explicit-atomic-statistics'):
            return jsonify({'error': 'explicit same-origin atomic analysis required'}), 400
        try:
            raw = request.cookies.get(cookie)
            selection = _strict_json(raw) if raw and len(raw) <= 1024 else None
            if not isinstance(selection, dict) or set(selection) != {'path', 'kind', 'sha256'}:
                raise StructuredContractError('select a verified structured artifact first')
            data = _strict_json(request.get_data(as_text=True))
            required = {'source_revision', 'node_id', 'variable_id',
                        'occurrence_policy', 'bins', 'min_samples'}
            if not isinstance(data, dict) or set(data) != required:
                raise StructuredContractError('invalid HIA atomic statistics request fields')
            revision = 'sha256:' + selection['sha256']
            if data['source_revision'] != revision:
                raise StructuredContractError('HIA source revision mismatch: reopen the source')
            source, _ = read_source(project_root, selection['path'], selection['kind'], selection['sha256'])
            output = atomic_node_statistics(source, node_id=data['node_id'],
                        variable_id=data['variable_id'], occurrence_policy=data['occurrence_policy'],
                        bins=data['bins'], min_samples=data['min_samples'])
            return jsonify({'source_revision': revision, 'report': output})
        except (StructuredContractError, ValueError, TypeError, OSError, RecursionError) as exc:
            return jsonify({'error': str(exc)}), 409

    @blueprint.post('/workbench/structured/hierarchy/neighbors')
    def workbench_structured_hierarchy_neighbors():
        """Explicit source-pinned, read-only HIA-05 neighbor estimation."""
        if request.content_length is None or request.content_length > 65536:
            return jsonify({'error': 'neighbor analysis request exceeds 64 KB'}), 413
        if (not request.is_json or
                request.headers.get('X-ML-Lab-HIA') != 'explicit-neighbor-statistics'):
            return jsonify({'error': 'explicit same-origin neighbor analysis required'}), 400
        try:
            raw = request.cookies.get(cookie)
            selection = _strict_json(raw) if raw and len(raw) <= 1024 else None
            if not isinstance(selection, dict) or set(selection) != {'path', 'kind', 'sha256'}:
                raise StructuredContractError('select a verified structured artifact first')
            data = _strict_json(request.get_data(as_text=True))
            required = {'source_revision', 'node_id', 'variable_id', 'neighbors',
                        'conditioning', 'occurrence_policy', 'bins',
                        'min_samples', 'min_stratum_samples'}
            if not isinstance(data, dict) or set(data) != required:
                raise StructuredContractError('invalid HIA neighbor analysis request fields')
            revision = 'sha256:' + selection['sha256']
            if data['source_revision'] != revision:
                raise StructuredContractError('HIA source revision mismatch: reopen the source')
            source, _ = read_source(project_root, selection['path'], selection['kind'], selection['sha256'])
            output = analyze_neighbor_relationships(
                source, node_id=data['node_id'], variable_id=data['variable_id'],
                neighbors=data['neighbors'], conditioning=data['conditioning'],
                occurrence_policy=data['occurrence_policy'], bins=data['bins'],
                min_samples=data['min_samples'], min_stratum_samples=data['min_stratum_samples'],
            )
            return jsonify({'source_revision': revision, 'report': output})
        except (StructuredContractError, ValueError, TypeError, OSError, RecursionError) as exc:
            return jsonify({'error': str(exc)}), 409

    @blueprint.post('/workbench/structured/hierarchy/branch-information')
    def workbench_structured_hierarchy_branch_information():
        """Explicit SHA-pinned branch-level estimates, read-only and opt-in."""
        if request.content_length is None or request.content_length > 131072:
            return jsonify({'error': 'HIA branch request exceeds 128 KB'}), 413
        if (not request.is_json or
                request.headers.get('X-ML-Lab-HIA') != 'explicit-branch-information'):
            return jsonify({'error': 'explicit same-origin branch analysis required'}), 400
        try:
            raw = request.cookies.get(cookie)
            selection = _strict_json(raw) if raw and len(raw) <= 1024 else None
            if not isinstance(selection, dict) or set(selection) != {'path', 'kind', 'sha256'}:
                raise StructuredContractError('select a verified structured artifact first')
            data = _strict_json(request.get_data(as_text=True))
            required = {'source_revision', 'branch_root_id', 'mappings', 'context',
                        'occurrence_policy', 'bins', 'min_samples', 'min_stratum_samples'}
            if not isinstance(data, dict) or set(data) != required:
                raise StructuredContractError('invalid HIA branch analysis request fields')
            revision = 'sha256:' + selection['sha256']
            if data['source_revision'] != revision:
                raise StructuredContractError('HIA source revision mismatch: reopen the source')
            source, _ = read_source(project_root, selection['path'], selection['kind'], selection['sha256'])
            report = analyze_hierarchical_branch(
                source, branch_root_id=data['branch_root_id'], mappings=data['mappings'],
                context=data['context'], occurrence_policy=data['occurrence_policy'],
                bins=data['bins'], min_samples=data['min_samples'],
                min_stratum_samples=data['min_stratum_samples'],
            )
            return jsonify({'source_revision': revision, 'report': report})
        except (StructuredContractError, ValueError, TypeError, OSError, RecursionError) as exc:
            return jsonify({'error': str(exc)}), 409

    @blueprint.post('/workbench/structured/sandbox')
    def workbench_structured_sandbox():
        """Explicit analysis only; same origin, read-only, SHA-pinned per request."""
        if request.content_length is None or request.content_length > 96000:
            return jsonify({'error': 'sandbox request exceeds 96 KB'}), 413
        if (not request.is_json or
                request.headers.get('X-ML-Lab-Sandbox') != 'explicit-analysis'):
            return jsonify({'error': 'explicit same-origin JSON analysis request required'}), 400
        try:
            raw = request.cookies.get(cookie)
            if not raw or len(raw) > 1024:
                raise StructuredContractError('select a verified structured artifact first')
            selection = _strict_json(raw)
            if not isinstance(selection, dict) or set(selection) != {'path', 'kind', 'sha256'}:
                raise StructuredContractError('invalid selection: select the artifact again')
            source, _ = read_source(project_root, selection['path'], selection['kind'], selection['sha256'])
            payload = _strict_json(request.get_data(as_text=True))
            if not isinstance(payload, dict):
                raise StructuredContractError('sandbox request must be a JSON object')
            return jsonify(analyze_sandbox(source, source_revision='sha256:' + selection['sha256'],
                                           request_data=payload))
        except (StructuredContractError, ValueError, TypeError, OSError, RecursionError) as exc:
            return jsonify({'error': str(exc)}), 409

    @blueprint.post('/workbench/structured/network')
    def workbench_structured_network():
        """Explicit, SHA-pinned all-variable empirical analysis; no job or write."""
        if request.content_length is None or request.content_length > 32000:
            return jsonify({'error': 'network analysis request exceeds 32 KB'}), 413
        if (not request.is_json or
                request.headers.get('X-ML-Lab-Network') != 'explicit-analysis'):
            return jsonify({'error': 'explicit same-origin JSON network request required'}), 400
        try:
            raw = request.cookies.get(cookie)
            if not raw or len(raw) > 1024:
                raise StructuredContractError('select a verified structured source first')
            selection = _strict_json(raw)
            if not isinstance(selection, dict) or set(selection) != {'path', 'kind', 'sha256'}:
                raise StructuredContractError('invalid selection: reopen the source')
            payload = _strict_json(request.get_data(as_text=True))
            allowed = {'source_revision', 'variables', 'conditioning', 'bins',
                       'missing_policy', 'min_samples', 'min_stratum_samples'}
            if not isinstance(payload, dict) or set(payload) - allowed:
                raise StructuredContractError('unsupported network analysis request fields')
            source_revision = 'sha256:' + selection['sha256']
            if payload.get('source_revision') != source_revision:
                raise StructuredContractError('source revision mismatch: reopen the source')
            source, _ = read_source(project_root, selection['path'],
                                    selection['kind'], selection['sha256'])
            report = analyze_observed_network(
                source, variables=tuple(payload.get('variables', []))
                  if isinstance(payload.get('variables', []), list) else payload.get('variables'),
                conditioning=tuple(payload.get('conditioning', []))
                  if isinstance(payload.get('conditioning', []), list) else payload.get('conditioning'),
                bins=payload.get('bins', {}),
                missing_policy=payload.get('missing_policy', 'complete_case'),
                min_samples=payload.get('min_samples', 5),
                min_stratum_samples=payload.get('min_stratum_samples', 2),
            )
            return jsonify({'source_revision': source_revision, 'report': report,
                            'view': association_view(report, source_revision=source_revision)})
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
        discovered = ()
        try:
            if action == 'browse' or request.args.get('browse') == '1':
                discovered = browse_project_files(project_root)
            if action == 'clear':
                g.ml_lab_structured_selection = False
            else:
                if action == 'select':
                    source, metadata = read_source(project_root, request.form.get('path', ''),
                                                   request.form.get('kind', 'auto'))
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
                if action not in ('', 'select', 'browse', 'clear'):
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
                               source=metadata, kinds=LOCAL_FILE_KINDS, discovered=discovered, report=report, chart=chart,
                               report_json=(json.dumps(report, indent=2, ensure_ascii=False)
                                            if report is not None else ''),
                               error=error, values=submitted, model_text=model_text,
                               shared_graph_available=_shared_graph_available(),
                               shared_graph_revision=_shared_graph_asset_revision(),
                               graph_controller_revision=_structured_controller_revision()), status
