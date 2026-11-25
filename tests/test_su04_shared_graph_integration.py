"""SU-04: verify a real PAH_Shared contract, identity and read-only boundaries."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest
from jinja2 import Environment, FileSystemLoader

from ml_lab.structured import StructuredContractError, StructuredObject
from ml_lab.ui.structured_graph_view import make_shared_graph_view, make_shared_graph_payload
from ml_lab.ui.structured_workbench import read_source

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / 'tests/fixtures/structured_uncertainty'
UI = ROOT / 'src/ml_lab/ui'
SHARED = Path(__file__).resolve().parents[3] / 'shared/PAH_Shared-main/src/pah_shared/static/interactive_graph_view.mjs'


def source(name='xml_tree.json'):
    return StructuredObject.from_json((TESTS / name).read_text(encoding='utf-8'))


def project(name='xml_tree.json'):
    return make_shared_graph_view(source(name), fingerprint='a' * 64)


def test_original_ids_are_unchanged_and_only_display_fields_are_projected():
    original = source()
    view = project()
    assert view['schema'] == 'pah-shared.graph-view@1'
    assert view['graph_role'] == 'source_structure'
    assert view['source_revision'] == 'sha256:' + 'a'*64
    assert {n['id'] for n in view['nodes']} == {n['id'] for n in original.nodes}
    assert {e['id'] for e in view['edges']} == {e['id'] for e in original.edges}
    assert [(x['source'], x['target']) for x in view['edges']] == [
        (x['source_id'], x['target_id']) for x in original.edges]
    assert all(e['directed'] and e['traceable'] for e in view['edges'])
    assert view['display']['coverage']['scope'] == 'exhaustive'
    assert view['overlays'] == {'nodes': {}, 'edges': {}}
    assert original.to_dict() == source().to_dict()


def test_dag_has_no_inferred_source_hierarchy_trace():
    view = project('shared_child_dag.json')
    assert view['display']['layout'] == 'layers'
    assert all(not e['traceable'] for e in view['edges'])


def test_cyclic_directed_graph_uses_visual_circle_without_scientific_claim():
    view = project('cyclic_network.json')
    assert view['display']['layout'] == 'circular'
    assert all(not e['traceable'] for e in view['edges'])
    assert all(e['directed'] for e in view['edges'])


def test_undirected_network_has_explicit_undirected_edges():
    raw = source().to_dict()
    raw['topology'] = {'kind': 'undirected_graph'}
    raw['nodes'] = raw['nodes'][:2]
    raw['edges'] = [{'id': 'link', 'source_id': raw['nodes'][0]['id'],
                    'target_id': raw['nodes'][1]['id'], 'kind': 'link'}]
    obj = StructuredObject.from_dict(raw)
    view = make_shared_graph_view(obj, fingerprint='0'*64)
    assert all(not edge['directed'] and not edge['traceable'] for edge in view['edges'])
    assert view['display']['layout'] == 'circular'


def test_fingerprint_required_and_size_limits_are_explicit():
    with pytest.raises(StructuredContractError, match='SHA-256'):
        make_shared_graph_view(source(), fingerprint='fake')
    import ml_lab.ui.structured_graph_view as bridge
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(bridge, 'MAX_GRAPH_NODES', 2)
        with pytest.raises(StructuredContractError, match='exceeds shared viewer'):
            project()


def test_inspector_keeps_provenance_and_does_not_invent_scientific_values():
    obj = source()
    payload = make_shared_graph_payload(obj, fingerprint='a'*64)
    for node in obj.nodes:
        assert payload['inspector']['nodes'][node['id']] == dict(node)
    assert 'no variance' in payload['notice'].lower()
    assert not payload['view']['overlays']['nodes']


def test_read_source_provenance_is_the_display_revision(tmp_path):
    input_file = TESTS / 'xml_tree.json'
    (tmp_path/'source.json').write_bytes(input_file.read_bytes())
    obj,meta = read_source(tmp_path, 'source.json', 'structured')
    view = make_shared_graph_view(obj, fingerprint=meta['sha256'])
    assert view['source_revision'] == 'sha256:' + meta['sha256']
    (tmp_path/'source.json').write_text('{}')
    with pytest.raises(StructuredContractError, match='changed'):
        read_source(tmp_path, meta['path'], meta['kind'], meta['sha256'])


def test_template_contains_opt_in_controls_and_original_tabs():
    env = Environment(loader=FileSystemLoader(UI/'templates'))
    env.get_template('ml_lab_ui/workbench_structured.html')
    text = (UI/'templates/ml_lab_ui/workbench_structured.html').read_text()
    for fragment in ('data-graph-action="open"', 'data-graph-payload-url',
                     'data-graph-module-url', 'data-graph-inspector',
                     'section == \'data\'', 'data-graph-action="close"'):
        assert fragment in text
    # The main five-tab labels are provided by SECTIONS and the existing base
    # template; SU-04 must not duplicate or replace those navigation controls.
    routes_text = (UI/'workbench_structured_routes.py').read_text()
    assert "('data', 'Data')" in routes_text
    assert "('profiling', 'Feature Profiling')" in routes_text
    assert "('representation', 'Representation')" in routes_text
    assert "('comparison', 'Comparison')" in routes_text
    assert "('tools', 'General ML Tools')" in routes_text
    assert 'No graph loaded. Opening does not run statistical analysis.' in text


def test_js_is_valid_and_links_to_actual_paH_shared_api():
    js = (UI/'static/ml_lab_structured_graph.js').read_text()
    assert 'mountInteractiveGraph' in js
    assert 'interactiveGraphRows' in js
    assert 'onSelect: inspect' in js
    assert 'await import(root.dataset.graphModuleUrl)' in js
    assert 'no-store' in js
    assert 'textContent' in js and 'innerHTML' not in js
    if shutil.which('node'):
        result = subprocess.run(['node','--check',str(UI/'static/ml_lab_structured_graph.js')],
                                capture_output=True,text=True)
        assert result.returncode == 0, result.stderr


def test_real_shared_viewer_contract():
    if not shutil.which('node') or not SHARED.is_file():
        pytest.skip('SU-02 shared viewer module not available beside source archive')
    code = '''import {validateInteractiveGraphView,interactiveGraphRows,interactiveGraphLayout} from %s;
const input=JSON.parse(await new Promise(resolve=>{let s='';process.stdin.on('data',d=>s+=d);process.stdin.on('end',()=>resolve(s));}));
const result=validateInteractiveGraphView(input);
const rows=interactiveGraphRows(result);
if(rows.nodes.length!==result.nodes.length || rows.edges.length!==result.edges.length) throw new Error('wrong record table');
if(interactiveGraphLayout(result).size !== result.nodes.length) throw new Error('incomplete layout');
process.stdout.write('OK '+result.nodes.length+' '+result.edges.length);
''' % json.dumps(SHARED.as_uri())
    for name in ('xml_tree.json','shared_child_dag.json','cyclic_network.json'):
        result=subprocess.run(['node','--input-type=module','-e',code],input=json.dumps(project(name)),
                              capture_output=True,text=True,timeout=20)
        assert result.returncode==0, result.stderr
        assert result.stdout.startswith('OK ')


def test_optional_flask_route_and_read_only_asset(tmp_path):
    flask = pytest.importorskip('flask')
    from ml_lab.ui import create_ui_blueprint
    app = flask.Flask(__name__)
    app.config['TESTING'] = True
    app.register_blueprint(create_ui_blueprint(name='su04',experiment_project_root=str(tmp_path)),url_prefix='/lab')
    client=app.test_client()
    (tmp_path/'source.json').write_bytes((TESTS/'xml_tree.json').read_bytes())
    path='/lab/workbench/structured'
    assert client.get(path + '/graph').status_code==409
    selected=client.post(path,data={'section':'data','action':'select','path':'source.json','kind':'structured'})
    assert selected.status_code==200
    graph=client.get(path+'/graph')
    assert graph.status_code==200
    assert graph.json['view']['graph_role']=='source_structure'
    assert client.post(path,data={'section':'data','action':'structure'}).status_code==200
    (tmp_path/'source.json').write_text('{}')
    assert client.get(path+'/graph').status_code==409
    assert client.get('/lab/pah-shared/../setup.py').status_code in (404,405)


def test_long_identifiers_fail_instead_of_corrupting_selection():
    raw = source().to_dict()
    raw['nodes'][0]['label'] = 'n' * 241
    obj = StructuredObject.from_dict(raw)
    with pytest.raises(StructuredContractError, match='no IDs were altered'):
        make_shared_graph_view(obj, fingerprint='1'*64)
