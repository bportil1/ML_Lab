"""HIA-03: verified hierarchy navigation over SU-04; no invented statistics."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from ml_lab.structured import StructuredObject, StructuredContractError, source_sha256
from ml_lab.ui.structured_hia_view import (
    OVERVIEW_SCHEMA, INSPECTION_SCHEMA, PAGE_SIZE,
    hierarchy_overview, inspect_workspace_scope,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/structured_uncertainty'
UI = ROOT / 'src/ml_lab/ui'
REV = 'sha256:' + '1' * 64


def tree():
    return StructuredObject.from_json((FIXTURE / 'xml_tree.json').read_text(encoding='utf-8'))


def scope(kind, id=None, level=None):
    return {'kind':kind, 'node_ids':[id] if id is not None else [], 'level':level}


def source_graph(topology, names, edges):
    return StructuredObject.from_dict({
        'schema': 'ml-lab.structured-object@1',
        'topology': {'kind':topology},
        'nodes': [{'id':n, 'kind':'element', 'attributes':{}} for n in names],
        'edges': [{'id':f'e{i}', 'kind':'edge', 'source_id':a, 'target_id':b}
                  for i, (a,b) in enumerate(edges)],
        'observation_unit': 'source', 'observations': [], 'provenance': {},
    })


def test_verified_tree_levels_and_original_branch_scope_no_estimates():
    source = tree()
    before = source.to_json()
    inventory = hierarchy_overview(source, source_revision=REV)
    assert inventory['schema'] == OVERVIEW_SCHEMA
    assert inventory['source_revision'] == REV
    assert inventory['source_digest'] == source_sha256(source)
    assert inventory['available'] is True
    assert inventory['level_policy'] == 'rooted_depth'
    assert inventory['branch_selection_allowed'] is True
    assert [x['node_count'] for x in inventory['levels']] == [1,1,1,2]
    selection = inspect_workspace_scope(source, source_revision=REV,
                                         scope_input=scope('branch', 'item'))
    assert selection['schema'] == INSPECTION_SCHEMA
    assert selection['source_revision'] == REV
    assert selection['selection']['node_ids'] == ['item','name','value']
    assert selection['member_page']['node_ids'] == ['item','name','value']
    assert selection['source_edges_are_statistical_dependencies'] is False
    assert all(metric['estimate'] is None for metric in selection['selection']['metrics'])
    node = inspect_workspace_scope(source, source_revision=REV,
                                    scope_input=scope('node', 'name'))
    assert node['node_context']['node_id'] == 'name'
    assert node['node_context']['parent_ids'] == ['item']
    assert node['node_context']['level'] == 3
    level = inspect_workspace_scope(source, source_revision=REV,
                                    scope_input=scope('level', level=3))
    assert level['selection']['node_ids'] == ['name', 'value']
    assert level['selection']['prior_level_node_ids'] == ['root','section','item']
    assert source.to_json() == before
    assert json.loads(json.dumps(level, allow_nan=False)) == level


def test_forest_and_dag_policies_and_unverified_topology_degrade_read_only():
    dag = source_graph('dag', ['r','a','b','c'], [('r','a'),('a','b'),('b','c'),('r','c')])
    inventory = hierarchy_overview(dag, source_revision=REV)
    assert inventory['available'] and inventory['level_policy'] == 'dag_longest_path'
    assert inventory['branch_selection_allowed'] is False
    assert inspect_workspace_scope(dag, source_revision=REV,
                                   scope_input=scope('level',level=3))['selection']['node_ids'] == ['c']
    with pytest.raises(StructuredContractError, match='branch'):
        inspect_workspace_scope(dag, source_revision=REV, scope_input=scope('branch','r'))
    cyclic = source_graph('directed_graph', ['x','y'], [('x','y'),('y','x')])
    report = hierarchy_overview(cyclic, source_revision=REV)
    assert not report['available']
    assert report['levels'] == []
    assert report['reason'] == 'levels_require_verified_tree_forest_or_dag'
    assert report['node_count'] == 2  # Still usable in original graph view


def test_large_hierarchy_displays_all_members_using_bounded_pages_not_sampling():
    count = 5201
    source = source_graph('tree', [f'n{i}' for i in range(count)],
                          [(f'n{i}',f'n{i+1}') for i in range(count-1)])
    output = inspect_workspace_scope(source, source_revision=REV,
                                      scope_input=scope('network'), offset=count - 50)
    assert output['member_page']['total'] == count
    assert len(output['selection']['node_ids']) == count
    assert output['member_page']['node_ids'] == [f'n{i}' for i in range(count-50, count)]
    assert output['member_page']['has_previous']
    assert not output['member_page']['has_next']
    assert PAGE_SIZE == 80


def test_malformed_scope_and_offsets_rejected_not_repaired():
    source = tree()
    with pytest.raises(StructuredContractError, match='offset'):
        inspect_workspace_scope(source,source_revision=REV,scope_input=scope('network'),offset=True)
    with pytest.raises(StructuredContractError, match='offset'):
        inspect_workspace_scope(source,source_revision=REV,scope_input=scope('network'),offset=-1)
    with pytest.raises(StructuredContractError, match='scope'):
        inspect_workspace_scope(source,source_revision=REV,scope_input={'kind':'level','node_ids':[]})
    with pytest.raises(StructuredContractError, match='unknown'):
        inspect_workspace_scope(source,source_revision=REV,scope_input=scope('node','fake'))


def test_flask_route_revalidates_cookie_sha_and_rejects_unsupported_requests(tmp_path):
    flask = pytest.importorskip('flask')
    from ml_lab.ui import create_ui_blueprint
    app = flask.Flask(__name__)
    app.config['TESTING'] = True
    app.register_blueprint(create_ui_blueprint(name='hia03', experiment_project_root=str(tmp_path)),
                           url_prefix='/lab')
    client = app.test_client()
    base = '/lab/workbench/structured'
    assert client.get(base + '/hierarchy').status_code == 409
    source_file = tmp_path / 'xml.json'
    source_file.write_bytes((FIXTURE / 'xml_tree.json').read_bytes())
    assert client.post(base, data={'section':'data','action':'select','path':'xml.json',
                                    'kind':'structured'}).status_code == 200
    summary = client.get(base + '/hierarchy')
    assert summary.status_code == 200, summary.json
    assert summary.json['available'] is True
    selected_revision = summary.json['source_revision']
    payload = {'source_revision':selected_revision,'scope':scope('level',level=3),'offset':0}
    url = base + '/hierarchy/inspect'
    headers = {'X-ML-Lab-HIA':'explicit-inspection'}
    assert client.post(url,json=payload).status_code == 400
    assert client.post(url,json={**payload,'source_revision':REV},headers=headers).status_code == 409
    assert client.post(url,json=payload,headers=headers).json['selection']['node_ids'] == ['name','value']
    assert client.post(url,json={**payload,'extraneous':'x'},headers=headers).status_code == 409
    assert client.get(base + '/graph').status_code == 200  # Original viewer not replaced.
    source_file.write_text('{}')
    assert client.get(base + '/hierarchy').status_code == 409
    assert client.post(url,json=payload,headers=headers).status_code == 409



def test_deep_layers_keep_distinct_display_groups_beyond_old_depth_39_cap():
    from ml_lab.ui.structured_graph_view import make_shared_graph_view
    count = 45
    source = source_graph('tree',[f'n{i}' for i in range(count)],
                          [(f'n{i}',f'n{i+1}') for i in range(count-1)])
    view = make_shared_graph_view(source, fingerprint='a'*64)
    groups = {item['id']: item['group'] for item in view['nodes']}
    assert groups['n39'] == 'depth-39'
    assert groups['n40'] == 'depth-40'
    assert groups['n44'] == 'depth-44'
    assert len(set(groups.values())) == count

def test_frontend_additive_controls_and_cached_scripts():
    template = (UI / 'templates/ml_lab_ui/workbench_structured.html').read_text()
    graph = (UI / 'static/ml_lab_structured_graph.js').read_text()
    hierarchy = (UI / 'static/ml_lab_hia_workspace.js').read_text()
    styles = (UI / 'static/ml_lab_workbench.css').read_text()
    for fragment in ('data-hia-overview-url', 'data-hia-inspect-url',
                     'data-hia-action="node"', 'data-hia-action="branch"',
                     'data-hia-action="level"', 'data-hia-action="network"',
                     'data-hia-action="fullscreen"', 'data-hia-action="popout"',
                     'data-hia-action="export-graph"', 'data-hia-action="export-scope"',
                     'ml_lab_hia_workspace.js'):
        assert fragment in template
    assert "mlwb:source-open', {detail: {payload, module, viewer}}" in graph
    assert 'mlwb:graph-viewport' in graph and 'mlwb:graph-select' in graph
    assert "'X-ML-Lab-HIA':'explicit-inspection'" in hierarchy
    assert 'textContent' in hierarchy and 'innerHTML' not in hierarchy
    assert "root.dataset.hiaOverviewUrl" in hierarchy
    assert 'requestFullscreen' in hierarchy and 'window.open' in hierarchy
    assert 'scope(' in hierarchy and 'selection.node_ids' in hierarchy
    assert 'mlwb-hia-in-scope' in styles
    assert "action('branch').disabled = true" in hierarchy
    assert "background.setAttribute('fill','#f8fbff')" in hierarchy
    routes = (UI / 'workbench_structured_routes.py').read_text()
    assert "'ml_lab_hia_workspace.js'" in routes
    assert 'data-graph-action="open"' in template and 'data-su10-sandbox' in template
    if shutil.which('node'):
        for filename in ('ml_lab_hia_workspace.js', 'ml_lab_structured_graph.js'):
            result = subprocess.run(['node','--check',str(UI/'static'/filename)],
                                    capture_output=True,text=True)
            assert result.returncode == 0, result.stderr
