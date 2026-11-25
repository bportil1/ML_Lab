"""Whole observed-variable network: no edge/observation conflation or silent subset."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from ml_lab.structured import StructuredContractError, StructuredObject
from ml_lab.structured.network_information import analyze_observed_network
from ml_lab.ui.structured_network_view import association_view
from ml_lab.ui.structured_source_intake import delimited_to_structured


def data_table(headers, rows):
    body = ','.join(headers) + '\n' + '\n'.join(','.join(map(str, row)) for row in rows) + '\n'
    blob = body.encode()
    return delimited_to_structured(blob, relative_path='network.csv',
        sha256=sha256(blob).hexdigest(), delimiter=',')[0]


def direct_source(rows):
    names = tuple(rows[0])
    return StructuredObject.from_dict({
        'schema':'ml-lab.structured-object@1', 'topology':{'kind':'tree'},
        'nodes':[{'id':'root','kind':'root','attributes':{}}], 'edges':[],
        'observation_unit':'record',
        'observations':[{'observation_id':f'obs-{i}', 'values':{
            name:({'status':'present','value':value} if value is not None
                  else {'status':'missing','value':None,'source_reason':'path_absent'})
            for name,value in row.items()}}
            for i,row in enumerate(rows)],
        'provenance':{'test':'synthetic'},
    })


def pair(report, a, b):
    return next(item for item in report['pairs'] if {item['x'],item['y']} == {a,b})


def test_full_three_column_network_includes_every_pair_and_true_zero():
    source = data_table(('a','b','c'), [(0,0,0),(0,0,1),(1,1,0),(1,1,1)]*3)
    before = source.to_json()
    report = analyze_observed_network(source, min_samples=2)
    assert report['variables'] == ['a','b','c']
    assert report['support'] == {'aligned_rows':12,'total_variables':3,
                                   'pairs_requested':3,'pairs_measured':3}
    assert pair(report,'a','b')['value_bits'] == pytest.approx(1)
    assert pair(report,'a','c')['value_bits'] == pytest.approx(0)
    assert pair(report,'b','c')['value_bits'] == pytest.approx(0)
    assert source.to_json() == before
    assert report == analyze_observed_network(source, min_samples=2)


def test_conditional_network_detects_xor_interaction_hidden_from_pairwise_mi():
    rows = [{'x':x,'y':x ^ z,'z':z} for _ in range(4) for x in (0,1) for z in (0,1)]
    source = direct_source(rows)
    marginal = analyze_observed_network(source, variables=('x','y'), min_samples=2)
    conditioned = analyze_observed_network(source, variables=('x','y'),
                                          conditioning=('z',), min_samples=2)
    assert pair(marginal,'x','y')['value_bits'] == pytest.approx(0, abs=1e-12)
    assert pair(conditioned,'x','y')['value_bits'] == pytest.approx(1)
    assert conditioned['metric'] == 'empirical_conditional_mutual_information'
    assert conditioned['conditioning'] == ['z']
    with pytest.raises(StructuredContractError, match='cannot also'):
        analyze_observed_network(source, variables=('x','y','z'), conditioning=('z',))


def test_full_joint_total_correlation_finds_three_way_xor_dependency():
    # Every marginal pair is independent, despite a deterministic 3-way rule.
    source = direct_source([{'x':x,'y':y,'z':x ^ y}
                            for _ in range(4) for x in (0,1) for y in (0,1)])
    result = analyze_observed_network(source, min_samples=2)
    assert all(p['value_bits'] == pytest.approx(0, abs=1e-12)
               for p in result['pairs'])
    assert result['joint_dependency']['joint_entropy_bits'] == pytest.approx(2)
    assert result['joint_dependency']['total_correlation_bits'] == pytest.approx(1)
    assert result['joint_dependency']['support'] == 16


def test_per_pair_complete_case_support_and_missing_as_category():
    source = direct_source([{'a':'x','b':'x','c':'a'},
                            {'a':'y','b':None,'c':'b'},
                            {'a':'x','b':'x','c':'b'},
                            {'a':'y','b':'y','c':'a'}])
    out = analyze_observed_network(source, min_samples=3)
    assert pair(out,'a','b')['support'] == 3
    assert pair(out,'a','b')['excluded_rows'] == 1
    assert pair(out,'a','c')['support'] == 4
    cat = analyze_observed_network(source, missing_policy='as_category', min_samples=3)
    assert pair(cat,'a','b')['support'] == 4
    with pytest.raises(StructuredContractError, match='missing'):
        analyze_observed_network(source, missing_policy='error')
    insufficient = analyze_observed_network(source, min_samples=4)
    assert pair(insufficient,'a','b')['status'] == 'insufficient_samples'
    assert pair(insufficient,'a','b')['value_bits'] is None


def test_floating_numeric_requires_explicit_binning():
    src = direct_source([{'x':float(i % 4),'y':i % 2} for i in range(8)])
    with pytest.raises(StructuredContractError, match='explicit discretization'):
        analyze_observed_network(src)
    out = analyze_observed_network(src, bins={'x':{'mode':'equal_width','bins':4}})
    assert out['estimator']['discretization']['x']['actual_bins'] == 4
    assert pair(out,'x','y')['value_bits'] is not None


def test_repeated_cells_and_no_observations_fail_explicitly():
    src = direct_source([{'a':i%2,'b':i%2} for i in range(8)])
    content = src.to_dict(); content.pop('object_id',None)
    content['observations'][0]['values']['a']={'status':'multiple','value':None,'occurrences':[
        {'status':'present','value':1},{'status':'present','value':2}]}
    repeated = StructuredObject.from_dict(content)
    with pytest.raises(StructuredContractError, match='repeated'):
        analyze_observed_network(repeated)
    content['observations'] = []
    content['topology'] = {'kind':'tree'}
    structural = StructuredObject.from_dict(content)
    with pytest.raises(StructuredContractError, match='aligned observation'):
        analyze_observed_network(structural)


def test_honors_network_computational_cap_without_implicit_sampling():
    headings = [f'c{i:03d}' for i in range(80)]
    src = direct_source([{name: i % 3 for name in headings} for i in range(100)])
    report = analyze_observed_network(src)
    assert report['support']['pairs_requested'] == 3160
    assert report['support']['pairs_measured'] == 3160
    assert report['joint_dependency']['status'] == 'joint_dimensionality_limit'
    too_big = direct_source([{name: i % 3 for name in headings} for i in range(4000)])
    with pytest.raises(StructuredContractError, match='exceeds 10,000,000'):
        analyze_observed_network(too_big)
    selected = analyze_observed_network(too_big, variables=tuple(headings[:3]))
    assert selected['support']['pairs_requested'] == 3


def test_view_uses_separate_undirected_statistical_edges_and_full_matrix():
    src = data_table(('f','g','h'), [(0,0,0),(0,0,1),(1,1,0),(1,1,1)]*3)
    report = analyze_observed_network(src)
    view = association_view(report, source_revision='sha256:' + 'e'*64)
    assert view['graph_role'] == 'empirical_statistical_association'
    assert len(view['nodes']) == 3 and len(view['edges']) == 3
    assert all(edge['directed'] is False and edge['traceable'] is False
               for edge in view['edges'])
    assert set(view['overlays']['edges']) == {edge['id'] for edge in view['edges']}
    assert json.loads(json.dumps(view, allow_nan=False)) == view


def test_single_file_workflow_template_has_whole_network_and_shared_view():
    from jinja2 import Environment, FileSystemLoader
    root = Path(__file__).resolve().parents[1]
    env = Environment(loader=FileSystemLoader(root/'src/ml_lab/ui/templates'))
    template = env.get_template('ml_lab_ui/workbench_structured.html')
    html = (root/'src/ml_lab/ui/templates/ml_lab_ui/workbench_structured.html').read_text()
    assert 'Whole-network analysis' in html
    assert 'ml_lab_structured_network.js' in html
    assert 'workbench_structured_network' in html
    assert 'data-mlwb-information-network' in html
    assert template is not None


def test_local_flask_endpoint_checks_source_revision_and_returns_full_matrix(tmp_path):
    flask = pytest.importorskip('flask')
    from ml_lab.ui import create_ui_blueprint
    src = tmp_path/'network.csv'
    src.write_text('a,b,c\n'+('0,0,0\n0,0,1\n1,1,0\n1,1,1\n'*3))
    app = flask.Flask(__name__)
    app.config['TESTING'] = True
    app.register_blueprint(create_ui_blueprint(name='network',
        experiment_project_root=str(tmp_path)), url_prefix='/tools/ml-lab')
    client = app.test_client()
    root='/tools/ml-lab/workbench/structured'
    response=client.post(root,data={'section':'data','action':'select',
        'path':'network.csv','kind':'auto'})
    assert response.status_code == 200
    endpoint=root+'/network'
    raw = {'source_revision':'sha256:'+sha256(src.read_bytes()).hexdigest(),
        'variables':[], 'conditioning':[], 'bins':{}, 'min_samples':2}
    response = client.post(endpoint, json=raw,
      headers={'X-ML-Lab-Network':'explicit-analysis'})
    assert response.status_code == 200
    data = response.get_json()
    assert len(data['report']['pairs']) == 3
    assert data['view']['schema'] == 'pah-shared.graph-view@1'
    invalid=client.post(endpoint,json={**raw,'source_revision':'sha256:'+'a'*64},
      headers={'X-ML-Lab-Network':'explicit-analysis'})
    assert invalid.status_code == 409
    without_header=client.post(endpoint,json=raw)
    assert without_header.status_code == 400
    src.write_text(src.read_text()+'0,1,0\n')
    stale=client.post(endpoint,json=raw,
      headers={'X-ML-Lab-Network':'explicit-analysis'})
    assert stale.status_code == 409
