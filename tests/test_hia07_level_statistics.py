"""HIA-07: complete source-level inventory versus declared observed subset."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml_lab.structured import (
    HIA_LEVEL_SCHEMA, StructuredContractError, StructuredObject,
    analyze_hierarchical_level, source_sha256,
)

ROOT = Path(__file__).resolve().parents[1]


def fixture(x=None, y=None, z=None, *, topology='tree'):
    x = x if x is not None else [0, 0, 1, 1] * 4
    y = y if y is not None else list(x)
    z = z if z is not None else [i % 2 for i in range(len(x))]
    records = []
    for i, (a, b, c) in enumerate(zip(x, y, z)):
        values = {}
        for name, value in (('x', a), ('y', b), ('z', c)):
            values[name] = ({'status':'present','value':value} if value is not None else
                            {'status':'explicit_null', 'value':None, 'source_reason':'path_absent'})
        records.append({'observation_id':f'r{i}', 'values':values})
    edges = [('e1','root','a'),('e2','root','b'),('e3','root','c'),('e4','a','leaf')]
    if topology == 'dag':
        edges.append(('e5','b','leaf'))
    return StructuredObject.from_dict({
        'schema':'ml-lab.structured-object@1',
        'topology':{'kind':topology},
        'nodes':[{'id':n, 'kind':'atom', 'attributes':{}}
                 for n in ('root','a','b','c','leaf')],
        'edges':[{'id':edge, 'kind':'source_link','source_id':a,'target_id':b}
                 for edge,a,b in edges],
        'observation_unit':'record', 'observations':records, 'provenance':{},
    })


def mapped():
    return [{'node_id':'a','variable_id':'x'}, {'node_id':'b','variable_id':'y'}]


def analyze(source=None, **kwargs):
    return analyze_hierarchical_level(source or fixture(), level=1,
                                      mappings=kwargs.pop('mappings', mapped()), **kwargs)


def test_joint_entropy_redundant_level_and_source_preservation():
    source = fixture()
    original = source.to_json()
    report = analyze(source)
    assert report['schema'] == HIA_LEVEL_SCHEMA
    assert report['source_digest'] == source_sha256(source)
    assert report['level_policy'] == 'rooted_depth'
    assert report['level_node_ids'] == ['a','b','c']
    assert report['level_node_count'] == 3
    assert report['mapped_node_count'] == 2 and report['unmapped_node_count'] == 1
    assert report['statistical_scope'] == 'explicit_mapped_level_subset'
    assert report['preceding_level_node_count'] == 1
    assert report['conditional_information_status'] == 'not_computed_hia08'
    info = report['level_information']
    assert info['status'] == 'ok' and info['support'] == 16
    assert info['joint_entropy_bits'] == pytest.approx(1)
    assert info['sum_atomic_entropy_bits'] == pytest.approx(2)
    assert info['total_correlation_bits'] == pytest.approx(1)
    assert info['distinct_joint_states'] == 2
    assert [i['mi_with_other_mapped_members_bits'] for i in info['per_mapped_node']] == pytest.approx([1,1])
    assert info['per_mapped_node'][0]['node_id'] == 'a'
    assert report['binding_plan']['variables'][0]['node_ids'] == ['a']
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert source.to_json() == original


def test_independent_variables_joint_level_information():
    x = [0,0,1,1] * 4
    y = [0,1,0,1] * 4
    info = analyze(fixture(x,y))['level_information']
    assert info['status'] == 'ok'
    assert info['joint_entropy_bits'] == pytest.approx(2)
    assert info['sum_atomic_entropy_bits'] == pytest.approx(2)
    assert info['total_correlation_bits'] == pytest.approx(0)
    assert [v['mi_with_other_mapped_members_bits'] for v in info['per_mapped_node']] == pytest.approx([0,0])


def test_full_level_coverage_includes_all_mapped_nodes():
    report = analyze(mappings=[*mapped(), {'node_id':'c','variable_id':'z'}])
    assert report['unmapped_node_count'] == 0
    assert report['complete_level_statistical_coverage']
    assert report['statistical_scope'] == 'complete_level'
    assert report['level_information']['support'] == 16


def test_root_or_single_mapping_has_zero_total_correlation():
    report = analyze_hierarchical_level(fixture(), level=0, mappings=[
        {'node_id':'root','variable_id':'x'}])
    assert report['level_node_ids'] == ['root']
    info = report['level_information']
    assert info['joint_entropy_bits'] == pytest.approx(1)
    assert info['total_correlation_bits'] == pytest.approx(0)
    assert info['per_mapped_node'][0]['mi_with_other_mapped_members_bits'] == pytest.approx(0)


def test_multi_parent_dag_longest_path_level_policy():
    report = analyze(fixture(topology='dag'))
    assert report['source_topology'] == 'dag'
    assert report['level_policy'] == 'dag_longest_path'
    assert report['level_node_ids'] == ['a','b','c']
    assert report['level_information']['total_correlation_bits'] == pytest.approx(1)
    leaf = analyze_hierarchical_level(fixture(topology='dag'), level=2,
                                      mappings=[{'node_id':'leaf','variable_id':'z'}])
    assert leaf['level_node_ids'] == ['leaf']
    assert leaf['preceding_level_node_count'] == 4


def test_missing_support_uses_one_common_cohort_not_separate_marginals():
    x = [0,1,0,1,0,1,0,1]
    y = [None,None,0,1,0,1,0,1]
    report = analyze(fixture(x,y))
    info = report['level_information']
    assert info['support'] == 6
    assert info['excluded_rows'] == 2
    assert info['excluded_by_reason'] == {'incomplete_case':2}
    assert info['joint_entropy_bits'] == pytest.approx(1)
    assert info['total_correlation_bits'] == pytest.approx(1)


def test_missing_all_rows_has_unavailable_estimates():
    info = analyze(fixture([None]*6, [None]*6, [0]*6))['level_information']
    assert info['status'] == 'insufficient_samples'
    assert info['support'] == 0
    assert info['joint_entropy_bits'] is None
    assert info['total_correlation_bits'] is None
    assert info['per_mapped_node'] == []


def test_repeated_values_and_numeric_binning_are_explicit():
    raw = fixture().to_dict()
    raw['observations'][0]['values']['x'] = {'status':'multiple','value':None,
        'occurrences':[{'status':'present','value':0},{'status':'present','value':1}]}
    source = StructuredObject.from_dict(raw)
    report = analyze(source)
    assert report['level_information']['status'] == 'unresolved_repeated_values'
    assert report['level_information']['joint_entropy_bits'] is None
    with pytest.raises(StructuredContractError, match='scalar_only'):
        analyze(source, occurrence_policy='scalar_only')
    assert analyze(source, occurrence_policy='mean_numeric')['level_information']['status'] == 'float_requires_explicit_binning'
    bins = {'x':{'mode':'equal_width','bins':2}}
    assert analyze(source, occurrence_policy='mean_numeric', bins=bins)['level_information']['status'] == 'ok'


def test_float_discretization_required_and_sparse_state_warning():
    fl = [.1,.1,.9,.9] * 4
    assert analyze(fixture(fl,fl))['level_information']['status'] == 'float_requires_explicit_binning'
    info = analyze(fixture(fl,fl), bins={'x':{'mode':'equal_width','bins':2},
                                         'y':{'mode':'equal_width','bins':2}})['level_information']
    assert info['status'] == 'ok'
    assert info['total_correlation_bits'] == pytest.approx(1)
    values = list(range(100))
    sparse = analyze(fixture(values,values))['level_information']
    assert sparse['status'] == 'ok'
    assert sparse['sparse_joint_states_relative_to_support']
    assert sparse['distinct_joint_states'] == 100


def test_invalid_depth_wrong_member_ambiguous_mapping_unknown_column():
    obj = fixture()
    for depth in (-1,True,3,1.0,'1'):
        with pytest.raises(StructuredContractError, match='level'):
            analyze_hierarchical_level(obj, level=depth, mappings=mapped())
    with pytest.raises(StructuredContractError, match='selected exact structural level'):
        analyze(obj, mappings=[{'node_id':'leaf','variable_id':'x'}])
    with pytest.raises(StructuredContractError, match='distinct'):
        analyze(obj, mappings=[{'node_id':'a','variable_id':'x'},{'node_id':'b','variable_id':'x'}])
    with pytest.raises(StructuredContractError, match='distinct'):
        analyze(obj, mappings=[{'node_id':'a','variable_id':'x'},{'node_id':'a','variable_id':'y'}])
    with pytest.raises(StructuredContractError, match='unknown aligned observation'):
        analyze(obj, mappings=[{'node_id':'a','variable_id':'wrong'}])
    with pytest.raises(StructuredContractError, match='at least one level'):
        analyze(obj, mappings=[])
    with pytest.raises(StructuredContractError, match='invalid discretization'):
        analyze(obj, bins={'x':{'mode':'quantile','bins':1}})


def test_large_level_retains_every_source_node_and_exact_mapped_subset():
    count = 5101
    raw = fixture().to_dict()
    raw['nodes'] = [{'id':'root','kind':'atom','attributes':{}}] + [
        {'id':f'n{i}', 'kind':'atom', 'attributes':{}} for i in range(count)]
    raw['edges'] = [{'id':f'e{i}','kind':'source_link','source_id':'root','target_id':f'n{i}'}
                    for i in range(count)]
    source = StructuredObject.from_dict(raw)
    report = analyze_hierarchical_level(source, level=1, mappings=[
        {'node_id':'n0','variable_id':'x'}, {'node_id':'n5100','variable_id':'y'},
    ])
    assert report['level_node_count'] == count
    assert len(report['level_node_ids']) == count
    assert report['unmapped_node_count'] == count - 2
    assert report['level_information']['status'] == 'ok'


def test_source_only_level_never_fabricates_observations():
    raw = fixture().to_dict()
    raw['observation_unit'] = 'source'
    raw['observations'] = []
    raw['topology'].pop('capabilities', None)  # Re-derive source-only capabilities.
    raw.pop('object_id', None)
    obj = StructuredObject.from_dict(raw)
    report = analyze(obj)
    assert report['aligned_rows'] == 0
    assert report['level_information']['status'] == 'insufficient_samples'
    assert report['level_information']['joint_entropy_bits'] is None


def test_route_rejects_stale_revision_and_requires_opt_in(tmp_path):
    flask = pytest.importorskip('flask')
    from ml_lab.ui import create_ui_blueprint
    app = flask.Flask(__name__)
    app.config['TESTING'] = True
    app.register_blueprint(create_ui_blueprint(name='hia07', experiment_project_root=str(tmp_path)),
                           url_prefix='/lab')
    client = app.test_client()
    source = fixture()
    path = tmp_path/'sample.json'
    path.write_text(source.to_json(), encoding='utf-8')
    base = '/lab/workbench/structured'
    assert client.post(base, data={'section':'data','action':'select','path':'sample.json',
                                   'kind':'structured'}).status_code == 200
    graph = client.get(base+'/graph').json
    endpoint = base+'/hierarchy/level-information'
    body = {'source_revision':graph['view']['source_revision'], 'level':1,
            'mappings':mapped(),'occurrence_policy':'preserve','bins':{},'min_samples':2}
    headers = {'X-ML-Lab-HIA':'explicit-level-information'}
    assert client.post(endpoint, json=body).status_code == 400
    assert client.post(endpoint,json={**body,'source_revision':'sha256:'+'0'*64}, headers=headers).status_code == 409
    result = client.post(endpoint,json=body, headers=headers)
    assert result.status_code == 200, result.json
    assert result.json['report']['level_information']['total_correlation_bits'] == pytest.approx(1)
    assert client.post(endpoint,json={**body,'unexpected':'data'}, headers=headers).status_code == 409
    path.write_text('{}',encoding='utf-8')
    assert client.post(endpoint,json=body, headers=headers).status_code == 409
    assert not list(tmp_path.glob('*.db'))


def test_ui_integrates_level_panel_without_removing_previous_inspectors():
    template = (ROOT/'src/ml_lab/ui/templates/ml_lab_ui/workbench_structured.html').read_text()
    route = (ROOT/'src/ml_lab/ui/workbench_structured_routes.py').read_text()
    script = (ROOT/'src/ml_lab/ui/static/ml_lab_hia_level.js').read_text()
    assert 'data-hia-level-url' in template
    assert 'data-hia-level-action="compute"' in template
    assert 'data-hia-level-action="export"' in template
    assert 'ml_lab_hia_level.js' in template and 'ml_lab_hia_level.js' in route
    assert 'data-hia-atomic-action="compute"' in template
    assert 'data-hia-neighbor-action="compute"' in template
    assert 'data-hia-branch-action="compute"' in template
    assert 'data-su10-sandbox' in template
    assert 'mlwb:hia-scope' in script
    assert 'explicit-level-information' in script
    assert 'data-hia-level-results' in template
