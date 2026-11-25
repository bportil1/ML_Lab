"""HIA-08 empirical conditional-level information; source topology is never a PGM."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml_lab.structured import (HIA_CONDITIONAL_LEVEL_SCHEMA, StructuredContractError,
                               StructuredObject, analyze_hierarchical_conditional_level,
                               source_sha256)

ROOT = Path(__file__).resolve().parents[1]


def fixture(*, x=None, z=None, y=None, topology='tree'):
    # Independent balanced X and Z; Y=X yields I(Y;X|Z)=1.
    if x is None:
        x = [0, 0, 1, 1] * 8
    if z is None:
        z = [0, 1, 0, 1] * 8
    if y is None:
        y = list(x)
    observations = []
    for i, (a, b, c) in enumerate(zip(x, z, y)):
        observations.append({'observation_id': f'row-{i}', 'values': {
            name: ({'status':'present', 'value':value} if value is not None else
                   {'status':'explicit_null', 'value':None, 'source_reason':'path_absent'})
            for name, value in (('x',a),('z',b),('y',c))}})
    edges = [('e1','root','l'), ('e2','root','other'),('e3','l','target')]
    if topology == 'dag':
        edges.append(('e4','other','target'))
    return StructuredObject.from_dict({
        'schema':'ml-lab.structured-object@1', 'topology':{'kind':topology},
        'nodes':[{'id':name, 'kind':'atom', 'attributes':{}}
                 for name in ('root','l','other','target')],
        'edges':[{'id':edge,'kind':'source_link','source_id':a,'target_id':b}
                 for edge,a,b in edges],
        'observation_unit':'record', 'observations':observations, 'provenance':{},
    })


def mapping(node, col):
    return {'node_id':node,'variable_id':col}


def analyze(source=None, **opts):
    return analyze_hierarchical_conditional_level(
        source or fixture(), level=1, mappings=[mapping('l','x')],
        preceding_mappings=[mapping('root','z')], **opts)


def test_conditional_entropy_and_target_increment_on_one_matched_cohort():
    source = fixture()
    before = source.to_json()
    result = analyze(source, target=mapping('target','y'))
    assert result['schema'] == HIA_CONDITIONAL_LEVEL_SCHEMA
    assert result['source_digest'] == source_sha256(source)
    assert result['level_node_ids'] == ['l','other']
    assert result['preceding_level_node_ids'] == ['root']
    assert result['unmapped_level_node_count'] == 1
    assert result['mapped_preceding_node_count'] == 1
    assert result['statistical_scope'] == 'explicit_mapped_level_and_preceding_subsets'
    assert result['independence_model'] == 'empirical_joint_no_factorization'
    info = result['level_conditioned_information']
    assert info['status'] == 'ok' and info['support'] == 32
    assert info['level_entropy_bits'] == pytest.approx(1)
    assert info['level_conditional_entropy_bits'] == pytest.approx(1)
    assert info['level_prior_mutual_information_bits'] == pytest.approx(0)
    assert info['conditional_total_correlation_bits'] == pytest.approx(0)
    assert info['conditioning_strata'] == 2
    assert info['per_mapped_level_node'][0]['node_id'] == 'l'
    target = result['target_increment_information']
    assert target['status'] == 'ok' and target['support'] == 32
    assert target['target_incremental_information_bits'] == pytest.approx(1)
    assert target['target_entropy_given_prior_bits'] == pytest.approx(1)
    assert target['target_entropy_given_level_and_prior_bits'] == pytest.approx(0)
    assert json.loads(json.dumps(result,allow_nan=False)) == result
    assert source.to_json() == before


def test_confounding_target_zero_after_prior_conditioning():
    z = [0,0,1,1] * 8
    result = analyze(fixture(x=z,z=z,y=z),target=mapping('target','y'))
    info=result['level_conditioned_information']
    assert info['level_entropy_bits'] == pytest.approx(1)
    assert info['level_conditional_entropy_bits'] == pytest.approx(0)
    assert info['level_prior_mutual_information_bits'] == pytest.approx(1)
    assert result['target_increment_information']['target_incremental_information_bits'] == pytest.approx(0)


def test_joint_prior_conditioning_not_product_of_independent_parents():
    # X = XOR(Z1,Z2): independent of either Z alone but fully determined by
    # the JOINT pair. This catches erroneous per-conditioner MI aggregation.
    records=[]
    for _ in range(8):
        for z1,z2 in ((0,0),(0,1),(1,0),(1,1)):
            records.append((z1 ^ z2,z1,z2))
    raw=fixture().to_dict()
    raw['nodes'].append({'id':'second_root','kind':'atom','attributes':{}})
    raw['topology']={'kind':'forest'}
    raw['observations']=[{'observation_id':f'row-{i}', 'values':{
        n:{'status':'present','value':v}
        for n,v in zip(('x','z','y'), row)}} for i,row in enumerate(records)]
    source=StructuredObject.from_dict(raw)
    result=analyze_hierarchical_conditional_level(
        source,level=1,mappings=[mapping('l','x')],
        preceding_mappings=[mapping('root','z'),mapping('second_root','y')])
    assert result['level_conditioned_information']['conditioning_strata'] == 4
    assert result['level_conditioned_information']['level_conditional_entropy_bits'] == pytest.approx(0)
    assert result['level_conditioned_information']['level_prior_mutual_information_bits'] == pytest.approx(1)


def test_incomplete_target_cohort_is_independent_from_base():
    x=[0,0,1,1]*8; z=[0,1,0,1]*8
    target=[None]*24 + x[24:]
    result=analyze(fixture(x=x,z=z,y=target),target=mapping('target','y'))
    assert result['level_conditioned_information']['support'] == 32
    assert result['target_increment_information']['support'] == 8
    assert result['target_increment_information']['excluded_rows'] == 24
    assert result['target_increment_information']['status'] == 'ok'


def test_joint_strata_min_support_marks_estimate_unavailable():
    # 8 previous-level states each occur once; no imputed estimate.
    unique=list(range(8))
    result=analyze(fixture(x=[0,1]*4, z=unique, y=[0,1]*4), min_stratum_samples=2)
    info=result['level_conditioned_information']
    assert info['status'] == 'insufficient_conditioning_support'
    assert info['under_supported_strata'] == 8
    assert info['level_conditional_entropy_bits'] is None


def test_missing_binning_and_repeated_cells_are_explicit():
    raw=fixture().to_dict()
    raw['observations'][0]['values']['x']={
        'status':'multiple','value':None,'occurrences':[
            {'status':'present','value':0},{'status':'present','value':1}]}
    source=StructuredObject.from_dict(raw)
    assert analyze(source)['level_conditioned_information']['status']=='unresolved_repeated_values'
    with pytest.raises(StructuredContractError, match='scalar_only'):
        analyze(source, occurrence_policy='scalar_only')
    assert analyze(source, occurrence_policy='mean_numeric')['level_conditioned_information']['status']=='float_requires_explicit_binning'
    assert analyze(source, occurrence_policy='mean_numeric',
                   bins={'x':{'mode':'equal_width','bins':2}})['level_conditioned_information']['status']=='ok'
    floating=fixture(x=[.1,.2,.7,.8]*8)
    assert analyze(floating)['level_conditioned_information']['status']=='float_requires_explicit_binning'


def test_rejects_ambiguity_target_leakage_and_wrong_levels():
    source=fixture()
    with pytest.raises(StructuredContractError, match='preceding-level mappings'):
        analyze_hierarchical_conditional_level(source,level=1,
            mappings=[mapping('l','x')],preceding_mappings=[])
    with pytest.raises(StructuredContractError, match='strictly earlier'):
        analyze_hierarchical_conditional_level(source,level=1,
            mappings=[mapping('l','x')],preceding_mappings=[mapping('target','z')])
    with pytest.raises(StructuredContractError, match='distinct'):
        analyze_hierarchical_conditional_level(source,level=1,
            mappings=[mapping('l','x')],preceding_mappings=[mapping('root','x')])
    with pytest.raises(StructuredContractError, match='distinct'):
        analyze(source,target=mapping('root','z'))
    with pytest.raises(StructuredContractError, match='outside the selected'):
        analyze(source,target=mapping('other','y'))
    with pytest.raises(StructuredContractError, match='unknown original'):
        analyze(source,target=mapping('fake','y'))
    with pytest.raises(StructuredContractError, match='unknown aligned'):
        analyze(source,target=mapping('target','wrong'))
    for level in (-1,True,3,'1'):
        with pytest.raises(StructuredContractError):
            analyze_hierarchical_conditional_level(source,level=level,
                mappings=[mapping('l','x')],preceding_mappings=[])


def test_level_zero_has_empty_conditioner_and_target_may_be_prior_unmapped():
    source=fixture()
    report=analyze_hierarchical_conditional_level(source,level=0,
        mappings=[mapping('root','z')],preceding_mappings=[])
    assert report['level_conditioned_information']['prior_entropy_bits']==pytest.approx(0)
    assert report['level_conditioned_information']['level_conditional_entropy_bits']==pytest.approx(1)
    # An output at an earlier structural depth is allowed provided it is not
    # itself a declared conditioner. A selected level-2 variable can be tested
    # against a root-level target Y while conditioning on mapped level-1 X.
    report=analyze_hierarchical_conditional_level(source,level=2,
        mappings=[mapping('target','y')], preceding_mappings=[mapping('l','x')],
        target=mapping('root','z'))
    assert report['target_increment_information']['status']=='ok'
    assert report['mapped_preceding_node_count']==1
    assert report['unmapped_preceding_node_count']==2


def test_dag_levels_preserved_and_source_only_unavailable():
    result=analyze(fixture(topology='dag'))
    assert result['level_policy']=='dag_longest_path'
    source=fixture()
    raw=source.to_dict(); raw['observation_unit']='source'; raw['observations']=[]
    raw['topology'].pop('capabilities',None); raw.pop('object_id',None)
    source=StructuredObject.from_dict(raw)
    result=analyze(source)
    assert result['level_conditioned_information']['status']=='insufficient_samples'
    assert result['level_conditioned_information']['level_conditional_entropy_bits'] is None


def test_large_level_keeps_all_original_ids_without_sampling():
    source=fixture()
    raw=source.to_dict()
    count=5101
    raw['nodes']=[{'id':'root','kind':'atom','attributes':{}}]+[
        {'id':f'n{i}','kind':'atom','attributes':{}} for i in range(count)]
    raw['edges']=[{'id':f'e{i}','kind':'source_link','source_id':'root','target_id':f'n{i}'}
                  for i in range(count)]
    source=StructuredObject.from_dict(raw)
    result=analyze_hierarchical_conditional_level(source,level=1,
        mappings=[mapping('n0','x')],preceding_mappings=[mapping('root','z')])
    assert result['level_node_count']==count
    assert len(result['level_node_ids'])==count
    assert result['unmapped_level_node_count']==count-1
    assert result['level_conditioned_information']['status']=='ok'


def test_route_requires_opt_in_and_pinned_revision(tmp_path):
    flask=pytest.importorskip('flask')
    from ml_lab.ui import create_ui_blueprint
    app=flask.Flask(__name__); app.config['TESTING']=True
    app.register_blueprint(create_ui_blueprint(name='hia08',experiment_project_root=str(tmp_path)),
                           url_prefix='/lab')
    client=app.test_client()
    path=tmp_path/'test.json'; path.write_text(fixture().to_json(),encoding='utf-8')
    page='/lab/workbench/structured'
    assert client.post(page,data={'section':'data','action':'select','path':'test.json',
                                  'kind':'structured'}).status_code==200
    revision=client.get(page+'/graph').json['view']['source_revision']
    endpoint=page+'/hierarchy/conditional-information'
    body={'source_revision':revision,'level':1,'mappings':[mapping('l','x')],
          'preceding_mappings':[mapping('root','z')], 'target':mapping('target','y'),
          'occurrence_policy':'preserve','bins':{},'min_samples':2,'min_stratum_samples':2}
    headers={'X-ML-Lab-HIA':'explicit-conditional-level-information'}
    assert client.post(endpoint,json=body).status_code==400
    assert client.post(endpoint,json={**body,'source_revision':'sha256:'+'0'*64},headers=headers).status_code==409
    result=client.post(endpoint,json=body,headers=headers)
    assert result.status_code==200, result.json
    assert result.json['report']['target_increment_information']['target_incremental_information_bits']==pytest.approx(1)


def test_frontend_is_versioned_and_extends_existing_workspace():
    scripts=(ROOT/'src/ml_lab/ui/static/ml_lab_hia_conditional.js').read_text()
    routes=(ROOT/'src/ml_lab/ui/workbench_structured_routes.py').read_text()
    html=(ROOT/'src/ml_lab/ui/templates/ml_lab_ui/workbench_structured.html').read_text()
    assert "'ml_lab_hia_conditional.js'" in routes
    assert 'v=graph_controller_revision' in html
    assert 'mlwb:hia-scope' in scripts
    assert 'explicit-conditional-level-information' in scripts
    assert 'data-hia-conditional-results' in html
    assert 'data-hia-level-results' in html and 'data-hia-branch-results' in html
