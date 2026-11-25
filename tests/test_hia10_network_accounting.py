"""HIA-10 matched-cohort, cumulative and cross-level information accounting."""
import json

import pytest

from ml_lab.structured import (analyze_hierarchical_network_accounting,
                               StructuredContractError, StructuredObject)


def fixture(x=None,z=None,y=None):
    z = [0,0,1,1]*8 if z is None else z
    x = [0,1,0,1]*8 if x is None else x
    y = list(x) if y is None else y
    observations = [{'observation_id':f'obs-{i}', 'values':{
        key:({'status':'present','value':v} if v is not None else
             {'status':'explicit_null','value':None,'source_reason':'path_absent'})
        for key,v in (('z',a),('x',b),('y',c))}}
        for i,(a,b,c) in enumerate(zip(z,x,y))]
    return StructuredObject.from_dict({
        'schema':'ml-lab.structured-object@1','topology':{'kind':'tree'},
        'nodes':[{'id':n,'kind':'atom','attributes':{}} for n in ('root','left','right','leaf')],
        'edges':[{'id':e,'kind':'source_link','source_id':a,'target_id':b}
                 for e,a,b in [('edge0','root','left'),('edge1','root','right'),('edge2','left','leaf')]],
        'observation_unit':'record','observations':observations,'provenance':{},
    })


def m(node,col):return {'node_id':node,'variable_id':col}


def inspect(source=None, **kwargs):
    return analyze_hierarchical_network_accounting(
        source or fixture(),mappings=[m('root','z'),m('left','x')],**kwargs)


def test_independent_levels_chain_rule_and_target_increment():
    r=inspect(target=m('leaf','y'))
    info=r['network_information']
    assert info['status']=='ok' and info['support']==32
    assert info['joint_entropy_bits']==pytest.approx(2.)
    assert info['total_correlation_bits']==pytest.approx(0.)
    assert [row['level'] for row in info['levels']]==[0,1]
    assert [row['incremental_joint_entropy_given_prior_bits'] for row in info['levels']]==pytest.approx([1,1])
    assert [row['target_incremental_information_bits'] for row in info['levels']]==pytest.approx([0,1])
    assert info['target_mutual_information_bits']==pytest.approx(1.)
    assert info['cross_level_pairs'][0]['mutual_information_bits']==pytest.approx(0.)
    assert r['structural_node_count']==4 and r['mapped_node_count']==2
    assert not r['complete_observation_coverage']
    assert json.loads(json.dumps(r,allow_nan=False))==r


def test_redundant_level_is_not_double_counted():
    z=[0,0,1,1]*8
    r=inspect(fixture(x=z,z=z,y=z),target=m('leaf','y'))
    info=r['network_information']
    assert info['joint_entropy_bits']==pytest.approx(1.)
    assert info['total_correlation_bits']==pytest.approx(1.)
    assert info['levels'][1]['incremental_joint_entropy_given_prior_bits']==pytest.approx(0.)
    assert info['levels'][1]['information_shared_with_prior_bits']==pytest.approx(1.)
    assert info['levels'][1]['target_incremental_information_bits']==pytest.approx(0.)
    assert info['cross_level_pairs'][0]['mutual_information_bits']==pytest.approx(1.)


def test_unmapped_level_inventory_and_source_remains_unchanged():
    source=fixture()
    unchanged=source.to_json()
    info=inspect(source)['network_information']
    assert info['target_mutual_information_bits'] is None
    assert info['levels'][0]['unmapped_node_count']==0
    assert info['levels'][1]['unmapped_node_count']==1
    assert source.to_json()==unchanged


def test_matched_target_cohort_reports_missing_values():
    x=[0,1,0,1]*8
    y=x[:16]+[None]*16
    info=inspect(fixture(x=x,y=y),target=m('leaf','y'))['network_information']
    assert info['support']==16
    assert info['excluded_rows']==16
    assert info['status']=='ok'


def test_insufficient_support_has_no_pretend_information():
    info=inspect(fixture(x=[None]*32))['network_information']
    assert info['status']=='insufficient_samples'
    assert info['joint_entropy_bits'] is None
    assert info['levels']==[]


def test_invalid_node_and_duplicate_target_rejected():
    with pytest.raises(StructuredContractError, match='distinct'):
        inspect(target=m('left','y'))
    with pytest.raises(StructuredContractError, match='distinct'):
        analyze_hierarchical_network_accounting(fixture(),mappings=[m('root','z'),m('left','z')])
    with pytest.raises(StructuredContractError,match='unknown original'):
        inspect(target=m('not-a-node','y'))


def test_source_ui_and_asset_contract():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    route=(root/'src/ml_lab/ui/workbench_structured_routes.py').read_text()
    html=(root/'src/ml_lab/ui/templates/ml_lab_ui/workbench_structured.html').read_text()
    js=(root/'src/ml_lab/ui/static/ml_lab_hia_network.js').read_text()
    assert '/workbench/structured/hierarchy/network-accounting' in route
    assert 'explicit-network-information-accounting' in route and 'explicit-network-information-accounting' in js
    assert 'ml_lab_hia_network.js' in html
    assert 'data-hia-network-results' in html
    assert 'info.cross_level_pairs' in js
