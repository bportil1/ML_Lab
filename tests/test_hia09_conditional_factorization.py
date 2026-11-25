"""HIA-09 exact, opt-in, in-sample independence model comparisons."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml_lab.structured import (
    StructuredObject, StructuredContractError, source_sha256,
    HIA_FACTORIZATION_SCHEMA, analyze_hierarchical_factorized_level,
    analyze_hierarchical_conditional_level,
)

ROOT = Path(__file__).resolve().parents[1]


def m(node, col):
    return {'node_id': node, 'variable_id': col}


def fixture(observations, *, kind='forest'):
    columns = ('x1', 'x2', 'z1', 'z2')
    return StructuredObject.from_dict({
        'schema': 'ml-lab.structured-object@1', 'topology': {'kind': kind},
        'nodes': [{'id': node, 'kind': 'atom', 'attributes': {}}
                  for node in ('r1', 'r2', 'a', 'b')],
        'edges': [{'id': 'edge1', 'kind': 'source_link', 'source_id': 'r1', 'target_id': 'a'},
                  {'id': 'edge2', 'kind': 'source_link', 'source_id': 'r2', 'target_id': 'b'}],
        'observation_unit': 'record',
        'observations': [{'observation_id': str(i), 'values': {
            name: {'status': 'present', 'value': row[j]} if row[j] is not None else
                  {'status': 'explicit_null', 'value': None, 'source_reason': 'path_absent'}
            for j, name in enumerate(columns)}} for i, row in enumerate(observations)],
        'provenance': {},
    })


def rows_independent(repeats=12):
    return [(a,b,z1,z2) for _ in range(repeats)
            for z1 in (0,1) for z2 in (0,1) for a in (0,1) for b in (0,1)]


def analyze(source, **overrides):
    args = dict(level=1, mappings=[m('a','x1'), m('b','x2')],
                preceding_mappings=[m('r1','z1'),m('r2','z2')],
                factorization_mode='shared_joint_prior', parent_sets={},
                assumption_confirmed=True)
    args.update(overrides)
    return analyze_hierarchical_factorized_level(source, **args)


def test_independent_factors_have_zero_gap_and_all_source_ids():
    source=fixture(rows_independent())
    before=source.to_json()
    result=analyze(source)
    model=result['model']
    assert result['schema']==HIA_FACTORIZATION_SCHEMA
    assert result['source_digest']==source_sha256(source)
    assert result['level_node_ids']==['a','b']
    assert result['preceding_level_node_ids']==['r1','r2']
    assert result['declared_parent_sets']=={'a':['r1','r2'],'b':['r1','r2']}
    assert result['prior_joint_distribution_factorized'] is False
    assert result['assumption_confirmed'] is True
    assert model['status']=='ok' and model['support']==len(rows_independent())
    assert model['empirical_joint_conditional_entropy_bits']==pytest.approx(2)
    assert model['factorized_model_conditional_entropy_bits']==pytest.approx(2)
    assert model['empirical_model_kl_divergence_bits']==pytest.approx(0,abs=1e-10)
    assert model['empirical_conditional_total_correlation_bits']==pytest.approx(0,abs=1e-10)
    assert model['omitted_prior_information_bits']==pytest.approx(0,abs=1e-10)
    assert len(model['per_joint_prior_stratum'])==4
    assert json.loads(json.dumps(result,allow_nan=False))==result
    assert source.to_json()==before


def test_conditional_redundancy_equals_kl_when_all_prior_parents_used():
    observations=[(x,x,z1,z2) for _ in range(12)
                  for z1 in (0,1) for z2 in (0,1) for x in (0,1)]
    result=analyze(fixture(observations))
    model=result['model']
    assert model['empirical_joint_conditional_entropy_bits']==pytest.approx(1)
    assert model['factorized_cross_entropy_bits']==pytest.approx(2)
    assert model['empirical_model_kl_divergence_bits']==pytest.approx(1)
    assert model['empirical_conditional_total_correlation_bits']==pytest.approx(1)
    assert model['omitted_prior_information_bits']==pytest.approx(0)
    assert all(s['empirical_model_kl_divergence_given_stratum_bits']==pytest.approx(1)
               for s in model['per_joint_prior_stratum'])


def test_explicit_parent_sets_isolate_omitted_prior_information():
    observations=[(z1,z2,z1,z2) for _ in range(16)
                  for z1 in (0,1) for z2 in (0,1)]
    source=fixture(observations)
    complete=analyze(source)
    assert complete['model']['empirical_model_kl_divergence_bits']==pytest.approx(0)
    no_parents=analyze(source,factorization_mode='explicit_prior_parent_sets',
                       parent_sets={'a':[], 'b':[]})
    model=no_parents['model']
    assert model['empirical_model_kl_divergence_bits']==pytest.approx(2)
    assert model['empirical_conditional_total_correlation_bits']==pytest.approx(0)
    assert model['omitted_prior_information_bits']==pytest.approx(2)
    assert model['decomposition_difference_bits']==pytest.approx(0,abs=1e-10)
    exact=analyze(source,factorization_mode='explicit_prior_parent_sets',
                  parent_sets={'a':['r1'],'b':['r2']})
    assert exact['model']['empirical_model_kl_divergence_bits']==pytest.approx(0)


def test_conditional_xor_requires_joint_prior_vector():
    observations=[(z1^z2, 1, z1,z2) for _ in range(16)
                  for z1 in (0,1) for z2 in (0,1)]
    source=fixture(observations)
    full=analyze(source)
    assert full['model']['empirical_model_kl_divergence_bits']==pytest.approx(0)
    incomplete=analyze(source,factorization_mode='explicit_prior_parent_sets',
                       parent_sets={'a':['r1'],'b':['r2']})
    assert incomplete['model']['empirical_model_kl_divergence_bits']==pytest.approx(1)
    assert incomplete['model']['omitted_prior_information_bits']==pytest.approx(1)


def test_no_automatic_independence_or_source_edge_parents():
    source=fixture(rows_independent())
    with pytest.raises(StructuredContractError,match='assumption_confirmed'):
        analyze(source,assumption_confirmed=False)
    with pytest.raises(StructuredContractError,match='assumption_confirmed'):
        analyze(source,assumption_confirmed=1)
    with pytest.raises(StructuredContractError,match='factorization_mode'):
        analyze(source,factorization_mode='guess_parents_from_graph')
    with pytest.raises(StructuredContractError,match='parent overrides'):
        analyze(source,parent_sets={'a':['r1']})
    for parents in ({'a':['r1']}, {'a':['b'],'b':[]},
                    {'a':['r1','r1'],'b':[]}, {'a':['r1'],'b':['unknown']},
                    {'a':'r1','b':[]}, {'a':[],'b':[],'missing':[]}):
        with pytest.raises(StructuredContractError):
            analyze(source,factorization_mode='explicit_prior_parent_sets',parent_sets=parents)


def test_non_estimable_cohort_does_not_fabricate_model():
    source=fixture([(0,1,0,1),(1,0,None,1),(0,1,1,0)])
    result=analyze(source,min_samples=4)
    assert result['model']['status']=='insufficient_samples'
    assert result['model']['empirical_model_kl_divergence_bits'] is None
    assert result['model']['per_joint_prior_stratum']==[]
    assert result['empirical_level_conditioned_information']['support']==2


def test_sparse_prior_strata_and_target_do_not_change_base_cohort():
    source=fixture(rows_independent(repeats=1))
    result=analyze(source,min_stratum_samples=10)
    assert result['model']['status']=='insufficient_conditioning_support'
    assert result['model']['factorized_cross_entropy_bits'] is None
    assert result['empirical_level_conditioned_information']['under_supported_strata']==4


def test_subset_mapping_and_float_bins_are_explicit():
    source=fixture([(float(x),x,z1,z2) for _ in range(4)
                    for x in (0,1) for z1 in (0,1) for z2 in (0,1)])
    invalid=analyze(source)
    assert invalid['model']['status']=='float_requires_explicit_binning'
    valid=analyze(source,bins={'x1':{'mode':'equal_width','bins':2}})
    assert valid['model']['status']=='ok'
    assert valid['model']['empirical_model_kl_divergence_bits']==pytest.approx(1)


def test_root_level_has_no_prior_and_still_requires_explicit_consent():
    source=fixture(rows_independent())
    result=analyze_hierarchical_factorized_level(
        source,level=0,mappings=[m('r1','z1'),m('r2','z2')],
        preceding_mappings=[],assumption_confirmed=True,
        factorization_mode='shared_joint_prior',parent_sets={})
    assert result['model']['status']=='ok'
    assert result['model']['conditioning_strata']==1
    assert result['model']['empirical_joint_conditional_entropy_bits']==pytest.approx(2)
    assert result['model']['empirical_model_kl_divergence_bits']==pytest.approx(0)


def test_optional_target_keeps_independent_complete_case_cohort():
    base=fixture(rows_independent(repeats=3)).to_dict()
    base['nodes'].append({'id':'target','kind':'atom','attributes':{}})
    base['edges'].append({'id':'e3','kind':'source_link','source_id':'a','target_id':'target'})
    for i, row in enumerate(base['observations']):
        if i % 3 == 0:
            row['values']['y']={'status':'explicit_null','value':None,
                                'source_reason':'path_absent'}
        else:
            row['values']['y']={'status':'present','value':row['values']['x1']['value']}
    result=analyze(StructuredObject.from_dict(base),target=m('target','y'))
    assert result['model']['status']=='ok'
    assert result['model']['support']==len(base['observations'])
    assert result['empirical_target_information']['support'] < result['model']['support']
    assert result['model']['empirical_model_kl_divergence_bits']==pytest.approx(0)


def test_empirical_estimator_unchanged_by_model_use():
    source=fixture(rows_independent())
    before=analyze_hierarchical_conditional_level(source,level=1,
        mappings=[m('a','x1'),m('b','x2')],
        preceding_mappings=[m('r1','z1'),m('r2','z2')])
    analyze(source)
    after=analyze_hierarchical_conditional_level(source,level=1,
        mappings=[m('a','x1'),m('b','x2')],
        preceding_mappings=[m('r1','z1'),m('r2','z2')])
    assert before==after
    assert before['independence_model']=='empirical_joint_no_factorization'


def test_ui_endpoint_and_opt_in_controller_contract():
    routes=(ROOT/'src/ml_lab/ui/workbench_structured_routes.py').read_text()
    html=(ROOT/'src/ml_lab/ui/templates/ml_lab_ui/workbench_structured.html').read_text()
    js=(ROOT/'src/ml_lab/ui/static/ml_lab_hia_factorization.js').read_text()
    old=(ROOT/'src/ml_lab/ui/static/ml_lab_hia_conditional.js').read_text()
    assert "@blueprint.post('/workbench/structured/hierarchy/factorized-information')" in routes
    assert "'ml_lab_hia_factorization.js'" in routes
    assert 'invalid HIA factorization analysis request fields' in routes
    assert 'data-hia-factorization-confirm' in html
    assert 'data-hia-factorization-url' in html
    assert "'ml_lab_hia_factorization.js', v=graph_controller_revision" in html
    assert "'X-ML-Lab-HIA':'explicit-conditional-factorization'" in js
    assert "data-hia-factorization-action" in html
    assert 'mlwb:hia-conditional-mappings' in old
    assert 'mlwb:hia-conditional-mappings' in js
