"""HIA-04: measured atomic results require explicit SU-05 identity mapping."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from ml_lab.structured import (
    StructuredContractError, StructuredObject, atomic_node_statistics, source_sha256,
)
from ml_lab.structured.atomic_node_statistics import ATOMIC_SCHEMA

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / 'src/ml_lab/ui'
FIXTURE = ROOT / 'tests/fixtures/structured_uncertainty'


def cell(value):
    if value is None:
        return {'status':'explicit_null','value':None,'source_reason':'explicit_xsi_nil'}
    return {'status':'present','value':value}


def source(values, *, extra=None, observation_unit='record'):
    # Each row is a real aligned source observation; neither node nor edge is a sample.
    return StructuredObject.from_dict({
        'schema':'ml-lab.structured-object@1',
        'topology':{'kind':'tree'},
        'nodes':[{'id':'root','kind':'folder','attributes':{}},
                 {'id':'atom','kind':'value','attributes':{}}],
        'edges':[{'id':'edge','kind':'contains','source_id':'root','target_id':'atom'}],
        'observation_unit':observation_unit,
        'observations':[{'observation_id':f'o{i}', 'values':{
            'x':value,'other':cell((extra or list(range(len(values))))[i]),
        }} for i,value in enumerate(values)],
        'provenance':{},
    })


def report(obj, **kw):
    return atomic_node_statistics(obj, node_id='atom', variable_id='x', **kw)


def test_exact_node_scalar_distribution_entropy_provenance_no_structural_inference():
    obj=source([cell('a'),cell('b'),cell('a'),cell(None)])
    before=obj.to_json()
    result=report(obj)
    assert result['schema']==ATOMIC_SCHEMA
    assert result['source_digest']==source_sha256(obj)
    assert result['node_context']['level']==1
    assert result['support']['aligned_rows']==4
    assert result['support']['observed_scalar_rows']==3
    assert result['support']['node_count_is_not_sample_count']
    assert result['status_counts']=={'explicit_null':1,'present':3}
    assert result['missing_reason_counts']=={'explicit_xsi_nil':1}
    assert result['status_rates']['explicit_null']==.25
    assert math.isclose(result['entropy']['bits'], -(2/3)*math.log2(2/3)-(1/3)*math.log2(1/3))
    assert result['entropy']['support']==3
    assert sorted(x['count'] for x in result['scalar_distribution'])==[1,2]
    assert result['numeric']['statistics'] is None
    assert not result['provenance']['statistical_edges_inferred_from_structure']
    assert result['provenance']['binding_plan']['variables'][0]['node_ids']==['atom']
    assert obj.to_json()==before
    assert json.loads(json.dumps(result, allow_nan=False))==result


def test_numeric_sample_variance_and_integer_discrete_entropy():
    obj=source([cell(1),cell(2),cell(3),cell(None)])
    result=report(obj)
    assert result['numeric']['status']=='available'
    assert result['numeric']['statistics']['mean']==2
    assert result['numeric']['statistics']['sample_variance']==1
    assert math.isclose(result['entropy']['bits'], math.log2(3))
    assert result['entropy']['discretization'] is None


def test_float_entropy_needs_explicit_bins_numeric_summary_remains():
    obj=source([cell(.1),cell(.2),cell(.8),cell(.9)])
    no_bins=report(obj)
    assert no_bins['entropy']['bits'] is None
    assert no_bins['entropy']['status']=='float_requires_explicit_binning'
    assert no_bins['numeric']['statistics']['count']==4
    binned=report(obj, bins={'mode':'equal_width','bins':2})
    assert binned['entropy']['status']=='ok'
    assert binned['entropy']['bits']==1
    assert binned['entropy']['discretization']['actual_bins']==2


def test_insufficient_samples_and_source_only_no_fabricated_entropy():
    obj=source([cell('a'),cell(None)])
    assert report(obj)['entropy']['status']=='insufficient_samples'
    raw=StructuredObject.from_json((FIXTURE/'xml_tree.json').read_text(encoding='utf-8'))
    record=raw.to_dict(); record['observations']=[]; record['topology'].pop('capabilities')
    raw=StructuredObject.from_dict(record)
    result=atomic_node_statistics(raw,node_id='name',variable_id='declared_column')
    assert result['support']['aligned_rows']==0
    assert result['entropy']['status']=='no_observation_rows'
    assert result['numeric']['statistics'] is None


def test_repeated_values_not_silently_averaged_or_discarded():
    multi={'status':'multiple','value':None,'occurrences':[cell(2),cell(4)]}
    obj=source([multi,cell(2),cell(4)])
    preserved=report(obj)
    assert preserved['entropy']['status']=='unresolved_repeated_values'
    assert preserved['status_counts']['multiple']==1
    assert preserved['scalar_distribution_excludes_repeated_rows']
    assert preserved['numeric']['statistics']['count']==2
    assert preserved['numeric']['excludes_unresolved_repeated_rows']
    with pytest.raises(StructuredContractError, match='scalar_only'):
        report(obj, occurrence_policy='scalar_only')
    averaged=report(obj, occurrence_policy='mean_numeric',bins={'mode':'equal_width','bins':2})
    assert averaged['support']['observed_scalar_rows']==3
    assert averaged['entropy']['status']=='ok'
    assert averaged['numeric']['statistics']['mean']==3
    assert averaged['status_counts'].get('multiple',0)==0
    assert averaged['original_binding_evidence']['repeated_rows']==1
    assert averaged['original_binding_evidence']['recorded_occurrences_in_repeated_rows']==2
    assert averaged['provenance']['binding_plan']['variables'][0]['occurrence_policy']=='mean_numeric'


def test_unsupported_mixed_type_and_invalid_explicit_mapping():
    obj=source([cell('x'),cell(12)])
    result=report(obj)
    assert result['numeric']['status']=='mixed_or_non_numeric'
    assert result['entropy']['status']=='ok'
    assert {entry['type'] for entry in result['scalar_distribution']}=={'str','int'}
    with pytest.raises(StructuredContractError, match='unknown original source node'):
        atomic_node_statistics(obj,node_id='nonexistent',variable_id='x')
    with pytest.raises(StructuredContractError, match='unknown aligned observation'):
        atomic_node_statistics(obj,node_id='atom',variable_id='not_a_column')
    with pytest.raises(StructuredContractError, match='bin'):
        report(obj,bins={'mode':'quantile','bins':1})
    with pytest.raises(StructuredContractError, match='min_samples'):
        report(obj,min_samples=1)


def test_route_opt_in_stale_source_rejection_and_no_db_writes(tmp_path):
    flask=pytest.importorskip('flask')
    from ml_lab.ui import create_ui_blueprint
    app=flask.Flask(__name__)
    app.config['TESTING']=True
    app.register_blueprint(create_ui_blueprint(name='hia04', experiment_project_root=str(tmp_path)),
                           url_prefix='/lab')
    client=app.test_client()
    path=tmp_path/'rows.json'
    obj=source([cell('yes'),cell('no'),cell(None)])
    path.write_text(obj.to_json(), encoding='utf-8')
    base='/lab/workbench/structured'
    assert client.post(base,data={'section':'data','action':'select','path':'rows.json',
                                  'kind':'structured'}).status_code==200
    asset=client.get(base+'/graph').json
    url=base+'/hierarchy/atomic'
    payload={'source_revision':asset['view']['source_revision'], 'node_id':'atom',
             'variable_id':'x','occurrence_policy':'preserve','bins':None,'min_samples':2}
    headers={'X-ML-Lab-HIA':'explicit-atomic-statistics'}
    assert client.post(url,json=payload).status_code==400  # explicit opt-in required
    assert client.post(url,json={**payload,'source_revision':'sha256:'+'0'*64},headers=headers).status_code==409
    answer=client.post(url,json=payload,headers=headers)
    assert answer.status_code==200,answer.json
    assert answer.json['report']['entropy']['status']=='ok'
    assert answer.json['report']['variable_id']=='x'
    assert client.post(url,json={**payload,'extra':'unapproved'},headers=headers).status_code==409
    path.write_text('{}', encoding='utf-8')
    assert client.post(url,json=payload,headers=headers).status_code==409
    assert not list(tmp_path.glob('*.db'))


def test_ui_contract_preserves_existing_graph_controls_and_opt_in_atomic_controls():
    template=(UI/'templates/ml_lab_ui/workbench_structured.html').read_text()
    script=(UI/'static/ml_lab_hia_atomic.js').read_text()
    routes=(UI/'workbench_structured_routes.py').read_text()
    assert 'data-hia-atomic-column' in template
    assert 'data-hia-atomic-action="compute"' in template
    assert 'data-hia-atomic-url' in template
    assert 'data-hia-workspace' in template
    assert 'data-su10-sandbox' in template
    assert 'data-graph-action="fit"' in template
    assert "'X-ML-Lab-HIA':'explicit-atomic-statistics'" in script
    assert "event.detail?.payload" in script
    assert 'workbench_structured_hierarchy_atomic' in routes
    assert "'ml_lab_hia_atomic.js'" in routes
