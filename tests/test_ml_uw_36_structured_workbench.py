"""ML-UW-36: headless bridge, contract, templates, optional Flask integration."""
import json
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader
from ml_lab.structured import StructuredContractError, StructuredObject
from ml_lab.ui.structured_workbench import read_source, analyze, independent_draft

FIXTURE = Path(__file__).parent / 'fixtures/structured_uncertainty/xml_tree.json'
UI = Path(__file__).resolve().parents[1] / 'src/ml_lab/ui'


def obj():
    doc = json.loads(FIXTURE.read_text())
    doc.pop('object_id', None)
    samples = [('0','0','a'),('0','1','a'),('1','1','b'),('1','0','b')] * 8
    doc['observations'] = [
        {'observation_id':f'id{i}', 'values':{name:{'status':'present','value':v}
                                           for name,v in zip(('x','y','group'),row)}}
        for i,row in enumerate(samples)]
    return StructuredObject.from_dict(doc)


def write(root, data):
    source = root / 'artifact.json'
    source.write_text(data.to_json(), encoding='utf-8')
    return source


def test_scoped_selection_fingerprint_and_reject_symlink(tmp_path):
    source = obj()
    write(tmp_path, source)
    result, meta = read_source(tmp_path, 'artifact.json', 'structured')
    assert result.to_json() == source.to_json()
    assert meta['rows'] == 32
    read_source(tmp_path, meta['path'], meta['kind'], meta['sha256'])
    write(tmp_path, StructuredObject.from_json(FIXTURE.read_text()))
    with pytest.raises(StructuredContractError, match='changed'):
        read_source(tmp_path, meta['path'], meta['kind'], meta['sha256'])
    (tmp_path/'linked.json').symlink_to(tmp_path/'artifact.json')
    with pytest.raises(StructuredContractError, match='symlink'):
        read_source(tmp_path, 'linked.json', 'structured')
    for path in ('../outside.json', '/etc/passwd', 'artifact.xml'):
        with pytest.raises(StructuredContractError):
            read_source(tmp_path, path, 'structured')


def test_adapter_does_not_invent_samples_and_dag_rejects_depth(tmp_path):
    (tmp_path/'raw.json').write_text('{"a":[1,2]}')
    source, meta = read_source(tmp_path, 'raw.json', 'json')
    assert meta['rows'] == 0
    with pytest.raises(StructuredContractError):
        analyze(source, 'profiling','information',{'x':'a','y':'b'})
    dag = Path(__file__).parent / 'fixtures/structured_uncertainty/shared_child_dag.json'
    (tmp_path/'dag.json').write_bytes(dag.read_bytes())
    source, meta = read_source(tmp_path, 'dag.json', 'node_edge')
    assert meta['topology'] == 'dag'
    with pytest.raises(StructuredContractError):
        analyze(source, 'representation','transform',{'mode':'depth','depth':'1'})


def test_descriptive_information_and_transform():
    source = obj()
    struct, _ = analyze(source, 'data','structure',{})
    assert struct['support']['aligned_observation_rows'] == 32
    stats, _ = analyze(source, 'profiling','observations',{'variables':'x,y'})
    assert stats['support']['aligned_rows'] == 32
    mi, _ = analyze(source, 'profiling','information',{'measure':'mutual_information','x':'x','y':'y'})
    assert mi['status'] == 'ok' and mi['value'] == pytest.approx(0)
    flat, transformed = analyze(source, 'representation','transform',{'mode':'flat'})
    assert flat['result']['support']['edges'] == 0
    assert len(transformed['observations']) == 32
    depth, transformed = analyze(source, 'representation','transform',{'mode':'depth','depth':'0'})
    assert depth['result']['support']['nodes'] < struct['support']['nodes']


def test_matched_comparison_and_probabilistic_are_explicit():
    source = obj()
    report, _ = analyze(source, 'comparison','compare', {
        'target':'y','predictors':'x','retained':'x','group':'group',
        'min_train':'2','min_test':'2'})
    assert {c['label'] for c in report['candidates']} == {'full','partial','flat'}
    assert report['support']['same_observation_ids_for_all_candidates']
    with pytest.raises(StructuredContractError):
        analyze(source, 'comparison','compare',{'target':'y','predictors':'x','retained':'x'})
    draft = independent_draft(source, ('x','y'))
    assert draft['arcs'] == []
    assert all(c['status']=='unfitted' for c in draft['conditionals'])
    verified, _ = analyze(source, 'tools','validate', {'model_json':json.dumps(draft)})
    assert verified['audit']['parameter_status'] == 'unfitted'
    fitted, spec = analyze(source, 'tools','fit', {'model_json':json.dumps(draft)})
    assert spec['conditionals'][0]['status']=='specified'
    posterior, _ = analyze(source, 'tools','query', {'model_json':json.dumps(spec),
                                                     'query':'y','evidence':'{"x":"1"}'})
    assert posterior['query']['status']=='ok'


def test_template_and_navigation_contract():
    templates = Environment(loader=FileSystemLoader(UI/'templates'))
    templates.get_template('ml_lab_ui/workbench_structured.html')
    navigation = (UI/'workbench_navigation.py').read_text()
    assert '"workbench_structured"' in navigation
    assert 'request.values.get("section")' in navigation
    assert '"data", "Data", "data_lab"' in navigation


def test_optional_mount_aware_browser(tmp_path):
    flask = pytest.importorskip('flask')
    from ml_lab.ui import create_ui_blueprint
    app=flask.Flask(__name__)
    app.config['TESTING']=True
    app.register_blueprint(create_ui_blueprint(name='uw36', experiment_project_root=str(tmp_path)),
                           url_prefix='/tools/ml-lab')
    client=app.test_client()
    write(tmp_path,obj())
    root='/tools/ml-lab/workbench/structured'
    assert client.get(root).status_code==200
    first=client.post(root,data={'section':'data','action':'select','path':'artifact.json','kind':'structured'})
    assert first.status_code==200 and b'32' in first.data
    report=client.post(root,data={'section':'profiling','action':'information','x':'x','y':'y','measure':'mutual_information'})
    assert report.status_code==200 and b'Scientific result' in report.data
    exported=client.post(root,data={'section':'data','action':'structure','download':'report'})
    assert exported.status_code==200 and exported.get_json()['support']['aligned_observation_rows']==32
    assert client.post(root,data={'section':'data','action':'clear'}).status_code==200
