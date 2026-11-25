"""SU-11: cross-format, bounded, reproducible scientific acceptance fixtures.

Fixtures here are deliberately *synthetic*. Real Code Analyzer, pyPIQUE, PAH
and project exports are only accepted by the explicit local CLI/host gates.
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from ml_lab.data.xml_ingest import inspect_xml_structure
from ml_lab.structured import (ObservationBindingPlan, StructuredContractError,
    StructuredObject, VariableBinding, source_sha256)
from ml_lab.structured.adapters import (HierarchyHandoffAdapter, JSONAdapter,
    XMLStructureAdapter, GraphAdapter, cross_module_adapters)
from ml_lab.ui.structured_sandbox import analyze_sandbox
from ml_lab.ui.structured_graph_view import make_shared_graph_payload, make_shared_graph_view
from ml_lab.ui.structured_workbench import read_source

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'tools/su11_acceptance.py'
FIXTURES = ROOT / 'tests/fixtures/structured_uncertainty'
REV = 'sha256:' + 'a' * 64


def synthetic_records(*, n=72):
    doc = {'schema': 'ml-lab.structured-object@1', 'topology': {'kind': 'tree'},
        'nodes': [{'id': name, 'kind': 'field', 'attributes': {}}
                  for name in ('root', 'feature', 'measure', 'target')],
        'edges': [{'id': name + '-edge', 'source_id': 'root',
                   'target_id': name, 'kind': 'contains'}
                  for name in ('feature', 'measure', 'target')],
        'observation_unit': 'record', 'observations': [],
        'provenance': {'synthetic': True, 'stage': 'su11'}}
    for i in range(n):
        doc['observations'].append({'observation_id': 'record-%03d' % i,
            'values': {
                'feature': {'status': 'present', 'value': 'on' if i % 2 else 'off'},
                'measure': {'status': 'present', 'value': float(i % 13)},
                'outcome': {'status': 'present', 'value': 'positive' if i % 2 else 'negative'},
                'cluster': {'status': 'present', 'value': 'c-%02d' % (i // 3)},
            }})
    obj = StructuredObject.from_dict(doc)
    binding = ObservationBindingPlan(source_sha256(obj), 'record', (
        VariableBinding('feature', ('feature',)),
        VariableBinding('measure', ('measure',)),
        VariableBinding('outcome', ('target',)),
    ), group_variables=('cluster',))
    return obj, binding


def test_complete_source_binding_science_candidate_retention_and_export_are_repeatable():
    source, plan = synthetic_records()
    original = source.to_json()
    payload = {'source_revision': REV, 'mode': 'preview', 'node_id': 'feature',
        'binding_plan': plan.to_dict(), 'uncertainty_targets': ['measure'],
        'include_information': True, 'information_target': 'outcome',
        'operation': 'prune', 'observation_policy': 'drop_exclusive',
        'retention': {'kind': 'categorical_target', 'predictors': ['feature','measure'],
            'target': 'outcome', 'group_variable': 'cluster'}}
    report = analyze_sandbox(source, source_revision=REV, request_data=payload)
    assert report == analyze_sandbox(source, source_revision=REV, request_data=payload)
    assert report['schema'] == 'ml-lab.structural-sandbox@1'
    assert report['observation_evidence']['aligned_rows'] == 72
    assert report['candidate']['manifest']['feature_impact']['dropped_variable_ids'] == ['feature']
    assert report['retention']['status'] == 'available'
    assert report['retention']['report']['split']['group_leakage_prevented']
    assert report['candidate']['view']['display']['coverage']['scope'] == 'exhaustive'
    assert report['graph_overlay']['overlay']['status'] in ('measured','unavailable')
    assert all(edge['kind'] == 'contains' for edge in source.edges)
    assert not report['candidate']['view']['overlays']['edges']
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert source.to_json() == original
    # Source topology remains a graph; the numerical statistics are attributed
    # only to explicitly mapped observation variables, not source edge counts.
    assert len(source.observations) == 72


def test_profile_from_actual_multi_record_xml_is_not_misrepresented_as_samples(tmp_path):
    raw = '''<root xmlns:t="urn:test"><record k="a"><t:amount>1</t:amount></record>
    <record k="b"><t:amount>2</t:amount><t:amount>4</t:amount></record>
    <record><t:amount /></record></root>'''
    file = tmp_path / 'actual.xml'; file.write_text(raw, encoding='utf-8')
    before = sha256(file.read_bytes()).hexdigest()
    adapted = XMLStructureAdapter().adapt(inspect_xml_structure(file))
    assert adapted.object.topology['kind'] == 'tree'
    assert len(adapted.object.observations) == 0
    assert 'record_level_text_and_attribute_values' in adapted.losses
    assert 'individual_occurrence_topology' in adapted.losses
    profile = next(n for n in adapted.object.nodes if n['id'].endswith('amount'))
    assert profile['attributes']['occurrence_count'] == 4
    assert adapted.object.provenance['source_sha256_claim'] == before
    assert sha256(file.read_bytes()).hexdigest() == before
    display = make_shared_graph_view(adapted.object, fingerprint=before)
    assert display['display']['coverage']['scope'] == 'exhaustive'
    assert not display['overlays']['nodes']


def test_json_and_synthetic_hierarchy_handoff_respect_source_identity():
    doc = JSONAdapter().adapt({'records': [{'a/b': 7}, {'a/b': None}], 'meta': {'~': 'x'}}).object
    assert '/a~1b' in next(n['id'] for n in doc.nodes if n['id'].endswith('a~1b'))
    assert not doc.observations
    handoff = {'schema': 'ml-lab.hierarchy-handoff@1', 'source_id': 'synthetic-test-export',
        'roots': ['n0'], 'nodes': [
            {'id': 'n0', 'kind': 'translation-unit', 'source_ref': {'line': 1}},
            {'id': 'n1', 'kind': 'function', 'source_ref': {'line': 2}},
        ], 'edges': [{'id': 'edge1', 'parent': 'n0', 'child': 'n1', 'kind': 'ast-child'}],
        'provenance': {'fixture': True}}
    converted = HierarchyHandoffAdapter().adapt(handoff)
    assert converted.object.topology['kind'] == 'tree'
    assert converted.object.observations == () or len(converted.object.observations) == 0
    assert converted.object.nodes[0]['source_ref']['line'] == 1
    assert 'not observations' in converted.warnings[0]
    assert make_shared_graph_view(converted.object, fingerprint='0'*64)['edges'][0]['traceable']
    with pytest.raises(StructuredContractError, match='roots'):
        HierarchyHandoffAdapter().adapt({**handoff, 'roots': ['n1']})


def test_synthetic_network_not_given_hierarchy_or_probabilistic_estimates():
    graph = {'schema': 'ml-lab.node-edge@1', 'topology': 'directed_graph',
        'nodes': [{'id': str(i), 'kind': 'node', 'attributes': {}} for i in range(5)],
        'edges': [{'id': 'e'+str(i), 'source_id': str(i), 'target_id': str((i+1)%5),
                  'kind': 'network'} for i in range(5)]}
    obj = GraphAdapter().adapt(graph).object
    assert 'rooted_hierarchy' not in obj.capabilities
    view = make_shared_graph_view(obj, fingerprint='0'*64)
    assert view['display']['layout'] == 'circular'
    assert all(not edge['traceable'] for edge in view['edges'])
    report = analyze_sandbox(obj, source_revision=REV,
        request_data={'source_revision': REV, 'mode': 'inspect', 'node_id': '0'})
    assert report['information']['status'] == 'unavailable'
    assert report['uncertainty']['status'] == 'unavailable'
    assert not report['graph_overlay']['overlay']['metrics']


def test_deep_tree_and_wide_graph_have_exhaustive_display_until_explicit_cap():
    n = 800
    doc = {'schema': 'ml-lab.structured-object@1', 'topology': {'kind': 'tree'},
           'nodes': [{'id': 'n%d'%i, 'kind':'node', 'attributes':{}} for i in range(n)],
           'edges': [{'id':'e%d'%i,'source_id':'n%d'%i,'target_id':'n%d'%(i+1),
                      'kind':'parent'} for i in range(n-1)],
           'observations': [], 'observation_unit':'graph', 'provenance': {}}
    obj = StructuredObject.from_dict(doc)
    view = make_shared_graph_view(obj, fingerprint='1'*64)
    assert len(view['nodes']) == 800
    assert len(view['edges']) == 799
    assert view['display']['coverage'] == {'scope':'exhaustive','total_nodes':800,'total_edges':799}
    assert {node['group'] for node in view['nodes']} <= {'depth-%02d'%i for i in range(40)}
    # The display is intentionally capped: never truncate invisibly.
    extra = {'schema':'ml-lab.structured-object@1','topology':{'kind':'undirected_graph'},
             'nodes':[{'id':'node%d'%i,'kind':'n','attributes':{}} for i in range(5001)],
             'edges':[], 'observations':[], 'observation_unit':'graph', 'provenance':{}}
    big = StructuredObject.from_dict(extra)
    assert len(big.nodes) == 5001
    with pytest.raises(StructuredContractError, match='exceeds shared viewer'):
        make_shared_graph_view(big, fingerprint='1'*64)


def test_local_artifact_isolation_stale_revision_rejection_and_symlinks(tmp_path):
    source, binding = synthetic_records(n=9)
    a, b = tmp_path/'A', tmp_path/'B'
    a.mkdir(); b.mkdir()
    for root in (a,b):
        (root/'source.json').write_text(source.to_json(), encoding='utf-8')
    first, meta_a = read_source(a,'source.json','structured')
    _, meta_b = read_source(b,'source.json','structured')
    assert meta_a['sha256'] == meta_b['sha256']
    # The cookie route is instance-and-root scoped; server never reads outside
    # the configured project root even when the same name exists elsewhere.
    (b/'source.json').write_text('{}')
    with pytest.raises(StructuredContractError, match='changed'):
        read_source(b,'source.json','structured',meta_b['sha256'])
    assert read_source(a,'source.json','structured',meta_a['sha256'])[0].to_json() == first.to_json()
    (a/'link.json').symlink_to(b/'source.json')
    with pytest.raises(StructuredContractError, match='symlink'):
        read_source(a,'link.json','structured')
    with pytest.raises(StructuredContractError, match='project-relative'):
        read_source(a,'../B/source.json','structured')
    assert read_source(a,'source.json','structured',meta_a['sha256'])[0].to_json() == source.to_json()


def _run_cli(root, *entries, viewer=None):
    args = [sys.executable,str(SCRIPT),'--project-root',str(root)]
    for entry in entries:
        args.extend(['--artifact',entry])
    if viewer:
        args.extend(['--viewer',str(viewer)])
    env = dict(os.environ, PYTHONPATH=str(ROOT/'src'))
    p = subprocess.run(args,capture_output=True,text=True,timeout=35,env=env)
    return p, json.loads(p.stdout)


def test_real_local_artifact_runner_sorts_pass_fail_and_skipped_gates(tmp_path):
    path = tmp_path/'example.json'
    path.write_bytes((FIXTURES/'xml_tree.json').read_bytes())
    p, result = _run_cli(tmp_path, 'structured:example.json')
    assert p.returncode == 0, p.stderr
    assert result['status'] == 'passed'
    gates = result['artifacts'][0]['gates']
    assert gates['interactive_graph']['scope'] == 'exhaustive'
    assert gates['shared_viewer']['status'] == 'skipped'
    assert result['environment_gates']['live_flask_pah_browser'] == 'not_executed_by_this_headless_runner'
    p, result = _run_cli(tmp_path, 'structured:example.json','structured:missing.json')
    assert p.returncode == 1
    assert result['status'] == 'failed'
    assert result['artifacts'][0]['status'] == 'passed'
    assert result['artifacts'][1]['status'] == 'failed'
    assert result['artifacts'][1]['error']
    p, result = _run_cli(tmp_path, 'structured:../outside.json')
    assert p.returncode == 1
    assert 'project-relative' in result['artifacts'][0]['error']


def test_runner_accepts_explicit_bound_rows_and_rejects_stale_binding(tmp_path):
    obj, binding = synthetic_records(n=18)
    (tmp_path/'input.json').write_text(obj.to_json(), encoding='utf-8')
    (tmp_path/'plan.json').write_text(binding.to_json(), encoding='utf-8')
    argv = [sys.executable, str(SCRIPT), '--project-root', str(tmp_path),
            '--artifact', 'structured:input.json', '--binding','plan.json',
            '--node', 'feature']
    env = dict(os.environ, PYTHONPATH=str(ROOT/'src'))
    res = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=25)
    assert res.returncode == 0, res.stderr
    item = json.loads(res.stdout)['artifacts'][0]
    assert item['gates']['observation_binding']['status'] == 'passed'
    assert item['gates']['observation_binding']['sample_rows'] == 18
    assert item['gates']['observation_binding']['scope'] == 'subtree'
    # When source bytes are changed the previously valid binding must fail.
    altered = obj.to_dict()
    altered.pop('object_id')
    altered['nodes'][0]['attributes']['changed'] = True
    (tmp_path/'input.json').write_text(StructuredObject.from_dict(altered).to_json())
    res = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=25)
    assert res.returncode != 0
    assert json.loads(res.stdout)['artifacts'][0]['status'] == 'failed'
    assert 'fingerprint' in json.loads(res.stdout)['artifacts'][0]['error']


def test_local_runner_checks_real_shared_node_module_when_explicit(tmp_path):
    viewer = os.environ.get('SU11_SHARED_VIEWER')
    if not viewer or not Path(viewer).is_file() or not shutil.which('node'):
        pytest.skip('supply SU11_SHARED_VIEWER path to installed PAH_Shared .mjs')
    (tmp_path/'source.json').write_bytes((FIXTURES/'xml_tree.json').read_bytes())
    p,result = _run_cli(tmp_path,'structured:source.json',viewer=viewer)
    assert p.returncode == 0, p.stderr
    assert result['artifacts'][0]['gates']['shared_viewer']['status'] == 'passed'


def test_ui_navigation_and_js_remain_single_renderer_consumers():
    base = ROOT/'src/ml_lab/ui'
    from jinja2 import Environment, FileSystemLoader
    env = Environment(loader=FileSystemLoader(base/'templates'))
    env.get_template('ml_lab_ui/workbench_structured.html')
    ui = (base/'templates/ml_lab_ui/workbench_structured.html').read_text()
    assert 'section == \'data\'' in ui
    assert 'data-su10-action="preview"' in ui
    assert 'data-su10-action="inspect"' in ui
    js = (base/'static/ml_lab_structured_graph.js').read_text()
    sandbox = (base/'static/ml_lab_structured_sandbox.js').read_text()
    assert 'await import(root.dataset.graphModuleUrl)' in js
    assert 'module.mountInteractiveGraph(' in sandbox
    assert 'innerHTML' not in sandbox and 'innerHTML' not in js
    assert 'fetch(panel.dataset.sandboxUrl' in sandbox
    for section in ('Data','Feature Profiling','Representation','Comparison','General ML Tools'):
        assert section in (base/'workbench_structured_routes.py').read_text()
    if shutil.which('node'):
        for file in ('ml_lab_structured_graph.js','ml_lab_structured_sandbox.js'):
            check = subprocess.run(['node','--check',str(base/'static'/file)],
                capture_output=True,text=True,timeout=10)
            assert check.returncode == 0, check.stderr
