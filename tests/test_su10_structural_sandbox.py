"""SU-10: opt-in UI orchestration of SU-05..09 without changing the source."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest
from jinja2 import Environment, FileSystemLoader

from ml_lab.structured import (
    StructuredContractError, StructuredObject, ObservationBindingPlan,
    VariableBinding, source_sha256,
)
from ml_lab.ui.structured_sandbox import analyze_sandbox

UI = Path(__file__).resolve().parents[1] / 'src/ml_lab/ui'
REVISION = 'sha256:' + 'c' * 64


def make_source(n=48):
    doc = {
        'schema': 'ml-lab.structured-object@1',
        'topology': {'kind': 'tree', 'capabilities': ['graph_edges', 'directed_edges',
            'acyclic', 'rooted_hierarchy', 'unique_parent', 'aligned_observations']},
        'nodes': [{'id': i, 'kind': 'uninterpreted', 'attributes': {}}
                  for i in ('root', 'a', 'b', 't')],
        'edges': [{'id': 'edge-' + i, 'source_id': 'root', 'target_id': i,
                   'kind': 'structural'} for i in ('a', 'b', 't')],
        'observation_unit': 'record', 'provenance': {'fixture': 'su10'}, 'observations': [],
    }
    for i in range(n):
        x = (i * 13) % 19
        doc['observations'].append({'observation_id': f'r{i:03}', 'values': {
            'a': {'status': 'present', 'value': x},
            'b': {'status': 'present', 'value': f'cat-{i % 4}'},
            'target': {'status': 'present', 'value': 'high' if x >= 9 else 'low'},
            'group': {'status': 'present', 'value': f'group-{i // 3}'},
        }})
    return StructuredObject.from_dict(doc)


def binding_for(source):
    return ObservationBindingPlan(source_sha256(source), 'record', (
        VariableBinding('a', ('a',)), VariableBinding('b', ('b',)),
        VariableBinding('target', ('t',))), group_variables=('group',))


def args(**overrides):
    data = {'source_revision': REVISION, 'mode': 'inspect', 'node_id': 'root',
            'binding_plan': None}
    data.update(overrides)
    return data


def test_source_original_graph_and_statistics_are_separate_and_unavailable_without_binding():
    source = make_source()
    original = source.to_json()
    report = analyze_sandbox(source, source_revision=REVISION,
                             request_data=args(node_id='a', include_information=True,
                                               uncertainty_targets=['a']))
    assert report['schema'] == 'ml-lab.structural-sandbox@1'
    assert report['observation_evidence']['reason'] == 'explicit_SU_05_binding_required'
    assert report['information']['status'] == 'unavailable'
    assert report['uncertainty']['status'] == 'unavailable'
    assert report['candidate']['status'] == 'unavailable'
    assert report['graph_overlay']['overlay'] == {'status': 'unavailable', 'metrics': []}
    assert source.to_json() == original


def test_selected_node_has_actual_observation_examples_missingness_variance_and_information():
    source = make_source()
    binding = binding_for(source)
    report = analyze_sandbox(source, source_revision=REVISION, request_data=args(
        node_id='a', binding_plan=binding.to_dict(),
        uncertainty_targets=['a'], information_target='target',
        include_information=True, information_bins=[['a', 'quantile', 4]]))
    assert report['observation_evidence']['scope_variable_ids'] == ['a']
    assert report['observation_evidence']['rows_displayed'] == 12
    assert report['observation_evidence']['rows_truncated']
    assert report['observation_evidence']['example_observations'][0]['observation_id'] == 'r000'
    assert report['uncertainty']['status'] == 'available'
    assert report['uncertainty']['support']['complete_case_rows'] == 48
    assert report['information']['status'] == 'available'
    assert report['information']['node']['node_id'] == 'a'
    assert report['information']['overlay']['evidence_is_node_scoped_not_edge_attributed']
    assert report['information']['estimator']['name'] == 'empirical_discrete'
    overlay = report['graph_overlay']
    assert overlay['node_id'] == 'a'
    assert overlay['overlay']['status'] == 'measured'
    assert all(metric['status'] == 'measured' or metric['value'] is None
               for metric in overlay['overlay']['metrics'])
    assert {m['name'] for m in overlay['overlay']['metrics']} >= {
        'candidate_joint_entropy', 'target_candidate_mi',
        'observed_variable_population_variance'}
    assert report['candidate']['status'] == 'unavailable'
    json.dumps(report, allow_nan=False)



def test_zero_variance_is_measured_zero_not_unavailable():
    source = make_source()
    doc = source.to_dict(); doc.pop('object_id')
    for row in doc['observations']:
        row['values']['a']['value'] = 0
    constant = StructuredObject.from_dict(doc)
    report = analyze_sandbox(constant, source_revision=REVISION,
        request_data=args(node_id='a', binding_plan=binding_for(constant).to_dict(),
                          uncertainty_targets=['a']))
    matched = [metric for metric in report['graph_overlay']['overlay']['metrics']
               if metric['name'] == 'observed_variable_population_variance']
    assert len(matched) == 1
    assert matched[0]['value'] == 0
    assert matched[0]['status'] == 'measured'


def test_preview_keeps_original_and_does_not_equate_structure_with_feature_loss():
    source = make_source()
    raw = source.to_json()
    report = analyze_sandbox(source, source_revision=REVISION, request_data=args(
        mode='preview', node_id='a', binding_plan=binding_for(source).to_dict(),
        operation='prune', observation_policy='preserve', retention={
            'kind': 'categorical_target', 'predictors': ['a', 'b'],
            'target': 'target', 'group_variable': 'group'}))
    candidate = report['candidate']
    assert candidate['status'] == 'available'
    assert candidate['manifest']['structural_loss']['removed_node_ids'] == ['a']
    assert candidate['manifest']['feature_impact']['dropped_variable_ids'] == []
    assert candidate['view']['schema'] == 'pah-shared.graph-view@1'
    assert not candidate['selection_survives']
    assert report['retention']['status'] == 'available'
    reference, proposed = report['retention']['report']['candidates']
    assert reference['metrics'] == proposed['metrics']
    assert source.to_json() == raw


def test_explicit_feature_deletion_affects_only_observed_columns_and_comparison():
    source = make_source()
    report = analyze_sandbox(source, source_revision=REVISION, request_data=args(
        mode='preview', node_id='a', binding_plan=binding_for(source).to_dict(),
        operation='prune', observation_policy='drop_exclusive', retention={
            'kind': 'categorical_target', 'predictors': ['a', 'b'],
            'target': 'target', 'group_variable': 'group'}))
    manifest = report['candidate']['manifest']
    assert manifest['feature_impact']['dropped_variable_ids'] == ['a']
    assert report['retention']['report']['candidates'][1]['evaluated_predictors'] == ['b']
    assert report['retention']['report']['split']['group_leakage_prevented']
    assert len(source.observations[0]['values']) == 4


def test_repeated_evaluation_is_deterministic_and_exportable():
    source = make_source()
    payload = args(mode='preview', node_id='b', operation='summarize',
                   binding_plan=binding_for(source).to_dict())
    left = analyze_sandbox(source, source_revision=REVISION, request_data=payload)
    right = analyze_sandbox(source, source_revision=REVISION, request_data=payload)
    assert left == right
    assert left['candidate']['manifest']['output_digest'].startswith('sha256:')
    assert left['retention']['reason'] == 'not_requested'


@pytest.mark.parametrize('payload,match', [
    (args(source_revision='sha256:' + 'a'*64), 'revision'),
    (args(node_id='not-present'), 'select'),
    (args(mode='train'), 'mode'),
    (args(mode='preview', operation='prune', observation_policy='drop_exclusive'), 'binding'),
    (args(extra='unsupported'), 'unsupported'),
    (args(uncertainty_targets=['a', 'a']), 'unique'),
    (args(mode='preview', operation='aggregate', max_depth=-1), 'max_depth'),
    (args(include_information='true'), 'boolean'),
])
def test_invalid_or_unverified_requests_fail_without_inference(payload, match):
    with pytest.raises(StructuredContractError, match=match):
        analyze_sandbox(make_source(), source_revision=REVISION, request_data=payload)


def test_binding_stale_complete_source_rejected_even_if_file_revision_is_fixed():
    source = make_source()
    binding = binding_for(source).to_dict()
    changed = source.to_dict(); changed.pop('object_id')
    changed['nodes'][0]['attributes']['note'] = 'revised'
    new_source = StructuredObject.from_dict(changed)
    with pytest.raises(StructuredContractError, match='fingerprint'):
        analyze_sandbox(new_source, source_revision=REVISION,
                        request_data=args(binding_plan=binding))


def test_unbound_dag_cannot_preview_hierarchical_pruning():
    source = make_source()
    doc = source.to_dict(); doc.pop('object_id')
    doc['topology'] = {'kind':'dag'}
    obj = StructuredObject.from_dict(doc)
    with pytest.raises(StructuredContractError, match='capability|rooted|depth'):
        analyze_sandbox(obj, source_revision=REVISION,
                        request_data=args(mode='preview', node_id='a', operation='prune'))


def test_template_is_opt_in_and_uses_same_shared_renderer():
    env = Environment(loader=FileSystemLoader(UI/'templates'))
    env.get_template('ml_lab_ui/workbench_structured.html')
    html = (UI/'templates/ml_lab_ui/workbench_structured.html').read_text()
    for needle in ('data-su10-sandbox', 'data-su10-action="inspect"',
                   'data-su10-action="preview"', 'data-su10-binding',
                   'data-su10-preview-surface', 'data-su10-action="export"'):
        assert needle in html
    js = (UI/'static/ml_lab_structured_sandbox.js').read_text()
    assert 'module.mountInteractiveGraph(' in js
    assert 'fetch(panel.dataset.sandboxUrl' in js
    assert 'mlwb:graph-select' in js
    assert 'mlwb:sandbox-overlay' in js
    assert 'innerHTML' not in js
    assert 'data-mlwb-shared-graph' in js
    if shutil.which('node'):
        for asset in ('ml_lab_structured_graph.js', 'ml_lab_structured_sandbox.js'):
            res = subprocess.run(['node', '--check', str(UI/'static'/asset)],
                                 capture_output=True, text=True)
            assert res.returncode == 0, res.stderr


def test_flask_route_requires_source_selection_json_and_explicit_header(tmp_path):
    flask = pytest.importorskip('flask')
    from ml_lab.ui import create_ui_blueprint
    app = flask.Flask(__name__)
    app.config['TESTING'] = True
    app.register_blueprint(create_ui_blueprint(name='su10',experiment_project_root=str(tmp_path)),
                           url_prefix='/lab')
    client = app.test_client()
    path = '/lab/workbench/structured'
    source = make_source()
    artifact = tmp_path / 'input.json'
    artifact.write_text(source.to_json(), encoding='utf-8')
    assert client.get(path+'/sandbox').status_code == 405
    assert client.post(path+'/sandbox',json=args()).status_code == 400
    assert client.post(path+'/sandbox',json=args(),headers={'X-ML-Lab-Sandbox':'explicit-analysis'}).status_code == 409
    selected = client.post(path, data={'section':'data','action':'select','path':'input.json','kind':'structured'})
    assert selected.status_code == 200
    graph = client.get(path+'/graph')
    assert graph.status_code == 200
    assert graph.json['binding_contract']['source_digest'] == source_sha256(source)
    data = args(source_revision=graph.json['view']['source_revision'],node_id='a')
    response = client.post(path+'/sandbox',json=data,
                           headers={'X-ML-Lab-Sandbox':'explicit-analysis'})
    assert response.status_code == 200
    assert response.json['node_id'] == 'a'
    assert artifact.read_text() == source.to_json()
    artifact.write_text('{}')
    assert client.post(path+'/sandbox',json=data,headers={'X-ML-Lab-Sandbox':'explicit-analysis'}).status_code == 409
