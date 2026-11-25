"""HIA-06: source-complete branch inventory, observed-subset information estimates."""
import json
from pathlib import Path

import pytest

from ml_lab.structured import (
    HIA_BRANCH_SCHEMA, StructuredContractError, StructuredObject,
    analyze_hierarchical_branch, source_sha256,
)

ROOT = Path(__file__).resolve().parents[1]


def fixture(x=None, y=None, z=None):
    x = x if x is not None else [0, 0, 1, 1] * 4
    y = y if y is not None else list(x)
    z = z if z is not None else list(x)
    entries = []
    for i, (a, b, c) in enumerate(zip(x, y, z)):
        cells = {}
        for key, value in (('x', a), ('y', b), ('z', c)):
            cells[key] = ({'status': 'present', 'value': value} if value is not None
                          else {'status': 'explicit_null', 'value': None, 'source_reason': 'path_absent'})
        entries.append({'observation_id': f'o{i}', 'values': cells})
    return StructuredObject.from_dict({
        'schema': 'ml-lab.structured-object@1',
        'topology': {'kind': 'tree'},
        'nodes': [{'id': name, 'kind': 'atom', 'attributes': {}}
                  for name in ('root', 'a', 'b', 'leaf')],
        'edges': [{'id': key, 'kind': 'source_link', 'source_id': a, 'target_id': b}
                  for key, a, b in (('e1', 'root', 'a'), ('e2', 'root', 'b'), ('e3', 'a', 'leaf'))],
        'observation_unit': 'record', 'observations': entries,
        'provenance': {},
    })


def mapped():
    return [{'node_id': 'a', 'variable_id': 'x'},
            {'node_id': 'leaf', 'variable_id': 'y'}]


def analyze(source=None, **kwargs):
    mappings = kwargs.pop('mappings', mapped())
    return analyze_hierarchical_branch(source or fixture(), branch_root_id='a',
                                       mappings=mappings, **kwargs)


def test_perfect_redundancy_full_branch_and_context_parent():
    source = fixture()
    before = source.to_json()
    r = analyze(source, context={'node_id':'root', 'variable_id':'z'})
    assert r['schema'] == HIA_BRANCH_SCHEMA
    assert r['source_digest'] == source_sha256(source)
    assert r['branch_node_ids'] == ['a', 'leaf']
    assert r['branch_node_count'] == 2 and r['unmapped_node_count'] == 0
    assert r['statistical_scope'] == 'complete_branch'
    stats = r['branch_information']
    assert stats['status'] == 'ok'
    assert stats['support'] == 16
    assert stats['joint_entropy_bits'] == pytest.approx(1)
    assert stats['sum_atomic_entropy_bits'] == pytest.approx(2)
    assert stats['total_correlation_bits'] == pytest.approx(1)
    assert [v['conditional_entropy_given_other_mapped_members_bits']
            for v in stats['per_mapped_node']] == pytest.approx([0, 0])
    ctx = r['context_information']
    assert ctx['status'] == 'ok'
    assert ctx['context_is_direct_source_parent'] is True
    assert ctx['branch_conditional_entropy_bits'] == pytest.approx(0)
    assert ctx['branch_context_mutual_information_bits'] == pytest.approx(1)
    assert ctx['conditional_total_correlation_bits'] == pytest.approx(0)
    assert r['binding_plan']['variables'][0]['node_ids'] == ['a']
    assert 'source' in r['source_topology'] or r['source_topology'] == 'tree'
    assert json.loads(json.dumps(r, allow_nan=False)) == r
    assert source.to_json() == before


def test_independent_nodes_total_correlation_is_zero():
    x = [0, 0, 1, 1] * 4
    y = [0, 1, 0, 1] * 4
    r = analyze(fixture(x, y, [1] * 16))
    stats = r['branch_information']
    assert stats['status'] == 'ok'
    assert stats['joint_entropy_bits'] == pytest.approx(2)
    assert stats['sum_atomic_entropy_bits'] == pytest.approx(2)
    assert stats['total_correlation_bits'] == pytest.approx(0)
    assert [i['mi_with_other_mapped_members_bits'] for i in stats['per_mapped_node']] == pytest.approx([0, 0])
    assert r['context_information'] is None


def test_partially_mapped_branch_not_mislabeled_complete():
    r = analyze_hierarchical_branch(fixture(), branch_root_id='root', mappings=mapped())
    assert r['branch_node_count'] == 4
    assert r['mapped_node_count'] == 2
    assert r['unmapped_node_count'] == 2
    assert r['statistical_scope'] == 'explicit_mapped_branch_subset'
    assert not r['complete_branch_statistical_coverage']
    assert r['branch_information']['joint_entropy_bits'] == pytest.approx(1)


def test_context_support_differs_from_branch_only_support():
    x = [0, 0, 1, 1, 0, 0, 1, 1]
    y = list(x)
    z = [None, None, 0, 1, 0, 0, 1, 1]
    r = analyze(fixture(x, y, z), context={'node_id':'root', 'variable_id':'z'})
    assert r['branch_information']['status'] == 'ok'
    assert r['branch_information']['support'] == 8
    assert r['context_information']['status'] == 'ok'
    assert r['context_information']['support'] == 6
    assert r['context_information']['excluded_by_reason'] == {'incomplete_case': 2}
    assert r['context_information']['joint_branch_entropy_on_context_cohort_bits'] is not None


def test_stratum_insufficiency_does_not_generate_invalid_conditional_estimates():
    x = [0, 0, 1, 1]
    y = list(x)
    z = [0, 0, 1, 2]
    r = analyze(fixture(x, y, z), context={'node_id':'root', 'variable_id':'z'},
                min_stratum_samples=2)
    assert r['branch_information']['status'] == 'ok'
    assert r['context_information']['status'] == 'insufficient_conditioning_support'
    assert r['context_information']['under_supported_strata'] == 2
    assert r['context_information']['branch_context_mutual_information_bits'] is None


def test_unresolved_repeated_values_are_not_silently_dropped():
    source = fixture().to_dict()
    source['observations'][1]['values']['x'] = {'status':'multiple','value':None,
                                                 'occurrences':[{'status':'present','value':0},
                                                                {'status':'present','value':1}]}
    obj = StructuredObject.from_dict(source)
    r = analyze(obj)
    assert r['branch_information']['status'] == 'unresolved_repeated_values'
    assert r['branch_information']['joint_entropy_bits'] is None
    assert r['branch_information']['excluded_by_reason']['unresolved_repeated_values'] == 1
    with pytest.raises(StructuredContractError, match='scalar_only'):
        analyze(obj, occurrence_policy='scalar_only')
    assert analyze(obj, occurrence_policy='mean_numeric')['branch_information']['status'] == 'float_requires_explicit_binning'
    r2 = analyze(obj, occurrence_policy='mean_numeric', bins={'x': {'mode':'equal_width','bins':2}})
    assert r2['branch_information']['status'] == 'ok'
    assert r2['binding_plan']['variables'][0]['occurrence_policy'] == 'mean_numeric'


def test_float_requires_explicit_binning():
    vals = [.1, .1, .9, .9] * 4
    r = analyze(fixture(vals, vals, vals))
    assert r['branch_information']['status'] == 'float_requires_explicit_binning'
    assert r['branch_information']['joint_entropy_bits'] is None
    r = analyze(fixture(vals, vals, vals), bins={'x': {'mode':'equal_width','bins':2},
                                               'y': {'mode':'equal_width','bins':2}})
    assert r['branch_information']['status'] == 'ok'
    assert r['branch_information']['total_correlation_bits'] == pytest.approx(1)


def test_missing_observations_no_fabricated_entropy():
    obj = fixture().to_dict()
    for row in obj['observations']:
        row['values']['x'] = {'status':'explicit_null','value':None,'source_reason':'path_absent'}
    r = analyze(StructuredObject.from_dict(obj))
    assert r['aligned_rows'] == len(obj['observations'])
    assert r['branch_information']['status'] == 'insufficient_samples'
    assert r['branch_information']['joint_entropy_bits'] is None


def test_reject_dag_nonmember_duplicate_context_overlap_and_invalid_bins():
    obj = fixture()
    with pytest.raises(StructuredContractError, match='selected exact branch'):
        analyze_hierarchical_branch(obj, branch_root_id='a', mappings=[{'node_id':'b','variable_id':'x'}])
    with pytest.raises(StructuredContractError, match='distinct'):
        analyze_hierarchical_branch(obj, branch_root_id='a', mappings=[{'node_id':'a','variable_id':'x'},
                          {'node_id':'leaf','variable_id':'x'}])
    with pytest.raises(StructuredContractError, match='outside'):
        analyze(obj, context={'node_id':'leaf','variable_id':'z'})
    with pytest.raises(StructuredContractError, match='distinct'):
        analyze(obj, context={'node_id':'root','variable_id':'x'})
    with pytest.raises(StructuredContractError, match='unknown aligned observation'):
        analyze_hierarchical_branch(obj, branch_root_id='a', mappings=[{'node_id':'a','variable_id':'unknown'}])
    with pytest.raises(StructuredContractError, match='invalid discretization'):
        analyze(obj, bins={'x': {'mode':'equal_width','bins':1}})
    with pytest.raises(StructuredContractError, match='at least one branch'):
        analyze(obj, mappings=[])
    malformed = obj.to_dict()
    malformed['edges'].append({'id':'e4', 'kind':'source_link','source_id':'b','target_id':'leaf'})
    malformed['topology'] = {'kind':'dag'}
    with pytest.raises(StructuredContractError, match='tree/forest'):
        analyze(StructuredObject.from_dict(malformed))


def test_large_branch_retains_every_source_identity_not_just_visible_page():
    # A large hierarchy is a navigation concern, not a reason to clip the
    # full structural inventory or mislabel an estimate over two columns.
    count = 5102
    source = fixture().to_dict()
    source['nodes'] = [{'id': f'n{i}', 'kind':'atom', 'attributes':{}} for i in range(count)]
    source['edges'] = [{'id': f'e{i}', 'kind':'source_link', 'source_id':f'n{i}',
                        'target_id':f'n{i + 1}'} for i in range(count - 1)]
    original = StructuredObject.from_dict(source)
    report = analyze_hierarchical_branch(original, branch_root_id='n0', mappings=[
        {'node_id':'n0','variable_id':'x'}, {'node_id':f'n{count - 1}','variable_id':'y'},
    ])
    assert report['branch_node_count'] == count
    assert len(report['branch_node_ids']) == count
    assert report['unmapped_node_count'] == count - 2
    assert report['statistical_scope'] == 'explicit_mapped_branch_subset'
    assert report['branch_information']['status'] == 'ok'


def test_sparse_state_warning_without_silent_rejection():
    x = list(range(100))
    y = list(x)
    r = analyze(fixture(x, y, x))
    assert r['branch_information']['status'] == 'ok'
    assert r['branch_information']['distinct_joint_states'] == 100
    assert r['branch_information']['sparse_joint_states_relative_to_support'] is True
    assert r['branch_information']['joint_entropy_bits'] == pytest.approx(6.64385619)


def test_route_guards_and_source_pinning(tmp_path):
    flask = pytest.importorskip('flask')
    from ml_lab.ui import create_ui_blueprint
    app = flask.Flask(__name__)
    app.config['TESTING'] = True
    app.register_blueprint(create_ui_blueprint(name='hia06', experiment_project_root=str(tmp_path)), url_prefix='/lab')
    client = app.test_client()
    source = fixture()
    path = tmp_path / 'sample.json'
    path.write_text(source.to_json())
    base = '/lab/workbench/structured'
    assert client.post(base, data={'section':'data','action':'select','path':'sample.json',
                                   'kind':'structured'}).status_code == 200
    graph = client.get(base+'/graph').json
    endpoint = base+'/hierarchy/branch-information'
    body = {'source_revision': graph['view']['source_revision'], 'branch_root_id':'a',
            'mappings': mapped(), 'context':{'node_id':'root','variable_id':'z'},
            'occurrence_policy':'preserve', 'bins':{},'min_samples':2,'min_stratum_samples':2}
    headers = {'X-ML-Lab-HIA':'explicit-branch-information'}
    assert client.post(endpoint, json=body).status_code == 400
    assert client.post(endpoint, json={**body,'source_revision':'sha256:'+'0'*64},headers=headers).status_code == 409
    r = client.post(endpoint, json=body, headers=headers)
    assert r.status_code == 200, r.json
    assert r.json['report']['branch_information']['total_correlation_bits'] == pytest.approx(1)
    assert client.post(endpoint, json={**body,'extra':42}, headers=headers).status_code == 409
    path.write_text('{}')
    assert client.post(endpoint, json=body, headers=headers).status_code == 409
    assert not list(tmp_path.glob('*.db'))


def test_existing_graph_controls_preserved_and_branch_controller_loaded():
    template = (ROOT/'src/ml_lab/ui/templates/ml_lab_ui/workbench_structured.html').read_text()
    routes = (ROOT/'src/ml_lab/ui/workbench_structured_routes.py').read_text()
    workspace = (ROOT/'src/ml_lab/ui/static/ml_lab_hia_workspace.js').read_text()
    script = (ROOT/'src/ml_lab/ui/static/ml_lab_hia_branch.js').read_text()
    assert 'data-hia-branch-url' in template
    assert 'data-hia-branch-action="compute"' in template
    assert 'data-hia-branch-action="export"' in template
    assert 'data-hia-atomic-action="compute"' in template
    assert 'data-hia-neighbor-action="compute"' in template
    assert 'data-su10-sandbox' in template
    assert 'ml_lab_hia_branch.js' in template and 'ml_lab_hia_branch.js' in routes
    assert 'mlwb:hia-scope' in workspace and 'mlwb:hia-scope' in script
    assert "'X-ML-Lab-HIA':'explicit-branch-information'" in script
