"""HIA-05: measured neighboring relationships from explicit SU-05 mappings."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml_lab.structured import (
    StructuredObject, StructuredContractError, analyze_neighbor_relationships,
    source_sha256,
)
from ml_lab.structured.neighbor_statistics import NEIGHBOR_SCHEMA

ROOT = Path(__file__).resolve().parents[1]


def cell(value):
    if value is None:
        return {'status': 'explicit_null', 'value': None, 'source_reason': 'path_absent'}
    return {'status': 'present', 'value': value}


def fixture(x=None, y=None, z=None, *, kind='tree'):
    x = x if x is not None else [0, 0, 1, 1] * 4
    y = y if y is not None else list(x)
    z = z if z is not None else list(x)
    return StructuredObject.from_dict({
        'schema': 'ml-lab.structured-object@1',
        'topology': {'kind': kind},
        'nodes': [{'id': key, 'kind': 'atom', 'attributes': {}}
                  for key in ['root', 'left', 'right', 'nested']],
        'edges': [{'id': key, 'kind': 'source_link', 'source_id': a, 'target_id': b}
                  for key, a, b in [('e1', 'root', 'left'),
                                    ('e2', 'root', 'right'),
                                    ('e3', 'left', 'nested')]],
        'observation_unit': 'record',
        'observations': [{'observation_id': f'o{i}', 'values': {
            'x': cell(a), 'y': cell(b), 'z': cell(c),
        }} for i, (a, b, c) in enumerate(zip(x, y, z))],
        'provenance': {},
    })


def analyze(source, **options):
    return analyze_neighbor_relationships(source, node_id='root', variable_id='x',
         neighbors=[{'node_id': 'left', 'variable_id': 'y'}], **options)


def test_perfect_association_conditional_information_and_context():
    source = fixture()
    before = source.to_json()
    report = analyze(source, conditioning=[{'node_id': 'right', 'variable_id': 'z'}])
    assert report['schema'] == NEIGHBOR_SCHEMA
    assert report['source_digest'] == source_sha256(source)
    assert report['binding_plan']['variables'][0]['node_ids'] == ['root']
    record = report['relationships'][0]
    assert record['structural_edges'] == [{'edge_id': 'e1', 'source_id': 'root', 'target_id': 'left', 'kind': 'source_link'}]
    assert record['mutual_information']['status'] == 'ok'
    assert record['mutual_information']['mi_bits'] == pytest.approx(1)
    assert record['mutual_information']['h_x_given_y_bits'] == pytest.approx(0)
    assert record['conditional_mutual_information']['status'] == 'ok'
    # I(X;Y|X)=0, although I(X;Y)=1: no independence inferred from graph.
    assert record['conditional_mutual_information']['mi_bits'] == pytest.approx(0)
    assert record['correlations']['pearson'] == pytest.approx(1)
    assert record['correlations']['spearman'] == pytest.approx(1)
    assert record['mutual_information']['support'] == len(source.observations)
    assert report['structural_context']['parent_ids'] == []
    assert record['structural_edges_are_not_statistical_dependence'] is True
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert source.to_json() == before


def test_independent_categorical_and_conditional_entropy():
    x = [0, 0, 1, 1] * 4
    y = [0, 1, 0, 1] * 4
    record = analyze(fixture(x, y, [1] * len(x)))['relationships'][0]
    assert record['mutual_information']['mi_bits'] == pytest.approx(0)
    assert record['mutual_information']['h_x_given_y_bits'] == pytest.approx(1)
    assert record['mutual_information']['h_y_given_x_bits'] == pytest.approx(1)
    assert record['correlations']['pearson'] == pytest.approx(0)
    assert record['correlations']['spearman'] == pytest.approx(0)


def test_pairwise_missingness_is_not_confused_with_cmi_support():
    x = [0, 0, 1, 1, 0, 0, 1, 1]
    y = list(x)
    z = [None, None, 1, 1, 0, 0, 1, 1]
    record = analyze(fixture(x, y, z), conditioning=[{'node_id': 'right', 'variable_id': 'z'}])['relationships'][0]
    assert record['mutual_information']['support'] == 8
    assert record['conditional_mutual_information']['support'] == 6
    assert record['conditional_mutual_information']['excluded_by_reason'] == {'incomplete_case': 2}
    assert record['correlations']['support'] == 8


def test_missing_support_and_constant_correlation_status():
    obj = fixture([1, None, 1, None], [2, 2, 2, 2], [5] * 4)
    record = analyze(obj)['relationships'][0]
    assert record['mutual_information']['support'] == 2
    assert record['mutual_information']['mi_bits'] == 0
    assert record['correlations']['status'] == 'constant_or_undefined_variance'
    assert record['correlations']['pearson'] is None
    assert analyze(obj, min_samples=3)['relationships'][0]['mutual_information']['status'] == 'insufficient_samples'


def test_float_entropy_requires_binning_but_numeric_correlations_remain():
    values = [.1, .2, .9, 1.0]
    unbinned = analyze(fixture(values, values, values))['relationships'][0]
    assert unbinned['mutual_information']['status'] == 'float_requires_explicit_binning'
    assert unbinned['mutual_information']['mi_bits'] is None
    assert unbinned['correlations']['pearson'] == pytest.approx(1)
    binned = analyze(fixture(values, values, values), bins={'x': {'mode':'equal_width','bins':2},
                                                           'y': {'mode':'equal_width','bins':2}})
    assert binned['relationships'][0]['mutual_information']['mi_bits'] == pytest.approx(1)


def test_unresolved_repeats_are_never_silently_averaged():
    rows = fixture([1, 2, 3], [1, 2, 3], [1, 2, 3]).to_dict()
    rows['observations'][1]['values']['x'] = {'status': 'multiple', 'value': None,
                                                'occurrences': [cell(1), cell(3)]}
    obj = StructuredObject.from_dict(rows)
    preserved = analyze(obj)['relationships'][0]
    assert preserved['mutual_information']['status'] == 'unresolved_repeated_values'
    assert preserved['correlations']['support'] == 2
    with pytest.raises(StructuredContractError, match='scalar_only'):
        analyze(obj, occurrence_policy='scalar_only')
    averaged = analyze(obj, occurrence_policy='mean_numeric', bins={'x': {'mode': 'equal_width', 'bins': 3}})
    assert averaged['relationships'][0]['mutual_information']['status'] == 'ok'
    assert averaged['binding_plan']['variables'][0]['occurrence_policy'] == 'mean_numeric'


def test_reject_nonadjacent_ambiguous_mapping_undeclared_conditioning_and_invalid_bins():
    obj = fixture()
    with pytest.raises(StructuredContractError, match='not directly adjacent'):
        analyze_neighbor_relationships(obj, node_id='root', variable_id='x',
             neighbors=[{'node_id':'nested','variable_id':'y'}])
    with pytest.raises(StructuredContractError, match='one-to-one'):
        analyze_neighbor_relationships(obj, node_id='root', variable_id='x',
             neighbors=[{'node_id':'left','variable_id':'x'}])
    with pytest.raises(StructuredContractError, match='one-to-one'):
        analyze(obj, conditioning=[{'node_id':'left','variable_id':'z'}])
    with pytest.raises(StructuredContractError, match='unknown aligned observation'):
        analyze_neighbor_relationships(obj, node_id='root', variable_id='x',
             neighbors=[{'node_id':'left','variable_id':'nonexistent'}])
    with pytest.raises(StructuredContractError, match='bin'):
        analyze(obj, bins={'x':{'mode':'quantile','bins':1}})
    with pytest.raises(StructuredContractError, match='conditioning'):
        analyze(obj, conditioning=[{'node_id':'right','variable_id':'z'}] * 9)


def test_route_request_guard_source_revision_and_read_only(tmp_path):
    flask = pytest.importorskip('flask')
    from ml_lab.ui import create_ui_blueprint
    app = flask.Flask(__name__)
    app.config['TESTING'] = True
    app.register_blueprint(create_ui_blueprint(name='hia05', experiment_project_root=str(tmp_path)), url_prefix='/lab')
    client = app.test_client()
    path = tmp_path / 'project.json'
    source = fixture()
    path.write_text(source.to_json())
    base = '/lab/workbench/structured'
    assert client.post(base, data={'section':'data','action':'select','path':'project.json',
                                   'kind':'structured'}).status_code == 200
    graph = client.get(base+'/graph').json
    endpoint = base+'/hierarchy/neighbors'
    payload = {'source_revision':graph['view']['source_revision'],'node_id':'root',
      'variable_id':'x','neighbors':[{'node_id':'left','variable_id':'y'}],
      'conditioning':[], 'bins':{},'occurrence_policy':'preserve','min_samples':2,
      'min_stratum_samples':2}
    headers = {'X-ML-Lab-HIA':'explicit-neighbor-statistics'}
    assert client.post(endpoint, json=payload).status_code == 400
    assert client.post(endpoint, json={**payload,'source_revision':'sha256:'+'0'*64}, headers=headers).status_code == 409
    r = client.post(endpoint, json=payload, headers=headers)
    assert r.status_code == 200, r.json
    assert r.json['report']['relationships'][0]['mutual_information']['mi_bits'] == pytest.approx(1)
    assert client.post(endpoint, json={**payload,'unexpected':42}, headers=headers).status_code == 409
    path.write_text('{}')
    assert client.post(endpoint, json=payload, headers=headers).status_code == 409
    assert not list(tmp_path.glob('*.db'))


def test_ui_preserves_all_controls_and_loads_atomic_and_neighbor_scripts():
    template = (ROOT/'src/ml_lab/ui/templates/ml_lab_ui/workbench_structured.html').read_text()
    routes = (ROOT/'src/ml_lab/ui/workbench_structured_routes.py').read_text()
    script = (ROOT/'src/ml_lab/ui/static/ml_lab_hia_neighbors.js').read_text()
    assert 'data-hia-neighbors-url' in template
    assert 'data-hia-neighbor-action="compute"' in template
    assert 'data-hia-atomic-action="compute"' in template
    assert 'data-hia-workspace' in template
    assert 'data-su10-sandbox' in template
    assert 'ml_lab_hia_atomic.js' in template
    assert 'ml_lab_hia_neighbors.js' in template
    assert 'ml_lab_hia_neighbors.js' in routes
    assert "'X-ML-Lab-HIA':'explicit-neighbor-statistics'" in script
    assert 'mlwb:graph-select' in script
