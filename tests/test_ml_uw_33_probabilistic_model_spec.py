"""ML-UW-33: validated, explicit probabilistic model *specifications*."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from ml_lab.structured import StructuredObject, StructuredContractError
from ml_lab.structured.probabilistic import (
    MODEL_SCHEMA, ProbabilisticModelSpec, source_reference,
)

FIXTURE = Path(__file__).parent / 'fixtures' / 'structured_uncertainty' / 'xml_tree.json'


def source(rows=None, topology='tree'):
    original = json.loads(FIXTURE.read_text())
    original.pop('object_id', None)
    original['topology'] = {'kind': 'tree'}
    if rows is None:
        rows = [('0', '0'), ('0', '1'), ('1', '1'), ('1', '0')]
    original['observations'] = []
    for i, (x, y) in enumerate(rows):
        def cell(value):
            if value is None:
                return {'status': 'missing', 'value': None, 'source_reason': 'test'}
            if value == '':
                return {'status': 'empty', 'value': ''}
            return {'status': 'present', 'value': value}
        original['observations'].append({
            'observation_id': f'obs-{i}',
            'values': {'x': cell(x), 'y': cell(y)},
        })
    if topology == 'directed_graph':
        original['topology'] = {'kind': 'directed_graph'}
        original['edges'].append({'id': 'reverse', 'kind': 'structural',
                                  'source_id': 'value', 'target_id': 'root'})
    return StructuredObject.from_dict(original)


def spec(*, fitted=True, mapping=True):
    variables = [
        {'id': 'X', 'observed_variable': 'x', 'states': ['0', '1'],
         'source_node_ids': ['root'] if mapping else []},
        {'id': 'Y', 'observed_variable': 'y', 'states': ['0', '1'],
         'source_node_ids': ['section'] if mapping else []},
    ]
    return {
        'schema': MODEL_SCHEMA,
        'model_type': 'discrete_bayesian_network',
        'variables': variables,
        'arcs': [{'parent': 'X', 'child': 'Y',
                  'source_edge_ids': ['e1'] if mapping else []}],
        'conditionals': [
            {'variable_id': 'X', 'family': 'categorical',
             'status': 'specified' if fitted else 'unfitted',
             'rows': [{'given': {}, 'probabilities': {'0': 0.4, '1': 0.6}}]
             if fitted else []},
            {'variable_id': 'Y', 'family': 'categorical',
             'status': 'specified' if fitted else 'unfitted',
             'rows': [
                 {'given': {'X': '0'}, 'probabilities': {'0': 0.3, '1': 0.7}},
                 {'given': {'X': '1'}, 'probabilities': {'0': 0.8, '1': 0.2}},
             ] if fitted else []},
        ],
        'missing_policy': 'complete_case',
        'provenance': {'authored_as': 'synthetic_fixture'},
    }


def validated(payload=None, *, data=None):
    return ProbabilisticModelSpec.from_dict(spec() if payload is None else payload,
                                            source=source() if data is None else data)


def test_explicit_model_dag_and_assumptions_are_not_source_topology():
    model = validated()
    assert model.audit['root_variables'] == ['X']
    assert model.audit['parent_variables'] == {'X': [], 'Y': ['X']}
    assert model.audit['parameter_status'] == 'specified'
    assert model.audit['assumptions']['source_graph_is_probability_model'] is False
    assert model.audit['assumptions']['source_edges_mapped_explicitly'] == 1
    assert model.audit['assumptions']['fitting_implemented'] is False
    assert model.audit['assumptions']['inference_implemented'] is False
    assert model.audit['support']['complete_case_rows'] == 4
    assert model.source_reference == source_reference(source())
    model.require_fully_specified()


def test_serialization_determinism_immutability_and_no_runtime_side_effects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = source()
    model = validated(data=data)
    serialized = model.to_json()
    assert ProbabilisticModelSpec.from_json(serialized, source=data).to_json() == serialized
    assert not list(tmp_path.iterdir())
    assert model.model_id.startswith('sha256:')
    assert model.model_id == validated(data=data).model_id
    record = model.to_dict()
    record['variables'][0]['states'][0] = 'tampered'
    assert model.to_dict()['variables'][0]['states'] == ['0', '1']
    assert json.loads(serialized)['model_id'] == model.model_id
    with pytest.raises(TypeError):
        model._payload['model_id'] = 'mutated'


def test_empty_model_arcs_do_not_auto_import_source_edges():
    payload = spec(mapping=False)
    payload['arcs'] = []
    payload['conditionals'][1]['rows'] = [
        {'given': {}, 'probabilities': {'0': 0.5, '1': 0.5}}]
    m = validated(payload)
    assert m.audit['root_variables'] == ['X', 'Y']
    assert m.to_dict()['arcs'] == []
    assert m.audit['assumptions']['source_edges_mapped_explicitly'] == 0


def test_source_may_be_cyclic_but_model_graph_remains_separate_and_acyclic():
    data = source(topology='directed_graph')
    assert 'acyclic' not in data.capabilities
    model = validated(data=data)
    assert model.audit['model_topological_order'] == ['X', 'Y']


def test_unfitted_contract_is_valid_without_fake_probabilities():
    m = validated(spec(fitted=False))
    assert m.audit['parameter_status'] == 'unfitted'
    assert all(not c['rows'] for c in m.to_dict()['conditionals'])
    with pytest.raises(StructuredContractError, match='specified'):
        m.require_fully_specified()


def test_partially_specified_conditionals_are_honestly_labeled():
    raw = spec()
    raw['conditionals'][0].update(status='unfitted', rows=[])
    m = validated(raw)
    assert m.audit['parameter_status'] == 'partial'
    with pytest.raises(StructuredContractError):
        m.require_fully_specified()


def test_missing_rows_reported_without_imputation():
    data = source([('0', '1'), (None, '1'), ('1', None), ('1', '0')])
    m = validated(data=data)
    assert m.audit['support']['rows'] == 4
    assert m.audit['support']['complete_case_rows'] == 2
    assert m.audit['support']['excluded_rows'] == 2
    assert m.audit['support']['per_variable']['X']['unavailable'] == 1
    assert m.audit['assumptions']['missingness_mechanism'] == 'not_inferred'
    raw = spec()
    raw['missing_policy'] = 'reject'
    with pytest.raises(StructuredContractError, match='missing or multiple'):
        validated(raw, data=data)


def test_numeric_observation_requires_explicit_discretization():
    data = source()
    raw_source = data.to_dict()
    raw_source.pop('object_id')
    raw_source['topology'] = {'kind': 'tree'}
    raw_source['observations'][0]['values']['x']['value'] = 0
    with pytest.raises(StructuredContractError, match='explicit binning required'):
        validated(data=StructuredObject.from_dict(raw_source))


def test_empty_value_must_be_explicitly_declared_in_domain():
    data = source([('', '0'), ('1', '1')])
    with pytest.raises(StructuredContractError, match='finite string domain'):
        validated(data=data)
    raw = spec()
    raw['variables'][0]['states'] = ['', '1']
    raw['conditionals'][0]['rows'][0]['probabilities'] = {'': 0.4, '1': 0.6}
    raw['conditionals'][1]['rows'][0]['given'] = {'X': ''}
    validated(raw, data=data)


def test_spec_requires_validated_aligned_source_observations():
    raw_source = source().to_dict()
    raw_source.pop('object_id')
    raw_source['topology'] = {'kind': 'tree'}
    raw_source['observations'] = []
    with pytest.raises(StructuredContractError, match='aligned source observations'):
        validated(data=StructuredObject.from_dict(raw_source))
    payload = spec()
    payload['variables'][0]['observed_variable'] = 'unknown_variable'
    with pytest.raises(StructuredContractError, match='aligned variables'):
        validated(payload)
    with pytest.raises(StructuredContractError, match='validated StructuredObject'):
        ProbabilisticModelSpec.from_dict(spec(), source={'nodes': []})


def test_cycle_in_model_is_rejected_even_when_source_is_cyclic():
    raw = spec(mapping=False)
    raw['arcs'].append({'parent': 'Y', 'child': 'X'})
    with pytest.raises(StructuredContractError, match='acyclic'):
        validated(raw, data=source(topology='directed_graph'))


@pytest.mark.parametrize('arc', [
    {'parent': 'X', 'child': 'X'},
    {'parent': 'X', 'child': 'missing'},
    {'parent': 'X', 'child': 'Y', 'source_edge_ids': ['unknown_edge']},
    {'parent': 'X', 'child': 'Y', 'source_edge_ids': ['e2']},
])
def test_invalid_arcs_fail_closed(arc):
    raw = spec()
    raw['arcs'] = [arc]
    with pytest.raises(StructuredContractError):
        validated(raw)


def test_duplicate_arc_or_models_with_duplicate_variable_id_rejected():
    raw = spec()
    raw['arcs'].append(deepcopy(raw['arcs'][0]))
    with pytest.raises(StructuredContractError, match='duplicate'):
        validated(raw)
    raw = spec()
    raw['variables'][1]['id'] = 'X'
    with pytest.raises(StructuredContractError, match='unique'):
        validated(raw)


@pytest.mark.parametrize('corruption', [
    lambda r: r['conditionals'][0].update(family='normal'),
    lambda r: r['conditionals'][0].update(status='magic'),
    lambda r: r['conditionals'][0]['rows'][0]['probabilities'].update({'0': 0.7}),
    lambda r: r['conditionals'][0]['rows'][0]['probabilities'].update({'0': -0.4, '1': 1.4}),
    lambda r: r['conditionals'][0]['rows'][0]['probabilities'].update({'0': float('nan')}),
    lambda r: r['conditionals'][0]['rows'][0]['probabilities'].update({'0': True}),
    lambda r: r['conditionals'][1]['rows'].pop(),
    lambda r: r['conditionals'][1]['rows'][0]['given'].update({'X': 'not_a_state'}),
    lambda r: r['conditionals'][1]['rows'][1]['given'].update({'X': '0'}),
    lambda r: r['conditionals'][1]['rows'][0]['given'].update({'extraneous': '1'}),
])
def test_invalid_cpt_specifications_rejected(corruption):
    raw = spec()
    corruption(raw)
    with pytest.raises(StructuredContractError):
        validated(raw)


def test_parent_table_combinatorics_guard():
    raw = spec(mapping=False, fitted=False)
    raw['variables'] = [
        {'id': f'V{i}', 'observed_variable': f'v{i}',
         'states': [f'{j}' for j in range(10)]}
        for i in range(6)
    ]
    raw['arcs'] = [{'parent': f'V{i}', 'child': 'V5'} for i in range(5)]
    raw['conditionals'] = [
        {'variable_id': f'V{i}', 'family': 'categorical',
         'status': 'unfitted', 'rows': []}
        for i in range(6)
    ]
    base = source().to_dict()
    base.pop('object_id')
    base['topology'] = {'kind': 'tree'}
    for row in base['observations']:
        row['values'] = {f'v{i}': {'status': 'present', 'value': '0'} for i in range(6)}
    with pytest.raises(StructuredContractError, match='limit'):
        validated(raw, data=StructuredObject.from_dict(base))


def test_source_fingerprint_must_match_not_only_user_controlled_object_id():
    data = source()
    model = validated(data=data)
    altered = data.to_dict()
    altered['observations'][0]['values']['x']['value'] = '1'
    other = StructuredObject.from_dict(altered)   # preserves object_id on purpose
    assert other.object_id == data.object_id
    with pytest.raises(StructuredContractError, match='source reference'):
        ProbabilisticModelSpec.from_json(model.to_json(), source=other)


def test_model_content_integrity_and_strict_json_parser():
    m = validated()
    tampered = m.to_dict()
    tampered['missing_policy'] = 'reject'
    with pytest.raises(StructuredContractError, match='model_id'):
        validated(tampered)
    with pytest.raises(StructuredContractError, match='duplicate JSON field'):
        ProbabilisticModelSpec.from_json('{"schema": 1, "schema": 2}', source=source())
    with pytest.raises(StructuredContractError):
        ProbabilisticModelSpec.from_json('NaN', source=source())


def test_input_order_of_cpt_rows_does_not_change_model_identity():
    raw = spec()
    raw['conditionals'][1]['rows'].reverse()
    assert validated(raw).model_id == validated().model_id


def test_no_latent_or_continuous_distribution_is_silently_accepted():
    raw = spec()
    raw['variables'][0].pop('observed_variable')
    with pytest.raises(StructuredContractError, match='missing fields'):
        validated(raw)
    raw = spec()
    raw['conditionals'][0]['family'] = 'gaussian'
    with pytest.raises(StructuredContractError, match='unsupported conditional family'):
        validated(raw)


def test_two_parent_dag_requires_complete_conditional_table():
    data = source().to_dict()
    data.pop('object_id')
    data['topology'] = {'kind': 'tree'}
    for i, row in enumerate(data['observations']):
        row['values']['z'] = {'status': 'present', 'value': '1' if i % 2 else '0'}
    obj = StructuredObject.from_dict(data)
    raw = spec(mapping=False, fitted=False)
    raw['variables'].append({'id': 'Z', 'observed_variable': 'z', 'states': ['0', '1']})
    raw['arcs'].append({'parent': 'X', 'child': 'Z'})
    raw['arcs'].append({'parent': 'Y', 'child': 'Z'})
    raw['conditionals'].append({'variable_id': 'Z', 'family': 'categorical',
                                'status': 'specified',
                                'rows': [
                                    {'given': {'X': x, 'Y': y},
                                     'probabilities': {'0': 0.25, '1': 0.75}}
                                    for x in ('1', '0') for y in ('1', '0')
                                ]})
    m = validated(raw, data=obj)
    assert m.audit['parent_variables']['Z'] == ['X', 'Y']
    assert [r['given'] for r in m.to_dict()['conditionals'][-1]['rows']] == [
        {'X': '0', 'Y': '0'}, {'X': '0', 'Y': '1'},
        {'X': '1', 'Y': '0'}, {'X': '1', 'Y': '1'},
    ]
    assert m.audit['parameter_status'] == 'partial'
    raw['conditionals'][-1]['rows'].pop()
    with pytest.raises(StructuredContractError, match='requires all 4'):
        validated(raw, data=obj)


def test_malformed_parent_assignment_is_rejected_without_type_errors():
    raw = spec()
    raw['conditionals'][1]['rows'][0]['given']['X'] = ['0']
    with pytest.raises(StructuredContractError, match='outside declared state domains'):
        validated(raw)
