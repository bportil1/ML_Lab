"""ML-UW-34: categorical BN fitting and bounded exact inference."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from ml_lab.structured import (
    StructuredObject, StructuredContractError, ProbabilisticModelSpec,
    fit_categorical_model, query_exact,
)
from ml_lab.structured.probabilistic_inference import FIT_SCHEMA, QUERY_SCHEMA

FIXTURE = Path(__file__).parent / 'fixtures' / 'structured_uncertainty' / 'xml_tree.json'


def data(rows, *, names=('x', 'y')):
    base = json.loads(FIXTURE.read_text())
    base.pop('object_id', None)
    base['observations'] = []
    for i, values in enumerate(rows):
        fields = {}
        for name, value in zip(names, values):
            if isinstance(value, dict):
                cell = value
            elif value is None:
                cell = {'status': 'missing', 'value': None, 'source_reason': 'test_missing'}
            elif value == '':
                cell = {'status': 'empty', 'value': ''}
            else:
                cell = {'status': 'present', 'value': value}
            fields[name] = cell
        base['observations'].append({'observation_id': f'obs-{i}', 'values': fields})
    return StructuredObject.from_dict(base)


def model(source, *, specified=False, separate=False):
    raw = {
        'schema': 'ml-lab.probabilistic-model@1',
        'model_type': 'discrete_bayesian_network',
        'variables': [
            {'id': 'X', 'observed_variable': 'x', 'states': ['0', '1']},
            {'id': 'Y', 'observed_variable': 'y', 'states': ['0', '1']},
        ],
        'arcs': [] if separate else [{'parent': 'X', 'child': 'Y'}],
        'conditionals': [
            {'variable_id': 'X', 'family': 'categorical',
             'status': 'specified' if specified else 'unfitted',
             'rows': [{'given': {}, 'probabilities': {'0': 0.4, '1': 0.6}}]
             if specified else []},
            {'variable_id': 'Y', 'family': 'categorical',
             'status': 'specified' if specified else 'unfitted',
             'rows': [
                 {'given': {'X': '0'}, 'probabilities': {'0': 0.8, '1': 0.2}},
                 {'given': {'X': '1'}, 'probabilities': {'0': 0.2, '1': 0.8}},
             ] if specified and not separate else (
                 [{'given': {}, 'probabilities': {'0': 0.5, '1': 0.5}}]
                 if specified else [])},
        ],
        'missing_policy': 'complete_case', 'provenance': {'test': 'synthetic'},
    }
    return ProbabilisticModelSpec.from_dict(raw, source=source)


def posterior(report):
    return {tuple(row['given'].values()): row['probability'] for row in report['posterior']}


def test_empirical_fit_counts_and_full_cpt():
    obj = data([('0', '0'), ('0', '0'), ('1', '1'), ('1', '1'),
                ('1', '0'), ('1', '0')])
    initial = model(obj)
    result = fit_categorical_model(initial, source=obj)
    assert result.report['schema'] == FIT_SCHEMA
    assert result.report['support']['complete_case_rows'] == 6
    assert result.model.audit['parameter_status'] == 'specified'
    assert initial.audit['parameter_status'] == 'unfitted'
    assert result.report['input_model_id'] == initial.model_id
    assert result.report['fitted_model_id'] == result.model.model_id
    cond = result.model.to_dict()['conditionals']
    assert cond[0]['rows'][0]['probabilities']['1'] == pytest.approx(4 / 6)
    assert cond[1]['rows'][0]['probabilities']['1'] == 0.0
    assert cond[1]['rows'][1]['probabilities']['1'] == 0.5
    assert result.report['conditionals'][1]['rows'][0]['state_counts'] == {'0': 2, '1': 0}
    assert result.report['conditionals'][1]['rows'][1]['observed_rows'] == 4
    assert result.report['diagnostics']['observation_independence'] == 'not_verified'
    assert result.report['training_diagnostics']['complete_case_log_likelihood'] is not None
    assert result.report['training_diagnostics']['zero_likelihood_rows'] == 0
    assert result.report['diagnostics']['source_edges_used_as_model_arcs'] is False


def test_smoothed_fit_is_explicit_and_reproducible():
    obj = data([('0', '0'), ('0', '0'), ('1', '1'), ('1', '1')])
    original = model(obj)
    first = fit_categorical_model(original, source=obj, pseudocount=1)
    second = fit_categorical_model(original, source=obj, pseudocount=1)
    assert first.model.model_id == second.model.model_id
    assert first.report == second.report
    probs = first.model.to_dict()['conditionals'][1]['rows'][0]['probabilities']
    assert probs == pytest.approx({'0': 0.75, '1': 0.25})
    assert first.report['configuration']['method'] == 'symmetric_dirichlet_posterior_mean'
    assert first.report['configuration']['pseudocount_per_state'] == 1.0
    assert all(p >= 0 for c in first.model.to_dict()['conditionals'] for r in c['rows']
               for p in r['probabilities'].values())


def test_sparse_parent_assignments_default_fails_explicit_prior_only_opt_in():
    obj = data([('0', '0'), ('0', '1')])
    initial = model(obj)
    with pytest.raises(StructuredContractError, match='parent assignment'):
        fit_categorical_model(initial, source=obj, pseudocount=1)
    with pytest.raises(StructuredContractError, match='requires positive pseudocount'):
        fit_categorical_model(initial, source=obj, min_parent_rows=0)
    fit = fit_categorical_model(initial, source=obj, pseudocount=1,
                                min_parent_rows=0)
    assert fit.report['diagnostics']['prior_only_parent_rows'] == 1
    assert fit.model.to_dict()['conditionals'][1]['rows'][1]['probabilities'] == {
        '0': 0.5, '1': 0.5,
    }
    assert fit.report['conditionals'][1]['rows'][1]['prior_only'] is True


def test_partial_specification_is_preserved_unless_explicitly_refitted():
    obj = data([('0', '0'), ('1', '1'), ('1', '1')])
    raw = model(obj, specified=True).to_dict()
    raw.pop('model_id')
    raw['conditionals'][1] = {
        'variable_id': 'Y', 'family': 'categorical', 'status': 'unfitted', 'rows': []}
    partial = ProbabilisticModelSpec.from_dict(raw, source=obj)
    original_root = partial.to_dict()['conditionals'][0]
    updated = fit_categorical_model(partial, source=obj, pseudocount=1)
    assert updated.model.to_dict()['conditionals'][0] == original_root
    assert updated.report['conditionals'][0]['action'] == 'preserved'
    with pytest.raises(StructuredContractError, match='no unfitted'):
        fit_categorical_model(model(obj, specified=True), source=obj)
    refitted = fit_categorical_model(model(obj, specified=True), source=obj,
                                     replace_specified=True)
    assert refitted.report['conditionals'][0]['action'] == 'refitted'
    assert refitted.model.to_dict()['conditionals'][0] != original_root


def test_missing_rows_excluded_not_imputed_and_not_counted_twice():
    obj = data([('0', '0'), (None, '0'), ('1', None), ('1', '1'),
                ('0', {'status': 'explicit_null', 'value': None,
                       'source_reason': 'explicit_xsi_nil'})])
    fit = fit_categorical_model(model(obj), source=obj)
    assert fit.report['support']['rows'] == 5
    assert fit.report['support']['complete_case_rows'] == 2
    assert fit.report['support']['excluded_rows'] == 3
    assert fit.report['support']['per_variable']['Y']['unavailable'] == 2
    assert fit.report['diagnostics']['excluded_rows_are_not_imputed']
    assert sum(row['observed_rows'] for row in fit.report['conditionals'][1]['rows']) == 2
    with pytest.raises(StructuredContractError, match='insufficient complete-case'):
        fit_categorical_model(model(obj), source=obj, min_complete_rows=3)


def test_repeated_multiple_cell_is_excluded():
    multiple = {'status': 'multiple', 'value': None, 'occurrences': [
        {'status': 'present', 'value': '0'}, {'status': 'present', 'value': '1'}]}
    obj = data([('0', '0'), ('1', '1'), (multiple, '0')])
    fit = fit_categorical_model(model(obj), source=obj)
    assert fit.report['support']['excluded_rows'] == 1


def test_fit_rejects_wrong_or_tampered_source():
    obj = data([('0', '0'), ('1', '1')])
    original = model(obj)
    raw = obj.to_dict()
    raw['observations'][0]['values']['x']['value'] = '1'
    changed = StructuredObject.from_dict(raw)
    assert changed.object_id == obj.object_id
    with pytest.raises(StructuredContractError, match='source reference'):
        fit_categorical_model(original, source=changed)
    with pytest.raises(StructuredContractError, match='source reference'):
        fit_categorical_model(original, source={'fake': True})


@pytest.mark.parametrize('kwargs', [
    {'pseudocount': True}, {'pseudocount': -1}, {'pseudocount': math.nan},
    {'pseudocount': float('inf')}, {'pseudocount': 1_000_001},
    {'min_complete_rows': 0}, {'min_complete_rows': True},
    {'min_parent_rows': -1}, {'min_parent_rows': 1.5},
    {'replace_specified': 1},
])
def test_invalid_fit_configuration_rejected(kwargs):
    obj = data([('0', '0'), ('1', '1')])
    with pytest.raises(StructuredContractError):
        fit_categorical_model(model(obj), source=obj, **kwargs)


def test_fitting_is_immutable_pure_and_returns_detached_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    obj = data([('0', '0'), ('1', '1')])
    original = model(obj)
    before = original.to_json()
    fit = fit_categorical_model(original, source=obj, pseudocount=2)
    assert original.to_json() == before
    assert not list(tmp_path.iterdir())
    detached = fit.report
    detached['support']['complete_case_rows'] = 999
    assert fit.report['support']['complete_case_rows'] == 2
    with pytest.raises(TypeError):
        fit._report['fit_id'] = 'not-real'
    assert json.dumps(fit.to_dict(), allow_nan=False)


def test_known_bayes_posterior_and_evidence_probability():
    obj = data([('0', '0'), ('1', '1')])
    m = model(obj, specified=True)
    r = query_exact(m, variables=['X'], evidence={'Y': '1'})
    assert r['schema'] == QUERY_SCHEMA
    assert r['status'] == 'ok'
    assert r['probability_evidence'] == pytest.approx(.56)
    assert r['log_probability_evidence'] == pytest.approx(math.log(.56))
    assert posterior(r)[('1',)] == pytest.approx(6 / 7)
    assert posterior(r)[('0',)] == pytest.approx(1 / 7)
    assert r['audit']['enumerated_assignments'] == 2
    assert r['audit']['causal_interpretation'] is False


def test_unconditional_joint_and_fitted_model_queries():
    obj = data([('0', '0'), ('0', '1'), ('1', '1'), ('1', '1')])
    fitted = fit_categorical_model(model(obj), source=obj).model
    r = query_exact(fitted, variables=['X', 'Y'])
    assert r['status'] == 'ok'
    assert r['probability_evidence'] == pytest.approx(1.0)
    assert r['audit']['enumerated_assignments'] == 4
    assert len(r['posterior']) == 4
    assert math.fsum(posterior(r).values()) == pytest.approx(1.0)
    assert posterior(r)[('0', '0')] == pytest.approx(.25)
    assert posterior(r)[('0', '1')] == pytest.approx(.25)
    assert posterior(r)[('1', '0')] == pytest.approx(0)
    assert posterior(r)[('1', '1')] == pytest.approx(.5)


def test_impossible_evidence_is_unavailable_not_zero_posterior():
    obj = data([('0', '0'), ('1', '1')])
    raw = model(obj, specified=True).to_dict()
    raw.pop('model_id')
    raw['conditionals'][0]['rows'][0]['probabilities'] = {'0': 1.0, '1': 0.0}
    m = ProbabilisticModelSpec.from_dict(raw, source=obj)
    result = query_exact(m, variables=['Y'], evidence={'X': '1'})
    assert result['status'] == 'unavailable'
    assert result['reason'] == 'zero_probability_evidence'
    assert result['posterior'] is None
    assert result['log_probability_evidence'] is None


def many_roots(count, *, rare_probability=None):
    names = tuple(f'v{i}' for i in range(count))
    obj = data([tuple('0' for _ in names)], names=names)
    payload = {
        'schema': 'ml-lab.probabilistic-model@1',
        'model_type': 'discrete_bayesian_network',
        'variables': [{'id': f'V{i}', 'observed_variable': name,
                       'states': ['0', '1']} for i, name in enumerate(names)],
        'arcs': [],
        'conditionals': [{
            'variable_id': f'V{i}', 'family': 'categorical', 'status': 'specified',
            'rows': [{'given': {}, 'probabilities': {
                '0': rare_probability if rare_probability is not None else 0.5,
                '1': 1 - rare_probability if rare_probability is not None else 0.5,
            }}],
        } for i in range(count)],
        'missing_policy': 'complete_case',
    }
    return ProbabilisticModelSpec.from_dict(payload, source=obj)


def test_rare_evidence_is_not_mislabeled_impossible_when_float_underflows():
    m = many_roots(20, rare_probability=1e-30)
    r = query_exact(m, variables=['V19'], evidence={f'V{i}': '0' for i in range(12)})
    assert r['status'] == 'ok'
    assert r['probability_evidence'] == 0.0
    assert r['audit']['probability_evidence_underflow']
    assert r['log_probability_evidence'] == pytest.approx(12 * math.log(1e-30))
    assert posterior(r)[('0',)] == pytest.approx(1e-30, rel=1e-10)


def test_query_complexity_is_bounded_and_evidence_pruning_works():
    m = many_roots(18)
    with pytest.raises(StructuredContractError, match='would enumerate'):
        query_exact(m, variables=['V0'])
    r = query_exact(m, variables=['V0'], evidence={f'V{i}': '0' for i in range(2, 18)})
    assert r['audit']['enumerated_assignments'] == 4
    with pytest.raises(StructuredContractError, match='would enumerate'):
        query_exact(m, variables=['V0'], evidence={f'V{i}': '0' for i in range(2, 18)},
                    max_assignments=3)
    with pytest.raises(StructuredContractError, match='max_assignments'):
        query_exact(m, variables=['V0'], max_assignments=1_000_000)


def test_bounded_query_outcomes():
    m = many_roots(13)
    with pytest.raises(StructuredContractError, match='outcomes'):
        query_exact(m, variables=[f'V{i}' for i in range(13)])


@pytest.mark.parametrize('kwargs', [
    {'variables': []}, {'variables': 'X'},
    {'variables': ['X', 'X']}, {'variables': ['X', 'missing']},
    {'variables': ['X'], 'evidence': {'X': '0'}},
    {'variables': ['X'], 'evidence': {'Y': '9'}},
    {'variables': ['X'], 'evidence': {'missing': '0'}},
    {'variables': ['X'], 'evidence': {'Y': 0}},
    {'variables': ['X'], 'max_assignments': True},
])
def test_invalid_queries_fail_closed(kwargs):
    obj = data([('0', '0'), ('1', '1')])
    with pytest.raises(StructuredContractError):
        query_exact(model(obj, specified=True), **kwargs)


def test_unfitted_and_partial_models_cannot_be_queried():
    obj = data([('0', '0'), ('1', '1')])
    with pytest.raises(StructuredContractError, match='specified'):
        query_exact(model(obj), variables=['X'])
    m = model(obj, specified=True)
    raw = m.to_dict()
    raw.pop('model_id')
    raw['conditionals'][1] = {'variable_id': 'Y', 'family': 'categorical',
                               'status': 'unfitted', 'rows': []}
    partial = ProbabilisticModelSpec.from_dict(raw, source=obj)
    with pytest.raises(StructuredContractError, match='specified'):
        query_exact(partial, variables=['X'])


def test_independent_sources_and_model_graphs_do_not_coerce_source_edges():
    obj = data([('0', '0'), ('1', '1')])
    m = model(obj, specified=True, separate=True)
    assert m.to_dict()['arcs'] == []
    r = query_exact(m, variables=['X'], evidence={'Y': '1'})
    assert posterior(r)[('1',)] == pytest.approx(.6)


def test_query_is_pure_no_io_and_json_safe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    obj = data([('0', '0'), ('1', '1')])
    m = model(obj, specified=True)
    before = m.to_json()
    q = query_exact(m, variables=['Y'], evidence={'X': '0'})
    assert m.to_json() == before
    assert json.dumps(q, sort_keys=True, allow_nan=False)
    assert not list(tmp_path.iterdir())


def test_three_node_collider_explaining_away():
    obj = data([('0', '0', '0'), ('1', '1', '1')], names=('x', 'y', 'z'))
    raw = {
        'schema': 'ml-lab.probabilistic-model@1',
        'model_type': 'discrete_bayesian_network',
        'variables': [{'id': name, 'observed_variable': name.lower(),
                       'states': ['0', '1']} for name in ('X', 'Y', 'Z')],
        'arcs': [{'parent': 'X', 'child': 'Z'}, {'parent': 'Y', 'child': 'Z'}],
        'conditionals': [
            {'variable_id': 'X', 'family': 'categorical', 'status': 'specified',
             'rows': [{'given': {}, 'probabilities': {'0': .5, '1': .5}}]},
            {'variable_id': 'Y', 'family': 'categorical', 'status': 'specified',
             'rows': [{'given': {}, 'probabilities': {'0': .5, '1': .5}}]},
            {'variable_id': 'Z', 'family': 'categorical', 'status': 'specified',
             'rows': [{'given': {'X': x, 'Y': y},
                       'probabilities': {'0': 1.0 if x == '0' and y == '0' else 0.0,
                                         '1': 0.0 if x == '0' and y == '0' else 1.0}}
                      for x in ['0', '1'] for y in ['0', '1']]},
        ],
        'missing_policy': 'complete_case',
    }
    m = ProbabilisticModelSpec.from_dict(raw, source=obj)
    prior = query_exact(m, variables=['X'])
    conditioned = query_exact(m, variables=['X'], evidence={'Z': '1', 'Y': '1'})
    assert posterior(prior)[('1',)] == pytest.approx(.5)
    assert posterior(conditioned)[('1',)] == pytest.approx(.5)
    conditioned2 = query_exact(m, variables=['X'], evidence={'Z': '1', 'Y': '0'})
    assert posterior(conditioned2)[('1',)] == pytest.approx(1.0)


def test_observed_impossible_rows_under_partial_preserved_model_are_flagged():
    obj = data([('0', '0'), ('1', '1')])
    raw = model(obj, specified=True).to_dict()
    raw.pop('model_id')
    raw['conditionals'][0]['rows'][0]['probabilities'] = {'0': 1.0, '1': 0.0}
    raw['conditionals'][1] = {'variable_id': 'Y', 'family': 'categorical',
                                'status': 'unfitted', 'rows': []}
    partially_specified = ProbabilisticModelSpec.from_dict(raw, source=obj)
    fit = fit_categorical_model(partially_specified, source=obj)
    diag = fit.report['training_diagnostics']
    assert diag['zero_likelihood_rows'] == 1
    assert diag['complete_case_log_likelihood'] is None
    assert diag['evaluation_scope'] == 'in_sample_only_not_holdout_validation'


def test_subnormal_evidence_probability_is_reported_when_representable():
    m = many_roots(12, rare_probability=1e-28)
    r = query_exact(m, variables=['V11'], evidence={f'V{i}': '0' for i in range(11)})
    assert r['status'] == 'ok'
    assert r['probability_evidence'] > 0.0
    assert r['probability_evidence'] == pytest.approx(1e-308, rel=1e-12, abs=0.0)
    assert r['audit']['probability_evidence_underflow'] is False
