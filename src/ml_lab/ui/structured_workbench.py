"""Bounded, domain-neutral bridge to the ML-UW-27..35 scientific APIs.

Source adapters interpret the originating format; the analysis engines only
receive validated StructuredObject instances. This module has no Flask, SQLite,
network, training or automatic artifact-writing dependencies.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from .structured_source_intake import project_source, delimited_to_structured

from ml_lab.structured import (
    StructuredObject, StructuredContractError, TransformationPlan, transform,
    describe_structure, describe_observations, describe_nested_variance,
    estimate_information, ProbabilisticModelSpec, fit_categorical_model,
    query_exact, ComparisonCandidate, ComparisonStudy, compare_representations,
)
from ml_lab.structured.adapters import cross_module_adapters

KINDS = ('structured', 'json', 'node_edge', 'xml_structure',
         'hierarchy_handoff', 'pypique_graph', 'pypique_graph_directed',
         'pypique_graph_undirected', 'pypique_backing', 'shared_graph')
LOCAL_FILE_KINDS = ('auto', 'xml_file', 'tabular') + KINDS
def _max_interactive_source_bytes() -> int | None:
    """Optional deployment guard, not a hardcoded graph-intake truncation.

    Zero (the default) means no artificial source-size rejection. Normal
    operating-system memory and the format's own safety checks still apply.
    Administrators may set a positive byte limit for constrained hosts.
    """
    configured = os.environ.get('ML_LAB_STRUCTURED_MAX_SOURCE_BYTES', '0')
    try:
        limit = int(configured)
    except (ValueError, TypeError) as exc:
        raise StructuredContractError('ML_LAB_STRUCTURED_MAX_SOURCE_BYTES must be a nonnegative integer') from exc
    if limit < 0:
        raise StructuredContractError('ML_LAB_STRUCTURED_MAX_SOURCE_BYTES must be nonnegative')
    return limit or None


def _strict_json(text):
    def pairs(items):
        record = {}
        for k, v in items:
            if k in record:
                raise StructuredContractError('duplicate JSON field: ' + k)
            record[k] = v
        return record
    def no_constant(value):
        raise StructuredContractError('nonfinite JSON number: ' + value)
    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=no_constant)
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise StructuredContractError('invalid JSON: ' + str(exc)) from exc


def read_source(root, path, kind, sha256=None):
    """Explicit project-relative artifact, no symlinks, no DB, SHA-pinned."""
    if kind not in LOCAL_FILE_KINDS:
        raise StructuredContractError('Choose a supported source adapter.')
    selected, relative = project_source(root, path)
    max_source_bytes = _max_interactive_source_bytes()
    if max_source_bytes is not None and selected.stat().st_size > max_source_bytes:
        raise StructuredContractError('Source exceeds configured ML_LAB_STRUCTURED_MAX_SOURCE_BYTES limit.')
    raw = selected.read_bytes()
    if max_source_bytes is not None and len(raw) > max_source_bytes:
        raise StructuredContractError('Source exceeds configured ML_LAB_STRUCTURED_MAX_SOURCE_BYTES limit.')
    fingerprint = hashlib.sha256(raw).hexdigest()
    if sha256 is not None and fingerprint != sha256:
        raise StructuredContractError('Source changed: select it again.')
    suffix = selected.suffix.lower()
    if kind == 'auto':
        if suffix == '.xml':
            adapter = 'xml_file'
        elif suffix in ('.csv', '.tsv'):
            adapter = 'tabular'
        else:
            try:
                parsed = _strict_json(raw.decode('utf-8-sig'))
            except UnicodeError as exc:
                raise StructuredContractError('Source must be UTF-8 JSON.') from exc
            schema = parsed.get('schema') if isinstance(parsed, dict) else None
            adapter = {
                'ml-lab.structured-object@1': 'structured',
                'ml-lab.xml-structure@2': 'xml_structure',
                'ml-lab.node-edge@1': 'node_edge',
                'pah-shared.graph@1': 'shared_graph',
                'pypique.concluded_graph': 'pypique_graph',
            }.get(schema, 'json')
            # Avoid hiding typed exports that omitted the optional node-edge schema.
            if adapter == 'json' and isinstance(parsed, dict) and {'nodes', 'edges', 'topology'} <= set(parsed):
                adapter = 'node_edge'
    else:
        adapter = kind
    if adapter == 'xml_file':
        if suffix != '.xml':
            raise StructuredContractError('Raw XML adapter requires an .xml file.')
        from ml_lab.data.xml_ingest import inspect_xml_structure, XmlSafetyError
        from ml_lab.structured.adapters import XMLStructureAdapter
        from xml.etree.ElementTree import ParseError
        try:
            profile = inspect_xml_structure(selected, source_sha256=fingerprint,
                                            max_source_bytes=max(1, len(raw)))
        except (ParseError, XmlSafetyError, LookupError, UnicodeError) as exc:
            raise StructuredContractError(f'Invalid XML structure: {exc}') from exc
        if hashlib.sha256(selected.read_bytes()).hexdigest() != fingerprint:
            raise StructuredContractError('Source changed during XML inspection; select again.')
        outcome = XMLStructureAdapter().adapt(profile)
        obj, warnings, losses = outcome.object, list(outcome.warnings), list(outcome.losses)
        warnings.append('Raw XML was safely inspected as a canonical path profile; no aligned observation rows were inferred.')
    elif adapter == 'tabular':
        if suffix not in ('.csv', '.tsv'):
            raise StructuredContractError('Tabular adapter requires a .csv or .tsv file.')
        obj, warnings, losses = delimited_to_structured(
            raw, relative_path=relative, sha256=fingerprint,
            delimiter='\t' if suffix == '.tsv' else ',')
    else:
        if suffix != '.json':
            raise StructuredContractError('This source adapter requires .json. Choose Auto for XML/CSV/TSV files.')
        try:
            text = raw.decode('utf-8-sig')
        except UnicodeError as exc:
            raise StructuredContractError('Source must be UTF-8 JSON.') from exc
        if adapter == 'structured':
            obj = StructuredObject.from_json(text)
            warnings, losses = [], []
        else:
            outcome = cross_module_adapters().adapt(adapter, _strict_json(text))
            obj, warnings, losses = outcome.object, list(outcome.warnings), list(outcome.losses)
    meta = {'path': relative, 'kind': kind, 'detected_adapter': adapter, 'sha256': fingerprint,
            'object_id': obj.object_id, 'topology': obj.topology['kind'],
            'nodes': len(obj.nodes), 'edges': len(obj.edges), 'rows': len(obj.observations),
            'variables': sorted(obj.observations[0]['values']) if obj.observations else [],
            'capabilities': sorted(obj.capabilities), 'warnings': warnings, 'losses': losses}
    return obj, meta


def _names(text):
    if not isinstance(text, str) or len(text) > 4096:
        raise StructuredContractError('invalid variable selection')
    parts = tuple(p.strip() for p in text.split(',') if p.strip())
    if len(parts) > 64 or len(parts) != len(set(parts)):
        raise StructuredContractError('variable names must be unique (max 64)')
    return parts


def _integer(text, low, high):
    try:
        val = int(text)
    except (ValueError, TypeError) as exc:
        raise StructuredContractError('invalid integer option') from exc
    if not low <= val <= high:
        raise StructuredContractError(f'integer must be in [{low}, {high}]')
    return val


def _number(text, low, high):
    try:
        val = float(text)
    except (ValueError, TypeError) as exc:
        raise StructuredContractError('invalid numeric option') from exc
    if not math.isfinite(val) or not low <= val <= high:
        raise StructuredContractError('nonfinite or out-of-range numeric option')
    return val


def independent_draft(source, variables):
    """Never invent source-to-probability edges or probability parameters."""
    if not 1 <= len(variables) <= 12:
        raise StructuredContractError('draft requires 1–12 observed categorical variables')
    source.require_aligned(variables)
    specs = []
    for name in variables:
        cells = [row['values'][name] for row in source.observations]
        if any(cell['status'] in ('present', 'empty') and type(cell['value']) is not str for cell in cells):
            raise StructuredContractError(name + ': categorical draft requires string values')
        states = sorted({cell['value'] for cell in cells if cell['status'] in ('present', 'empty')})
        if not 2 <= len(states) <= 64:
            raise StructuredContractError(name + ': need 2–64 observed categorical states')
        specs.append({'id': name, 'observed_variable': name, 'states': states})
    return {'schema': 'ml-lab.probabilistic-model@1', 'model_type': 'discrete_bayesian_network',
            'variables': specs, 'arcs': [],
            'conditionals': [{'variable_id': x['id'], 'family': 'categorical',
                              'status': 'unfitted', 'rows': []} for x in specs],
            'missing_policy': 'complete_case',
            'provenance': {'authored_as': 'unfitted_independent_UI_draft',
                           'warning': 'No dependencies or CPTs inferred from source graph.'}}


def analyze(source, section, action, form):
    """Return (scientific report, optional object/model JSON artifact)."""
    get = lambda name, default='': str(form.get(name, default))
    if section == 'data' and action == 'structure':
        return describe_structure(source), None
    if section == 'profiling':
        if action == 'structure':
            return describe_structure(source), None
        if action == 'observations':
            return describe_observations(source, variables=_names(get('variables')),
                                         numeric_variables=_names(get('numeric'))), None
        if action == 'variance':
            return describe_nested_variance(source, value_variable=get('value'),
                                            group_levels=_names(get('groups'))), None
        if action == 'information':
            bins = _strict_json(get('bins', '{}'))
            if not isinstance(bins, dict):
                raise StructuredContractError('bins must be a JSON mapping')
            return estimate_information(source, measure=get('measure', 'mutual_information'),
                x=get('x'), y=get('y') or None, given=_names(get('given')),
                bins=bins, missing_policy=get('missing', 'complete_case'),
                min_samples=_integer(get('min_samples', '2'), 2, 10000),
                min_stratum_samples=_integer(get('min_stratum', '2'), 1, 10000)), None
    if section == 'representation' and action == 'transform':
        mode = get('mode', 'full')
        plan = TransformationPlan(mode=mode,
            max_depth=_integer(get('depth', '1'), 0, 10000) if mode in ('depth', 'aggregate') else None,
            roots=_names(get('roots')) if mode == 'subtree' else (),
            variables=_names(get('projection')) or None)
        outcome = transform(source, plan)
        return {'schema': 'ml-lab.structured-ui-transform@1', 'audit': outcome.audit,
                'source': describe_structure(source),
                'result': describe_structure(outcome.object)}, outcome.object.to_dict()
    if section == 'comparison' and action == 'compare':
        target, predictors, retained = get('target'), _names(get('predictors')), _names(get('retained'))
        if not retained or not set(retained) <= set(predictors):
            raise StructuredContractError('retained predictors must be a nonempty subset')
        candidates = [ComparisonCandidate('full', TransformationPlan('full'), predictors)]
        if 'rooted_hierarchy' in source.capabilities:
            candidates.append(ComparisonCandidate('partial', TransformationPlan('depth',
                max_depth=_integer(get('depth', '1'), 0, 10000),
                variables=(target,) + retained), retained))
        candidates.append(ComparisonCandidate('flat', TransformationPlan('flat',
            variables=(target,) + retained), retained))
        study = ComparisonStudy(target=target, candidates=tuple(candidates),
            group_variable=get('group') or None,
            acknowledge_independent_rows=get('independent') == '1',
            seed=_integer(get('seed', '17'), 0, 2**32 - 1),
            holdout_fraction=_number(get('holdout', '0.25'), 0.05, 0.5),
            pseudocount=_number(get('pseudocount', '1'), 0.00001, 1000),
            min_train_rows=_integer(get('min_train', '5'), 1, 10000),
            min_test_rows=_integer(get('min_test', '2'), 1, 10000))
        return compare_representations(source, study), None
    if section == 'tools':
        if action == 'draft':
            payload = independent_draft(source, _names(get('variables')))
            return {'schema': 'ml-lab.structured-ui-draft@1', 'draft': payload}, payload
        if action in ('validate', 'fit', 'query', 'fit_query'):
            spec = get('model_json')
            if len(spec) > 120000:
                raise StructuredContractError('model JSON exceeds UI limit')
            model = ProbabilisticModelSpec.from_json(spec, source=source)
            if action == 'validate':
                return {'audit': model.audit, 'model_id': model.model_id}, model.to_dict()
            fit = None
            if action in ('fit', 'fit_query'):
                result = fit_categorical_model(model, source=source,
                    pseudocount=_number(get('pseudocount', '1'), 0, 1000),
                    min_complete_rows=_integer(get('min_complete', '2'), 1, 10000),
                    min_parent_rows=_integer(get('min_parent', '1'), 0, 10000),
                    replace_specified=get('replace') == '1')
                model, fit = result.model, result.report
            if action == 'fit':
                return {'fit': fit, 'model': model.to_dict()}, model.to_dict()
            evidence = _strict_json(get('evidence', '{}'))
            if not isinstance(evidence, dict) or any(type(k) is not str or type(v) is not str for k,v in evidence.items()):
                raise StructuredContractError('evidence must map string variable IDs to string states')
            query = query_exact(model, variables=_names(get('query')), evidence=evidence,
                                max_assignments=_integer(get('max_assignments', '16384'), 1, 131072))
            return {'fit': fit, 'query': query, 'model_id': model.model_id}, model.to_dict()
    raise StructuredContractError('unsupported structured operation')
