"""Generic graph artifact adapter: works for ASTs, DAGs, cyclic networks, etc.

Source owners must export node/edge artifacts through module APIs or files.
This adapter does not read other modules' SQLite databases or infer semantics.
"""
from __future__ import annotations

from collections.abc import Mapping

from ..contracts import CONTRACT_SCHEMA, StructuredContractError
from ..object import StructuredObject
from .registry import AdaptationResult

GRAPH_SCHEMA = 'ml-lab.node-edge@1'


class GraphAdapter:
    """Accept a validated StructuredObject or a typed node/edge mapping."""

    def adapt(self, source: object) -> AdaptationResult:
        if isinstance(source, StructuredObject):
            return AdaptationResult(source, CONTRACT_SCHEMA)
        if not isinstance(source, Mapping):
            raise StructuredContractError('graph source must be a mapping')
        schema = source.get('schema', GRAPH_SCHEMA)
        if schema == CONTRACT_SCHEMA:
            # This path keeps existing schema/capability claims under strict validation.
            return AdaptationResult(StructuredObject.from_dict(source), CONTRACT_SCHEMA)
        if schema != GRAPH_SCHEMA:
            raise StructuredContractError('unsupported generic graph source schema')
        allowed = {'schema', 'topology', 'nodes', 'edges', 'observations',
                   'observation_unit', 'provenance', 'source_id'}
        if set(source) - allowed:
            raise StructuredContractError('unrecognized graph source fields; refusing lossy import')
        if not {'topology', 'nodes', 'edges'}.issubset(source):
            raise StructuredContractError('graph import requires topology, nodes and edges')
        topology = source['topology']
        if isinstance(topology, str):
            topology = {'kind': topology}
        if not isinstance(topology, Mapping):
            raise StructuredContractError('graph topology must declare its kind')
        observations = source.get('observations', [])
        if observations and 'observation_unit' not in source:
            raise StructuredContractError('graph observations require an explicit observation unit')
        claimed_provenance = source.get('provenance', {})
        if not isinstance(claimed_provenance, Mapping):
            raise StructuredContractError('graph provenance must be a mapping')
        doc = {'schema': CONTRACT_SCHEMA, 'topology': topology,
               'nodes': source['nodes'], 'edges': source['edges'],
               'observation_unit': source.get('observation_unit', 'graph'),
               'observations': observations,
               'provenance': {'adapter': 'node-edge@1', 'source_schema': schema,
                              'source_provenance': claimed_provenance}}
        if 'source_id' in source:
            doc['provenance']['source_id'] = source['source_id']
        return AdaptationResult(StructuredObject.from_dict(doc), GRAPH_SCHEMA)
