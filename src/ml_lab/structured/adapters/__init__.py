"""Format-specific, pluggable StructuredObject adapters (ML-UW-29)."""
from .registry import AdaptationResult, AdapterRegistry, SourceAdapter
from .json_data import JSONAdapter
from .graph import GraphAdapter, GRAPH_SCHEMA
from .xml_structure import XMLStructureAdapter, XML_SCHEMA
from .ast_handoff import HierarchyHandoffAdapter, AST_HANDOFF_SCHEMA
from .cross_module import (
    PyPIQUEConcludedAdapter, SharedGraphProjectionAdapter,
    PYPIQUE_SCHEMA, SHARED_GRAPH_SCHEMA,
)


def default_adapters() -> AdapterRegistry:
    """Create a fresh registry; individual projects never share mutable state."""
    registry = AdapterRegistry()
    registry.register('json', JSONAdapter())
    registry.register('xml_structure', XMLStructureAdapter())
    registry.register('node_edge', GraphAdapter())
    return registry


def cross_module_adapters() -> AdapterRegistry:
    """Opt-in adapter registry for explicit intermodule *export* artifacts.

    Preserves the original ML-UW-29 default registry for existing clients.
    """
    registry = default_adapters()
    registry.register('hierarchy_handoff', HierarchyHandoffAdapter())
    registry.register('pypique_graph', PyPIQUEConcludedAdapter())
    registry.register('pypique_graph_directed', PyPIQUEConcludedAdapter(edge_filter='directed'))
    registry.register('pypique_graph_undirected', PyPIQUEConcludedAdapter(edge_filter='undirected'))
    registry.register('pypique_backing', PyPIQUEConcludedAdapter(graph_part='learned_probabilistic_backing'))
    registry.register('shared_graph', SharedGraphProjectionAdapter())
    return registry


__all__ = ['AdaptationResult', 'AdapterRegistry', 'SourceAdapter',
           'JSONAdapter', 'XMLStructureAdapter', 'GraphAdapter',
           'XML_SCHEMA', 'GRAPH_SCHEMA', 'default_adapters', 'cross_module_adapters',
           'HierarchyHandoffAdapter', 'AST_HANDOFF_SCHEMA', 'PyPIQUEConcludedAdapter',
           'SharedGraphProjectionAdapter', 'PYPIQUE_SCHEMA', 'SHARED_GRAPH_SCHEMA']
