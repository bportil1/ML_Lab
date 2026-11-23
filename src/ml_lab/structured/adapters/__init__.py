"""Format-specific, pluggable StructuredObject adapters (ML-UW-29)."""
from .registry import AdaptationResult, AdapterRegistry, SourceAdapter
from .json_data import JSONAdapter
from .graph import GraphAdapter, GRAPH_SCHEMA
from .xml_structure import XMLStructureAdapter, XML_SCHEMA


def default_adapters() -> AdapterRegistry:
    """Create a fresh registry; individual projects never share mutable state."""
    registry = AdapterRegistry()
    registry.register('json', JSONAdapter())
    registry.register('xml_structure', XMLStructureAdapter())
    registry.register('node_edge', GraphAdapter())
    return registry


__all__ = ['AdaptationResult', 'AdapterRegistry', 'SourceAdapter',
           'JSONAdapter', 'XMLStructureAdapter', 'GraphAdapter',
           'XML_SCHEMA', 'GRAPH_SCHEMA', 'default_adapters']
