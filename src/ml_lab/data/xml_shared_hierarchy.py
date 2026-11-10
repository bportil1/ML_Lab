"""P5-17: optional read-only XML structure -> PAH_Shared hierarchy projection.

The XML structural artifact, not this display projection, owns cardinality,
namespace identity, candidate selection, path navigation, and provenance.
No PAH_Shared Python/JS dependency is required to create the model.
"""
from __future__ import annotations

from typing import Any

from .types import XmlStructureArtifact

SCHEMA = "pah-shared.hierarchy@1"
MAX_NODES = 5000
MAX_DEPTH = 64


class XmlSharedHierarchyError(ValueError):
    """Structure cannot be represented without losing its identity."""


def project_xml_structure_hierarchy(artifact: XmlStructureArtifact) -> dict[str, Any]:
    """Build a read-only display model and separately retained XML provenance.

    No XML document is reopened; profiles are not modified. Namespaced canonical
    paths are JSON-pointer escaped by the authoritative XML inspector. This
    projection never unescapes or splits expanded QNames on their internal '/'.
    """
    if not isinstance(artifact, XmlStructureArtifact) or artifact.schema != "ml-lab.xml-structure@2":
        raise XmlSharedHierarchyError("Expected an XML structure v2 artifact")
    profiles = artifact.element_profiles
    if len(profiles) > MAX_NODES:
        raise XmlSharedHierarchyError("Hierarchy exceeds 5,000 nodes")
    keys = [p.canonical_path for p in profiles]
    if len(keys) != len(set(keys)):
        raise XmlSharedHierarchyError("Duplicate canonical XML path")
    known = set(keys)
    nodes: list[dict[str, Any]] = []
    paths: dict[str, dict[str, Any]] = {}
    for p in profiles:
        key = p.canonical_path
        if not isinstance(key, str) or not key.startswith("/") or len(key) > 96:
            raise XmlSharedHierarchyError("Canonical XML path is not a valid shared node ID")
        if not isinstance(p.depth, int) or p.depth < 1 or p.depth > MAX_DEPTH:
            raise XmlSharedHierarchyError("XML depth exceeds shared hierarchy limit")
        if not isinstance(p.path, str) or not p.path or len(p.path) > 240:
            raise XmlSharedHierarchyError("Display path is invalid or too long")
        if not isinstance(p.local_name, str) or not p.local_name or len(p.local_name) > 240:
            raise XmlSharedHierarchyError("XML element label is invalid or too long")
        parent = key.rpartition("/")[0] or None
        if parent is not None and parent not in known:
            raise XmlSharedHierarchyError("Parent XML path missing from structural profiles")
        if p.depth != key.count("/"):  # canonical paths escape namespace URI slashes
            raise XmlSharedHierarchyError("XML structural depth does not match canonical path")
        nodes.append({"id": key, "parent_id": parent, "label": p.local_name, "kind": "xml-element"})
        paths[key] = {
            "canonical_path": key, "display_path": p.path,
            "expanded_tag": p.tag, "namespace_uri": p.namespace_uri,
            "prefix": p.prefix, "occurrence_count": p.occurrence_count,
            "repeated": p.repeated, "optional": p.optional,
        }
    return {
        "model": {"schema": SCHEMA, "title": "XML structure", "nodes": nodes},
        "provenance": {
            "source_schema": artifact.schema,
            "source_sha256": artifact.source_sha256,
            "source_fingerprint": artifact.source_fingerprint,
            "nodes": paths,
        },
    }
