"""P5-15 opt-in, lossless XML *representation*, not an extraction/encoding pipeline.

Inputs are an already-inspected structure artifact or an Element supplied by the
existing trusted parser. This module never opens a source, executes a recipe, or
writes a dataset. All values remain strings; inference belongs to a separate stage.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import xml.etree.ElementTree as ET

from .types import XmlStructureArtifact

SCHEMA = "ml-lab.xml-typed-variables@1"
RECIPE_SCHEMA = "ml-lab.xml-typed-recipe@1"
KINDS = frozenset({"scalar", "object", "list", "content_tokens", "path_tokens", "linked_child_records"})


class XmlTypedRepresentationError(ValueError):
    """Invalid typed representation request, without changing legacy XML operations."""


def _escape(segment: str) -> str:
    return segment.replace("~", "~0").replace("/", "~1")


def _name(tag: str) -> dict[str, str | None]:
    if tag.startswith("{") and "}" in tag:
        uri, local = tag[1:].split("}", 1)
        return {"expanded_name": tag, "namespace_uri": uri, "local_name": local}
    return {"expanded_name": tag, "namespace_uri": None, "local_name": tag}


def describe_xml_variables(artifact: XmlStructureArtifact) -> dict[str, Any]:
    """Describe available representations from a structural artifact alone.

    Descriptors are capabilities, not assertions that individual record values
    were read. A profile with repeated children offers list representation.
    """
    profiles = list(artifact.element_profiles)
    descriptors = []
    for item in profiles:
        capabilities = ["object", "path_tokens"]
        if item.text_occurrence_count or item.attributes:
            capabilities.append("scalar")
        if item.repeated:
            capabilities.append("list")
        if item.mixed_content:
            capabilities.append("content_tokens")
        descriptors.append({
            "canonical_path": item.canonical_path,
            "display_path": item.path,
            "namespace_uri": item.namespace_uri,
            "local_name": item.local_name,
            "representations": capabilities,
            "cardinality": {
                "optional": item.optional,
                "repeated": item.repeated,
                "min_per_parent": item.min_per_parent,
                "max_per_parent": item.max_per_parent,
            },
            "has_mixed_content": item.mixed_content,
        })
    return {"schema": SCHEMA, "source_schema": "ml-lab.xml-structure@2",
            "source_sha256": artifact.source_sha256, "variables": descriptors,
            "value_scope": "structure_only", "encoding": "none"}


def build_typed_recipe(*, record_root_canonical_path: str,
                       variables: Mapping[str, str],
                       linked_child_paths: tuple[str, ...] = ()) -> dict[str, Any]:
    """Declare an explicit opt-in typed recipe; leave existing recipes untouched."""
    if not isinstance(record_root_canonical_path, str) or not record_root_canonical_path.startswith("/"):
        raise XmlTypedRepresentationError("record_root_canonical_path must be absolute")
    if not isinstance(variables, Mapping) or not variables:
        raise XmlTypedRepresentationError("select at least one typed variable")
    for path, kind in variables.items():
        if not isinstance(path, str) or not path.startswith("/") or kind not in KINDS:
            raise XmlTypedRepresentationError("invalid typed variable path or representation")
        if path != record_root_canonical_path and not path.startswith(record_root_canonical_path + "/"):
            raise XmlTypedRepresentationError("variable lies outside record root")
    for path in linked_child_paths:
        if not isinstance(path, str) or not path.startswith(record_root_canonical_path + "/"):
            raise XmlTypedRepresentationError("linked child path lies outside record root")
    return {"schema": RECIPE_SCHEMA, "version": 1,
            "record_root_canonical_path": record_root_canonical_path,
            "variables": [{"canonical_path": path, "representation": kind} for path, kind in variables.items()],
            "linked_child_paths": list(linked_child_paths),
            "encoding": "none", "legacy_extraction_modified": False}


def represent_xml_element(element: ET.Element, *, canonical_path: str | None = None,
                          max_depth: int = 128, max_nodes: int = 100_000) -> dict[str, Any]:
    """Describe an already-parsed element preserving order, mixed text and lineage.

    ElementTree has already expanded namespace names. Each child receives an
    occurrence index (among same-tag siblings) and an absolute path. Missing
    and empty text are distinct; no value is numerically converted. Both the
    structural children and interleaved content tokens are retained.
    """
    if not isinstance(element, ET.Element):
        raise XmlTypedRepresentationError("expected a parsed XML Element")
    if max_depth < 0 or max_nodes < 1:
        raise XmlTypedRepresentationError("invalid representation limits")
    root_path = canonical_path or "/" + _escape(str(element.tag))
    if not root_path.startswith("/"):
        raise XmlTypedRepresentationError("canonical_path must be absolute")
    count = 0

    def visit(node: ET.Element, path: str, position: int, depth: int) -> dict[str, Any]:
        nonlocal count
        count += 1
        if count > max_nodes or depth > max_depth:
            raise XmlTypedRepresentationError("XML representation exceeds explicit limits")
        children = list(node)
        counts: dict[str, int] = {}
        rendered = []
        for child in children:
            tag = str(child.tag)
            index = counts.get(tag, 0)
            counts[tag] = index + 1
            rendered.append(visit(child, path + "/" + _escape(tag), index, depth + 1))
        tokens: list[dict[str, Any]] = [{"kind": "text", "value": node.text, "present": node.text is not None}]
        for child, value in zip(rendered, children):
            tokens.append({"kind": "child", "canonical_path": child["canonical_path"],
                           "occurrence_index": child["occurrence_index"]})
            tokens.append({"kind": "tail", "value": value.tail, "present": value.tail is not None})
        return {"kind": "object", "name": _name(str(node.tag)),
                "canonical_path": path, "occurrence_index": position,
                "text": {"kind": "scalar", "value": node.text, "present": node.text is not None},
                "attributes": [{"name": _name(str(key)), "value": value, "kind": "scalar"}
                               for key, value in node.attrib.items()],
                "children": rendered, "content_tokens": tokens,
                "child_cardinality": [{"expanded_name": key, "count": value} for key, value in counts.items()]}

    return {"schema": SCHEMA, "value_scope": "parsed_element", "encoding": "none",
            "root": visit(element, root_path, 0, 0)}
