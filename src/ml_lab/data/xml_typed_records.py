"""Opt-in, bounded typed XML preview backed by ML Lab's existing safe parser.

The result is a preview artifact, not a registered dataset or an encoding step.
Nothing here changes C1--C7 extraction recipes or their output.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from .types import XmlStructureArtifact
from .xml_ingest import inspect_xml_structure
from .xml_typed_representation import (
    KINDS, RECIPE_SCHEMA, SCHEMA, XmlTypedRepresentationError,
    represent_xml_element,
)

PREVIEW_SCHEMA = "ml-lab.xml-typed-preview@1"


def _parts(path: str) -> tuple[str, ...]:
    if not isinstance(path, str) or not path.startswith("/") or path == "/":
        raise XmlTypedRepresentationError("invalid absolute canonical path")
    values = path.split("/")[1:]
    if not all(values):
        raise XmlTypedRepresentationError("empty canonical path segment")
    # The canonical XML path uses RFC 6901 escaping for slash and tilde.
    for value in values:
        for index, char in enumerate(value):
            if char == "~" and (index + 1 >= len(value) or value[index + 1] not in "01"):
                raise XmlTypedRepresentationError("invalid canonical path escape")
    return tuple(value.replace("~1", "/").replace("~0", "~") for value in values)


def _validate(artifact: XmlStructureArtifact, recipe: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...], tuple[str, ...]]:
    if not isinstance(recipe, Mapping) or recipe.get("schema") != RECIPE_SCHEMA or recipe.get("version") != 1:
        raise XmlTypedRepresentationError("unsupported typed recipe schema/version")
    if recipe.get("encoding") != "none" or recipe.get("legacy_extraction_modified") is not False:
        raise XmlTypedRepresentationError("typed preview cannot encode or modify legacy extraction")
    root = recipe.get("record_root_canonical_path")
    root_parts = _parts(root)
    profiles = {item.canonical_path for item in artifact.element_profiles}
    if root not in profiles:
        raise XmlTypedRepresentationError("record root not present in structure artifact")
    variables = recipe.get("variables")
    if not isinstance(variables, list) or not variables:
        raise XmlTypedRepresentationError("typed recipe requires variable declarations")
    selected: list[tuple[str, str]] = []
    seen: set[str] = set()
    for variable in variables:
        if not isinstance(variable, Mapping):
            raise XmlTypedRepresentationError("invalid variable declaration")
        path, kind = variable.get("canonical_path"), variable.get("representation")
        parts = _parts(path)
        if path not in profiles or parts[:len(root_parts)] != root_parts or kind not in KINDS or path in seen:
            raise XmlTypedRepresentationError("unknown, duplicate or out-of-root typed variable")
        if kind == "linked_child_records":
            raise XmlTypedRepresentationError("linked child records must use linked_child_paths")
        selected.append((path, kind))
        seen.add(path)
    links = recipe.get("linked_child_paths", [])
    if not isinstance(links, list) or len(links) != len(set(links)):
        raise XmlTypedRepresentationError("invalid linked child paths")
    for path in links:
        parts = _parts(path)
        if path not in profiles or len(parts) <= len(root_parts) or parts[:len(root_parts)] != root_parts:
            raise XmlTypedRepresentationError("linked child path is unknown or outside record root")
    return root_parts, tuple(selected), tuple(links)


def _matches(record: ET.Element, path_parts: tuple[str, ...], root_parts: tuple[str, ...]):
    """Yield elements with stable occurrence-addresses along their ancestor chain."""
    active = [(record, (0,))]
    for tag in path_parts[len(root_parts):]:
        new = []
        for parent, address in active:
            counts: Counter[str] = Counter()
            for child in parent:
                index = counts[str(child.tag)]
                counts[str(child.tag)] += 1
                if str(child.tag) == tag:
                    new.append((child, (*address, index)))
        active = new
        if not active:
            break
    return active


def _render(element: ET.Element, kind: str, path: str, address: tuple[int, ...], max_nodes: int):
    lineage = {"canonical_path": path, "occurrence_address": list(address)}
    if kind == "scalar":
        value: Any = {"kind": "scalar", "present": element.text is not None, "value": element.text}
    elif kind == "content_tokens":
        value = {"kind": kind, "tokens": represent_xml_element(element, canonical_path=path, max_nodes=max_nodes)["root"]["content_tokens"]}
    elif kind == "path_tokens":
        value = {"kind": kind, "segments": list(_parts(path)), "occurrence_address": list(address)}
    else:
        value = represent_xml_element(element, canonical_path=path, max_nodes=max_nodes)["root"]
        if kind == "list":
            value = {"kind": "list_item", "element": value}
    return {"lineage": lineage, "value": value}


def preview_typed_xml_records(
    path: str | Path, artifact: XmlStructureArtifact, recipe: Mapping[str, Any],
    *, max_records: int = 25, max_nodes_per_value: int = 10_000,
) -> dict[str, Any]:
    """Read bounded source records with verified source identity and explicit choice of shape.

    Calling this is an *opt-in* preview. Values are unconverted XML strings; no
    SQL, artifacts or registered datasets are created. Count and missingness are
    recorded per variable even when the structure declared that path optional.
    """
    if not 1 <= max_records <= 200 or not 1 <= max_nodes_per_value <= 100_000:
        raise XmlTypedRepresentationError("typed preview limits exceeded")
    root_parts, selected, links = _validate(artifact, recipe)
    source = Path(path).expanduser().resolve()
    current = inspect_xml_structure(source)  # Also enforces source/DTD/entity/depth/byte policy.
    if current.source_fingerprint != artifact.source_fingerprint:
        raise XmlTypedRepresentationError("XML source differs from the inspected structure")
    # Validate the originally selected structure against the newly inspected one.
    _validate(current, recipe)
    records: list[dict[str, Any]] = []
    stack: list[str] = []
    root_path = recipe["record_root_canonical_path"]
    with source.open("rb") as handle:
        for event, element in ET.iterparse(handle, events=("start", "end")):
            if event == "start":
                stack.append(str(element.tag))
                continue
            if tuple(stack) == root_parts:
                record_index = len(records)
                variables: list[dict[str, Any]] = []
                for field_path, kind in selected:
                    matches = _matches(element, _parts(field_path), root_parts)
                    variables.append({
                        "canonical_path": field_path, "representation": kind,
                        "present": bool(matches), "count": len(matches),
                        "items": [_render(node, kind, field_path, address, max_nodes_per_value)
                                  for node, address in matches],
                    })
                children: list[dict[str, Any]] = []
                for linked_path in links:
                    matches = _matches(element, _parts(linked_path), root_parts)
                    children.append({
                        "canonical_path": linked_path, "count": len(matches),
                        "records": [
                            {"parent_record_index": record_index, "occurrence_address": list(address),
                             "representation": represent_xml_element(node, canonical_path=linked_path, max_nodes=max_nodes_per_value)["root"]}
                            for node, address in matches
                        ],
                    })
                records.append({"record_index": record_index, "record_number": record_index + 1,
                                "record_root_canonical_path": root_path,
                                "variables": variables, "linked_children": children})
                element.clear()
                if len(records) >= max_records:
                    break
            stack.pop()

    root_profile = next(profile for profile in artifact.element_profiles if profile.canonical_path == root_path)
    return {
        "schema": PREVIEW_SCHEMA, "representation_schema": SCHEMA,
        "recipe_schema": RECIPE_SCHEMA, "source_sha256": current.source_sha256,
        "source_fingerprint": current.source_fingerprint,
        "record_root_canonical_path": root_path, "encoding": "none",
        "value_scope": "bounded_source_records", "records": records,
        "preview_record_count": len(records),
        "source_record_count": root_profile.occurrence_count,
        "truncated": root_profile.occurrence_count > len(records),
        "registered_dataset": False,
    }
