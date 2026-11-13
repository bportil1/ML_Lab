"""ML-UW-18: canonical XML field inventory and read-only null provenance.

This contract is intentionally *not* a collection grouping, extraction or export
implementation. ML-UW-19--21 may consume its stable field keys and condition
codes; existing C1--C7 behavior stays unchanged until those later increments.

A field may have multiple occurrences: they are preserved, never silently
coalesced into a scalar or interpreted as an ordinary missing value.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from .types import XmlFieldCandidate, XmlStructureArtifact
from .xml_ingest import inspect_xml_structure
from .xml_selection import build_xml_record_selection
from .xml_typed_records import _matches, _parts

INVENTORY_SCHEMA = 'ml-lab.xml-null-field-inventory@1'
PREVIEW_SCHEMA = 'ml-lab.xml-null-provenance-preview@1'
XSI_NIL = '{http://www.w3.org/2001/XMLSchema-instance}nil'
NULL_REASONS = frozenset({'path_absent', 'attribute_absent', 'explicit_xsi_nil'})
STATUSES = frozenset({'value', 'empty_element', *NULL_REASONS})


class XmlNullProvenanceError(ValueError):
    """Invalid field catalog, stale source, or unsupported preview request."""


def _candidate_fields(artifact: XmlStructureArtifact, root: str) -> list[XmlFieldCandidate]:
    """Include leaf elements that are *always empty* in inspected sources.

    Legacy scalar selection intentionally omits them because it is designed
    for extracting nonempty values. Here we need their existence to distinguish
    an empty element from a missing path in later records.
    """
    selected = list(build_xml_record_selection(artifact, root).fields)
    known = {field.field_id for field in selected}
    by_path = {profile.canonical_path: profile for profile in artifact.element_profiles}
    root_profile = by_path[root]
    for profile in artifact.element_profiles:
        path = profile.canonical_path
        if (not path.startswith(root + '/') or profile.child_paths or profile.mixed_content
                or profile.text_occurrence_count > 0 or 'text:' + path in known):
            continue
        chain = []
        cursor = path
        while cursor != root:
            chain.append(by_path[cursor])
            cursor = cursor.rsplit('/', 1)[0]
        selected.append(XmlFieldCandidate(
            field_id='text:' + path, kind='element_text',
            path=profile.path, canonical_path=path,
            relative_path=profile.path[len(root_profile.path.rstrip('/')) + 1:],
            source_element_path=profile.path,
            source_element_canonical_path=path,
            name=profile.local_name, local_name=profile.local_name,
            namespace_uri=profile.namespace_uri, prefix=profile.prefix,
            repeated=any(p.repeated for p in chain),
            optional=True,
        ))
    return selected


def catalog_xml_null_fields(
    artifacts: Mapping[str, XmlStructureArtifact], record_root_canonical_path: str,
) -> dict[str, Any]:
    """Build a namespace-aware union *inventory*, not a compatibility decision.

    ``column_key`` is the collision-free canonical field ID. Human-friendly
    output column names are deliberately deferred to the validated C4 recipe.
    Field IDs use expanded QName URIs, so differing namespace prefixes merge
    while identically named fields from different namespace URIs remain distinct.
    """
    if not artifacts or not isinstance(artifacts, Mapping):
        raise XmlNullProvenanceError('at least one inspected XML source is required')
    root_parts = _parts(record_root_canonical_path)
    if any(not isinstance(source_id, str) or not source_id for source_id in artifacts):
        raise XmlNullProvenanceError('source IDs must be nonempty strings')
    fields: dict[str, dict[str, Any]] = {}
    fingerprints: dict[str, str] = {}
    source_paths: dict[str, str] = {}
    for source_id, artifact in sorted(artifacts.items()):
        if not isinstance(source_id, str) or not source_id or not isinstance(artifact, XmlStructureArtifact):
            raise XmlNullProvenanceError('source IDs and XML structure artifacts are required')
        if source_id in fingerprints:
            raise XmlNullProvenanceError('duplicate source ID')
        if record_root_canonical_path not in {p.canonical_path for p in artifact.element_profiles}:
            raise XmlNullProvenanceError(f'{source_id}: chosen record root is absent')
        fingerprints[source_id] = artifact.source_fingerprint
        source_paths[source_id] = artifact.source_path
        for field in _candidate_fields(artifact, record_root_canonical_path):
            # Do not collapse distinct attribute namespaces or text/attribute kinds.
            element_parts = _parts(field.source_element_canonical_path)
            if element_parts[:len(root_parts)] != root_parts:
                raise XmlNullProvenanceError(f'{source_id}: field escapes record root')
            attribute_qname = (field.canonical_path[len(field.source_element_canonical_path) + 2:]
                               if field.kind == 'attribute' else None)
            entry = fields.setdefault(field.field_id, {
                'field_id': field.field_id, 'column_key': field.field_id,
                'kind': field.kind, 'canonical_path': field.canonical_path,
                'source_element_canonical_path': field.source_element_canonical_path,
                'attribute_qname': attribute_qname, 'display_path_hint': field.relative_path,
                'present_in': [], 'optional_in': [], 'repeated_in': [],
            })
            if (entry['kind'] != field.kind or entry['canonical_path'] != field.canonical_path or
                    entry['source_element_canonical_path'] != field.source_element_canonical_path or
                    entry['attribute_qname'] != attribute_qname):
                raise XmlNullProvenanceError(f'{source_id}: inconsistent canonical field identity')
            entry['present_in'].append(source_id)
            if field.optional:
                entry['optional_in'].append(source_id)
            if field.repeated:
                entry['repeated_in'].append(source_id)
    sources = sorted(fingerprints)
    ordered = []
    for field_id in sorted(fields):
        entry = fields[field_id]
        entry['missing_in'] = [source for source in sources if source not in entry['present_in']]
        ordered.append(entry)
    return {
        'schema': INVENTORY_SCHEMA,
        'record_root_canonical_path': record_root_canonical_path,
        'sources': fingerprints,
        'source_paths': source_paths,
        'fields': ordered,
        'source_count': len(sources),
        'notes': ['A missing path is not a structural incompatibility decision.',
                  'A repeated field needs an explicit extraction strategy.',
                  'column_key is stable; final display/export names belong to C4.'],
    }


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _observation(element: ET.Element, kind: str, attribute_qname: str | None,
                 address: tuple[int, ...]) -> dict[str, Any]:
    if kind == 'attribute':
        if attribute_qname is None:
            raise XmlNullProvenanceError('attribute name missing from field catalog')
        if attribute_qname not in element.attrib:
            status, value = 'attribute_absent', None
        else:
            status, value = 'value', element.attrib[attribute_qname]
    elif kind == 'element_text':
        if element.attrib.get(XSI_NIL, '').strip().lower() in {'true', '1'}:
            status, value = 'explicit_xsi_nil', None
        elif element.text is None or not element.text.strip():
            status, value = 'empty_element', ''
        else:
            # Preserve original textual contents; coercion and whitespace
            # normalization must be explicit in the later recipe.
            status, value = 'value', element.text
    else:
        raise XmlNullProvenanceError(f'unsupported field kind: {kind}')
    return {'status': status, 'value': value,
            'null_reason': status if status in NULL_REASONS else None,
            'raw_text': element.text if kind == 'element_text' else None,
            'occurrence_address': list(address)}


def preview_xml_null_provenance(
    path: str | Path, artifact: XmlStructureArtifact, inventory: Mapping[str, Any],
    source_id: str, *, max_records: int = 50,
) -> dict[str, Any]:
    """Inspect bounded records without modifying XML, recipes, or datasets.

    Every expected field appears for every previewed record. A missing field
    gets a typed None plus the exact reason. Repeated values are returned as
    separate observations; they are *not* a single scalar null.
    """
    if not isinstance(max_records, int) or isinstance(max_records, bool) or not 1 <= max_records <= 500:
        raise XmlNullProvenanceError('max_records must be between 1 and 500')
    if not isinstance(inventory, Mapping) or inventory.get('schema') != INVENTORY_SCHEMA:
        raise XmlNullProvenanceError('unsupported XML field inventory')
    if not isinstance(artifact, XmlStructureArtifact):
        raise XmlNullProvenanceError('expected a parsed XML structure artifact')
    sources = inventory.get('sources')
    if not isinstance(sources, dict) or sources.get(source_id) != artifact.source_fingerprint:
        raise XmlNullProvenanceError('source is not part of this field inventory or its fingerprint changed')
    root = inventory.get('record_root_canonical_path')
    root_parts = _parts(root)
    selected_fields = inventory.get('fields')
    if not isinstance(selected_fields, list) or not selected_fields:
        raise XmlNullProvenanceError('field inventory has no selectable fields')
    # Validate the field IDs and canonical paths before traversing the source.
    prepared = []
    seen: set[str] = set()
    for field in selected_fields:
        if not isinstance(field, Mapping) or not isinstance(field.get('field_id'), str):
            raise XmlNullProvenanceError('invalid field inventory entry')
        identifier = field['field_id']
        if identifier in seen or field.get('column_key') != identifier:
            raise XmlNullProvenanceError('duplicate or inconsistent field column key')
        seen.add(identifier)
        kind = field.get('kind')
        path_parts = _parts(field.get('source_element_canonical_path'))
        if path_parts[:len(root_parts)] != root_parts:
            raise XmlNullProvenanceError('field is outside the selected record root')
        if kind == 'attribute':
            attr = field.get('attribute_qname')
            if not isinstance(attr, str) or not attr or identifier != 'attribute:' + field['source_element_canonical_path'] + '@' + attr:
                raise XmlNullProvenanceError('inconsistent attribute field identity')
        elif kind == 'element_text':
            attr = None
            if identifier != 'text:' + field['source_element_canonical_path']:
                raise XmlNullProvenanceError('inconsistent text field identity')
        else:
            raise XmlNullProvenanceError('unsupported field kind')
        prepared.append((identifier, kind, path_parts, attr))
    source = Path(path).expanduser()
    if source.is_symlink():
        raise XmlNullProvenanceError('XML source must not be a symlink')
    source = source.resolve(strict=True)
    if source != Path(artifact.source_path).resolve() or str(source) != inventory.get('source_paths', {}).get(source_id):
        raise XmlNullProvenanceError('XML source location does not match inspected inventory')
    current = inspect_xml_structure(source)
    if current.source_fingerprint != artifact.source_fingerprint:
        raise XmlNullProvenanceError('XML source changed since structural inspection')
    if root not in {p.canonical_path for p in current.element_profiles}:
        raise XmlNullProvenanceError('record root disappeared from XML source')
    root_count = next(p.occurrence_count for p in current.element_profiles if p.canonical_path == root)
    records: list[dict[str, Any]] = []
    stack: list[str] = []
    with source.open('rb') as stream:
        for event, element in ET.iterparse(stream, events=('start', 'end')):
            if event == 'start':
                stack.append(str(element.tag))
                continue
            if tuple(stack) == root_parts:
                cells = []
                for identifier, kind, parts, attr in prepared:
                    matches = _matches(element, parts, root_parts)
                    if not matches:
                        observations = [{'status': 'path_absent', 'value': None,
                                         'null_reason': 'path_absent', 'raw_text': None,
                                         'occurrence_address': []}]
                        status = 'path_absent'
                    else:
                        observations = [_observation(node, kind, attr, address)
                                        for node, address in matches]
                        status = observations[0]['status'] if len(observations) == 1 else 'multiple'
                    cells.append({'field_id': identifier, 'column_key': identifier,
                                  'status': status, 'occurrence_count': len(matches),
                                  'observations': observations})
                records.append({'record_index': len(records),
                                'record_number': len(records) + 1, 'cells': cells})
                element.clear()
                if len(records) >= max_records:
                    break
            stack.pop()
    if _hash_file(source) != artifact.source_sha256:
        raise XmlNullProvenanceError('XML source changed during null-provenance preview')
    return {'schema': PREVIEW_SCHEMA, 'field_inventory_schema': INVENTORY_SCHEMA,
            'source_id': source_id, 'source_fingerprint': artifact.source_fingerprint,
            'record_root_canonical_path': root,
            'source_record_count': root_count, 'preview_record_count': len(records),
            'truncated': len(records) < root_count, 'records': records,
            'registered_dataset': False, 'source_modified': False}
