"""Adapt existing XML *structure profile artifacts*, never unverified raw XML.

An XmlStructureArtifact summarizes canonical paths and occurrence statistics;
it does not contain actual per-instance edges or values. Do not invent these.
The original safe XML parser and provenance verification remain responsible for
source inspection before conversion; the adapter performs no filesystem IO.
"""
from __future__ import annotations

from collections.abc import Mapping
import re

from ..contracts import CONTRACT_SCHEMA, StructuredContractError
from ..object import StructuredObject
from .registry import AdaptationResult

XML_SCHEMA = 'ml-lab.xml-structure@2'
_SHA = re.compile(r'^[0-9a-fA-F]{64}$')


def _parent_path(path: str) -> str | None:
    if not isinstance(path, str) or not path.startswith('/') or path == '/':
        raise StructuredContractError('invalid canonical XML path')
    parts = path.split('/')[1:]
    if not parts or not all(parts):
        raise StructuredContractError('empty canonical XML path segment')
    # XML canonical segments use JSON Pointer escapes for '/' and '~'.
    for part in parts:
        if re.search(r'~(?![01])', part):
            raise StructuredContractError('invalid XML path escape')
    return '/'.join(path.split('/')[:-1]) or None


def _positive_int(value: object, what: str) -> int:
    if type(value) is not int or value < 0:
        raise StructuredContractError(f'{what} must be a nonnegative integer')
    return value


class XMLStructureAdapter:
    """Convert an inspected XML profile, not its original XML instances."""

    def adapt(self, source: object) -> AdaptationResult:
        if hasattr(source, 'to_record') and callable(source.to_record):
            source = source.to_record()
        if not isinstance(source, Mapping) or source.get('schema') != XML_SCHEMA:
            raise StructuredContractError('expected versioned XML structure artifact')
        if not isinstance(source.get('source_sha256'), str) or not _SHA.fullmatch(source['source_sha256']):
            raise StructuredContractError('missing valid XML source SHA256 claim')
        if not isinstance(source.get('source_fingerprint'), str) or not source['source_fingerprint']:
            raise StructuredContractError('missing XML source fingerprint claim')
        profiles = source.get('element_profiles')
        if not isinstance(profiles, (list, tuple)) or not profiles:
            raise StructuredContractError('XML structure requires element profiles')
        root = source.get('root')
        if not isinstance(root, Mapping) or not isinstance(root.get('tag'), str):
            raise StructuredContractError('XML structure requires root metadata')
        nodes: list[dict] = []
        edges: list[dict] = []
        paths: set[str] = set()
        roots: list[str] = []
        for index, profile in enumerate(profiles):
            if not isinstance(profile, Mapping):
                raise StructuredContractError('XML element profile must be a mapping')
            path = profile.get('canonical_path')
            parent = _parent_path(path)
            tag = profile.get('tag')
            if not isinstance(tag, str) or not tag or not isinstance(profile.get('local_name'), str):
                raise StructuredContractError('XML element requires valid tag and local name')
            expected_segment = tag.replace('~', '~0').replace('/', '~1')
            if path.rsplit('/', 1)[-1] != expected_segment:
                raise StructuredContractError('canonical XML path does not match expanded tag')
            if path in paths:
                raise StructuredContractError('duplicate canonical XML element path')
            paths.add(path)
            count = _positive_int(profile.get('occurrence_count'), 'XML element occurrence_count')
            if parent is None:
                roots.append(path)
            else:
                edges.append({'id': 'xml-element-edge:' + path, 'kind': 'contains',
                              'source_id': parent, 'target_id': path})
            nodes.append({'id': path, 'kind': 'element',
                          'label': profile.get('local_name'),
                          'attributes': {'canonical_path': path, 'xml_tag': profile.get('tag'),
                                         'namespace_uri': profile.get('namespace_uri'),
                                         'occurrence_count': count},
                          'metadata': {'source_profile': profile}})
            attr_profiles = profile.get('attributes', [])
            if not isinstance(attr_profiles, (list, tuple)):
                raise StructuredContractError('XML attributes must be an array')
            for attr in attr_profiles:
                if not isinstance(attr, Mapping) or not isinstance(attr.get('name'), str) or not attr['name']:
                    raise StructuredContractError('invalid XML attribute profile')
                name = attr['name']
                aid = 'xml-attribute:' + path + '@' + name
                if aid in paths:
                    raise StructuredContractError('duplicate XML attribute path')
                paths.add(aid)
                attr_count = _positive_int(attr.get('occurrence_count'), 'XML attribute occurrence_count')
                if attr_count > count:
                    raise StructuredContractError('XML attribute count exceeds element occurrence count')
                nodes.append({'id': aid, 'kind': 'attribute',
                              'label': attr.get('local_name'),
                              'attributes': {'canonical_path': aid, 'xml_name': name,
                                             'namespace_uri': attr.get('namespace_uri'),
                                             'occurrence_count': attr['occurrence_count']},
                              'metadata': {'source_profile': attr}})
                edges.append({'id': 'xml-attribute-edge:' + aid, 'kind': 'has_attribute',
                              'source_id': path, 'target_id': aid})
        if len(roots) != 1:
            raise StructuredContractError('XML structure must have exactly one canonical root')
        root_tag = root['tag'].replace('~', '~0').replace('/', '~1')
        if roots[0] != '/' + root_tag:
            raise StructuredContractError('canonical XML root differs from declared root')
        if any(edge['source_id'] not in paths for edge in edges):
            raise StructuredContractError('XML profile contains a path with no parent profile')
        obj = StructuredObject.from_dict({
            'schema': CONTRACT_SCHEMA, 'topology': {'kind': 'tree'},
            'nodes': nodes, 'edges': edges,
            'observation_unit': 'source', 'observations': [],
            'provenance': {'adapter': 'xml-structure@1', 'source_schema': XML_SCHEMA,
                           'source_fingerprint_claim': source['source_fingerprint'],
                           'source_sha256_claim': source['source_sha256'],
                           'source_path_claim': source.get('source_path'),
                           'profile_scope': 'aggregate_canonical_paths',
                           'source_summary': {k: v for k, v in source.items()
                                              if k not in {'schema', 'source_path',
                                                           'source_sha256', 'source_fingerprint',
                                                           'element_profiles'}}},
        })
        return AdaptationResult(obj, XML_SCHEMA,
            warnings=('Source hashes are artifact claims; the adapter does not re-read the XML file.',),
            losses=('individual_occurrence_topology', 'original_sibling_order',
                    'record_level_text_and_attribute_values'))
