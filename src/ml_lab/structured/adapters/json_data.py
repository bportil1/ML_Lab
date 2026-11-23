"""Convert JSON values into a canonical *instance* tree.

JSON Pointer identities are independent of display labels and object-key order.
Array order is retained as edge ordinal; JSON object key order is not asserted.
No statistical sample is inferred from a single JSON document.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math

from ..contracts import CONTRACT_SCHEMA, StructuredContractError
from ..object import StructuredObject, _unique_json_pairs, _reject_json_constant
from .registry import AdaptationResult


def _kind(value: object) -> str:
    if value is None:
        return 'null'
    if type(value) is bool:
        return 'boolean'
    if type(value) is int:
        return 'integer'
    if type(value) is float:
        if not math.isfinite(value):
            raise StructuredContractError('JSON data must not contain NaN or Infinity')
        return 'number'
    if isinstance(value, str):
        return 'string'
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise StructuredContractError('JSON object keys must be strings')
        return 'object'
    if isinstance(value, (list, tuple)):
        return 'array'
    raise StructuredContractError(f'unsupported JSON value type: {type(value).__name__}')


def _pointer_escape(text: str) -> str:
    return text.replace('~', '~0').replace('/', '~1')


class JSONAdapter:
    """Adapt parsed JSON-compatible values or strict JSON text (explicit API)."""

    def __init__(self, *, max_nodes: int = 50_000, max_depth: int = 110):
        if not 1 <= max_nodes <= 1_000_000 or not 1 <= max_depth <= 110:
            raise StructuredContractError('invalid JSON adapter resource limits')
        self.max_nodes = max_nodes
        self.max_depth = max_depth

    def from_text(self, text: str | bytes | bytearray) -> AdaptationResult:
        if not isinstance(text, (str, bytes, bytearray)):
            raise StructuredContractError('JSON input must be text')
        try:
            data = json.loads(text, object_pairs_hook=_unique_json_pairs,
                              parse_constant=_reject_json_constant)
        except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
            raise StructuredContractError(f'invalid JSON source: {exc}') from exc
        return self.adapt(data)

    def adapt(self, source: object) -> AdaptationResult:
        nodes: list[dict] = []
        edges: list[dict] = []
        # Sort object keys (unordered in JSON); preserve array indexing.
        stack = [('#', '$', source, 0)]
        while stack:
            pointer, label, value, depth = stack.pop()
            if depth > self.max_depth:
                raise StructuredContractError('JSON source exceeds maximum depth')
            if len(nodes) >= self.max_nodes:
                raise StructuredContractError('JSON source exceeds maximum node count')
            category = _kind(value)
            attributes = {'json_type': category, 'json_pointer': pointer}
            if category not in {'object', 'array'}:
                attributes['value_state'] = ('explicit_null' if value is None
                                             else 'empty' if value == '' else 'present')
                attributes['value'] = value
            nodes.append({'id': pointer, 'kind': category, 'label': label,
                          'attributes': attributes})
            if category == 'object':
                entries = [(k, value[k], None) for k in sorted(value)]
            elif category == 'array':
                entries = [(str(i), item, i) for i, item in enumerate(value)]
            else:
                entries = []
            for key, child, ordinal in reversed(entries):
                child_ptr = pointer + '/' + _pointer_escape(key)
                edge = {'id': 'edge:' + child_ptr, 'kind': 'contains',
                        'source_id': pointer, 'target_id': child_ptr}
                if ordinal is not None:
                    edge['ordinal'] = ordinal
                edges.append(edge)
                stack.append((child_ptr, key, child, depth + 1))
        # Sort edges to make equivalent maps deterministic independent of traversal.
        edges.sort(key=lambda edge: edge['id'])
        # Fingerprint canonicalized content, not a path or an unchecked declaration.
        source_bytes = json.dumps(source, ensure_ascii=False, sort_keys=True,
                                  separators=(',', ':'), allow_nan=False).encode('utf-8')
        digest = hashlib.sha256(source_bytes).hexdigest()
        obj = StructuredObject.from_dict({
            'schema': CONTRACT_SCHEMA,
            'topology': {'kind': 'tree'}, 'nodes': nodes, 'edges': edges,
            'observation_unit': 'source', 'observations': [],
            'provenance': {'adapter': 'json@1', 'source_schema': 'json-value',
                           'content_sha256': digest,
                           'fingerprint_scope': 'canonical_json_value'},
        })
        return AdaptationResult(obj, 'json-value')
