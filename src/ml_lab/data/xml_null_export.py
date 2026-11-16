"""ML-UW-21: lossless CSV/TSV nulls and per-column XML null provenance.

Only XML-C5/C7 consolidated outputs use this opt-in format. Ordinary CSV/TSV
files retain their previous semantics. Source XML is never rewritten.
"""
from __future__ import annotations

import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .xml_null_provenance import NULL_REASONS, XSI_NIL
from .xml_typed_records import _matches, _parts

SCHEMA = 'ml-lab.xml-null-delimited@1'
COMMENTS = 'Comments'
NULL_TOKEN = r'\N'
EMPTY_TOKEN = r'\E'
ESCAPE = '\\'
UNRESOLVED = 'unresolved_null'


def encoding_manifest() -> dict[str, Any]:
    return {
        'schema': SCHEMA, 'null_token': NULL_TOKEN,
        'empty_string_token': EMPTY_TOKEN, 'escape_prefix': ESCAPE,
        'comments_column': COMMENTS,
        'comments_format': 'json_object_column_to_reason',
        'reason_codes': sorted([*NULL_REASONS, UNRESOLVED]),
        'metadata_only_columns': [COMMENTS],
    }


def encode_value(value: Any) -> str:
    """Unambiguously encode None, empty strings, and literal token-like text."""
    if value is None:
        return NULL_TOKEN
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = str(value)
    if not text:
        return EMPTY_TOKEN
    return ESCAPE + text if text.startswith(ESCAPE) else text


def decode_value(value: str) -> str | None:
    if value == NULL_TOKEN:
        return None
    if value == EMPTY_TOKEN:
        return ''
    if value.startswith(ESCAPE + ESCAPE):
        return value[1:]
    return value


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_null_aware_collection_table(path: str | Path, *, manifest: str | Path | None = None) -> dict[str, Any]:
    """Validate an output and its encoding without materializing its rows."""
    target = Path(path).expanduser().absolute()
    manifest_path = (Path(manifest).expanduser().absolute() if manifest is not None
                     else target.with_suffix(target.suffix + '.collection.json'))
    if (target.is_symlink() or manifest_path.is_symlink() or
            not target.is_file() or not manifest_path.is_file()):
        raise ValueError('XML collection table and manifest must be regular local files')
    data = json.loads(manifest_path.read_text(encoding='utf-8'))
    if data.get('schema') != 'ml-lab.xml-collection-materialization@1':
        raise ValueError('Unrecognized XML collection materialization manifest')
    candidates = [data.get('main_dataset'), *(data.get('child_datasets') or [])]
    matches = [item for item in candidates if isinstance(item, dict) and item.get('path') == str(target)]
    if len(matches) != 1:
        raise ValueError('Table not declared in this XML collection manifest')
    record = matches[0]
    if record.get('null_encoding', {}).get('schema') != SCHEMA or _hash(target) != record.get('sha256'):
        raise ValueError('XML collection null encoding or dataset checksum is invalid')
    delimiter = '\t' if target.suffix.lower() == '.tsv' else ','
    with target.open('r', encoding='utf-8', newline='') as stream:
        reader = csv.reader(stream, delimiter=delimiter)
        try:
            headers = next(reader)
        except StopIteration:
            headers = []
    if headers != record['columns'] or len(headers) != len(set(headers)):
        raise ValueError('XML collection header differs from recorded schema')
    return record


def read_null_aware_collection_table(path: str | Path, *, manifest: str | Path | None = None) -> list[dict[str, Any]]:
    """Verified explicit reimport of an exported table (including empty/null distinction).

    A matching manifest is mandatory; never reinterpret generic CSV token text.
    """
    target = Path(path).expanduser().absolute()
    record = verify_null_aware_collection_table(target, manifest=manifest)
    delimiter = '\t' if target.suffix.lower() == '.tsv' else ','
    with target.open('r', encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream, delimiter=delimiter)
        result = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError('Invalid XML collection row width')
            result.append({key: (value if key == COMMENTS else decode_value(value))
                           for key, value in row.items()})
    if len(result) != record['row_count']:
        raise ValueError('XML collection row count differs from the recorded export')
    return result


def collect_record_null_reasons(path: Path, *, artifact: Any, record_root: str,
                                aligned_fields: Sequence[Mapping[str, Any]],
                                max_records: int, stop_after: int | None = None) -> list[dict[str, str]]:
    """Read exact nil/absent conditions from verified record roots, without coercion.

    Repeated fields with disagreeing observations are intentionally not assigned
    a single reason: an output-column strategy may change their row semantics.
    """
    if max_records < 1 or max_records > 10000 or (stop_after is not None and stop_after < 1):
        raise ValueError('XML record null-reason limit out of range')
    source = Path(path)
    if source.is_symlink() or str(source.resolve(strict=True)) != str(Path(artifact.source_path).resolve(strict=True)):
        raise ValueError('XML source path no longer matches its inspected artifact')
    if 'sha256:' + _hash(source) != artifact.source_fingerprint:
        raise ValueError('XML source changed before null-provenance collection')
    root_parts = _parts(record_root)
    prepared = []
    for field in aligned_fields:
        fid = field['field_id']
        kind = field['kind']
        element_parts = _parts(field['source_element_canonical_path'])
        if element_parts[:len(root_parts)] != root_parts or kind not in {'element_text', 'attribute'}:
            raise ValueError('Invalid or out-of-root canonical XML null field')
        prepared.append((fid, kind, element_parts, field.get('attribute_qname')))
    records: list[dict[str, str]] = []
    stack: list[str] = []
    with source.open('rb') as stream:
        for event, element in ET.iterparse(stream, events=('start', 'end')):
            if event == 'start':
                stack.append(str(element.tag))
                continue
            if tuple(stack) == root_parts:
                reasons: dict[str, str] = {}
                for fid, kind, parts, attr in prepared:
                    matched = _matches(element, parts, root_parts)
                    if not matched:
                        reasons[fid] = 'path_absent'
                    elif kind == 'attribute':
                        statuses = {'attribute_absent' if attr not in node.attrib else 'value'
                                    for node, _ in matched}
                        if statuses == {'attribute_absent'}:
                            reasons[fid] = 'attribute_absent'
                    else:
                        statuses = {'explicit_xsi_nil' if node.attrib.get(XSI_NIL, '').strip().lower() in {'true', '1'}
                                    else 'value' for node, _ in matched}
                        if statuses == {'explicit_xsi_nil'}:
                            reasons[fid] = 'explicit_xsi_nil'
                records.append(reasons)
                element.clear()
                if len(records) > max_records:
                    raise ValueError('XML source exceeds the null-provenance record budget')
                if stop_after is not None and len(records) >= stop_after:
                    break
            stack.pop()
    if 'sha256:' + _hash(source) != artifact.source_fingerprint:
        raise ValueError('XML source changed while collecting null provenance')
    return records


def explain_row_nulls(row: dict[str, Any], column_identities: Mapping[str, tuple[Any, ...]], *,
                      source_id: str, record_reasons: Mapping[str, str] | None,
                      alignment: Mapping[str, Mapping[str, Any]]) -> None:
    """Apply *proven* field nulls and annotate all remaining null columns.

    Source-level absence is valid even for exploded rows. Record-level status
    is used only when an extraction is known to be one-row-per-record.
    """
    comments: dict[str, str] = {}
    for name, identity in column_identities.items():
        if name == COMMENTS:
            raise ValueError('Comments is a reserved XML collection metadata column')
        field_ids = identity[2]
        reason = None
        if len(field_ids) == 1:
            fid = field_ids[0]
            field = alignment.get(fid, {})
            reason = field.get('absence_reason_by_source', {}).get(source_id)
            if reason is None and record_reasons is not None:
                reason = record_reasons.get(fid)
        if reason in NULL_REASONS:
            # Explicit nil and absent fields take precedence over any legacy
            # extractor representation of an empty XML element.
            row[name] = None
        if row.get(name) is None:
            comments[name] = reason if reason in NULL_REASONS else UNRESOLVED
    row[COMMENTS] = json.dumps(comments, ensure_ascii=False, sort_keys=True) if comments else ''


def is_null_aware_collection_main(path: str | Path) -> bool:
    """True only for a main dataset with a local XML collection manifest.

    The presence of a manifest does not establish trust: callers must use the
    verified reader, which checks the recorded dataset SHA-256 before decoding.
    """
    target = Path(path).expanduser().absolute()
    return target.with_suffix(target.suffix + '.collection.json').is_file()


def read_null_aware_collection_frame(path: str | Path):
    """Pandas view of a verified XML collection output, preserving null vs empty.

    Numeric columns are inferred only if *all* present values are numeric.
    Columns containing genuine empty strings stay textual, since empty != null.
    """
    import pandas as pd

    rows = read_null_aware_collection_table(path)
    frame = pd.DataFrame.from_records(rows)
    for column in frame.columns:
        if column == COMMENTS or column.startswith('__source_'):
            continue
        series = frame[column]
        values = series.dropna()
        if not values.empty and values.map(lambda value: value != '').all():
            try:
                converted = pd.to_numeric(series, errors='raise')
            except (ValueError, TypeError):
                continue
            frame[column] = converted
    return frame
