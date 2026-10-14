"""XML-C7: bounded per-source XML collection export with restart-safe checkpoints.

The existing XML extraction routine is still bounded *per source* (maximum
10,000 generated main rows per source). Unlike C5, no collection-wide row list
is held in memory. Completed source extracts are staged with checksums, enabling
recovery without re-extracting completed, unchanged inputs.

No SQLite database is created or opened here. The handoff function returns a
module-owned, versioned logical catalog projection for a later ML_Lab adapter.
"""
from __future__ import annotations

import argparse
import fcntl
from contextlib import contextmanager
import csv
import json
import os
import re
import shutil
import sys
import tempfile
from collections import OrderedDict
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .xml_collection_export import XmlCollectionExportError, _sha, _verify_recipe, _rules_for
from .xml_collection_provenance import (
    _atomic_json, _hash, make_collection_event, provenance_path, verify_collection_event,
)
from .xml_ingest import inspect_xml_structure
from .xml_preview import preview_xml_tabularization

STATE_SCHEMA = 'ml-lab.xml-collection-scaled-state@1'
HANDOFF_SCHEMA = 'ml-lab.xml-collection-sqlite-handoff@1'
ProgressCallback = Callable[[dict[str, Any]], None]


def _paths(output: str | Path) -> tuple[Path, Path, Path]:
    target = Path(output).expanduser().absolute()
    if target.suffix.lower() not in {'.csv', '.tsv'}:
        raise XmlCollectionExportError('Scaled export requires a .csv or .tsv path')
    if target.is_symlink():
        raise XmlCollectionExportError('Output symlinks are not permitted')
    return (target, target.with_suffix(target.suffix + '.c7.state.json'),
            target.parent / ('.' + target.name + '.c7.parts'))




@contextmanager
def _output_lock(target: Path):
    """Serialize competing exporters for one output, including resumed exports.

    The lock file intentionally persists so removing it cannot open an inode
    replacement race with another waiting exporter. Linux/POSIX file locking
    is required; no network service or private database is involved.
    """
    lock_path = target.with_suffix(target.suffix + '.c7.lock')
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(lock_path, flags, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise XmlCollectionExportError('Another XML-C7 export or recovery is already running') from exc
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def inspect_xml_collection_scaled(output: str | Path) -> dict[str, Any]:
    """Read a local checkpoint for monitoring; never mutate a running job."""
    target, state_path, _ = _paths(output)
    if not state_path.exists():
        return {'schema': STATE_SCHEMA, 'status': 'no_checkpoint', 'output': str(target)}
    if state_path.is_symlink() or not state_path.is_file():
        raise XmlCollectionExportError('Unsafe or unreadable C7 checkpoint')
    data = json.loads(state_path.read_text(encoding='utf-8'))
    if data.get('schema') != STATE_SCHEMA or data.get('output') != str(target):
        raise XmlCollectionExportError('Unrecognized C7 checkpoint')
    return {'schema': STATE_SCHEMA, 'status': data['status'], 'output': str(target),
            'completed_sources': len(data['completed']), 'total_sources': len(data['source_membership']),
            'source_membership': list(data['source_membership']),
            'last_completed_source': data['completed'][-1]['relative_path'] if data['completed'] else None,
            'note': 'Checkpoint is non-authoritative; only final C6 provenance verifies publication'}


def _emit(callback: ProgressCallback | None, state: Mapping[str, Any], phase: str) -> None:
    if callback is None:
        return
    event = {'schema': 'ml-lab.xml-collection-job-event@1',
             'operation': 'xml_collection_scaled_export', 'phase': phase,
             'completed_sources': len(state['completed']),
             'total_sources': len(state['source_membership']),
             'status': state['status']}
    # Monitoring must not affect calculation, scientific output, or publication.
    try:
        callback(event)
    except Exception:
        pass


def _part_file(directory: Path, index: int) -> Path:
    return directory / f'{index:06d}.json'


def _file_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_json(path, value)


def _column_record(col: Any) -> dict[str, Any]:
    return {'name': col.name, 'strategy': col.strategy,
            'branch': col.repeated_branch_canonical_path,
            'field_ids': list(col.source_field_ids),
            'element_paths': list(col.source_element_canonical_paths)}


def _extract_part(path: Path, source: dict[str, Any], recipe: dict[str, Any], *,
                  max_source_bytes: int, max_source_records: int,
                  max_child_rows_per_source: int) -> dict[str, Any]:
    rel = source['relative_path']
    if source['record_root_occurrences'] > max_source_records:
        raise XmlCollectionExportError(f'{rel}: exceeds per-source record limit ({max_source_records})')
    artifact = inspect_xml_structure(path, max_source_bytes=max_source_bytes)
    if artifact.source_fingerprint != source['source_fingerprint']:
        raise XmlCollectionExportError(f'{rel}: source changed before extraction')
    extracted = preview_xml_tabularization(
        path, artifact, recipe['record_root_canonical_path'],
        source['present_field_ids'], rules=_rules_for(source, recipe['rules']),
        max_rows=None, max_child_rows=None, max_output_rows=10000,
    )
    if extracted.main_table is None or extracted.main_table.truncated or any(t.truncated for t in extracted.child_tables):
        raise XmlCollectionExportError(f'{rel}: per-source extract was truncated; no partial rows are accepted')
    if any(len(table.rows) > max_child_rows_per_source for table in extracted.child_tables):
        raise XmlCollectionExportError(f'{rel}: child table exceeds per-source row limit')
    if _sha(path) != artifact.source_sha256:
        raise XmlCollectionExportError(f'{rel}: source changed during extraction')
    return {
        'schema': 'ml-lab.xml-collection-source-part@1', 'relative_path': rel,
        'sha256': artifact.source_sha256,
        'main': {'columns': [_column_record(c) for c in extracted.main_table.columns],
                 'rows': extracted.main_table.rows},
        'children': [{'name': table.name, 'columns': [_column_record(c) for c in table.columns],
                      'rows': table.rows} for table in extracted.child_tables],
        'source_report': {
            'relative_path': rel, 'sha256': artifact.source_sha256,
            'root_record_count': source['record_root_occurrences'],
            'output_row_count': len(extracted.main_table.rows),
            'child_row_counts': {t.name: len(t.rows) for t in extracted.child_tables},
            'preview_signature': extracted.preview_signature,
            'missing_field_ids': source['missing_field_ids'],
        },
        'warnings': [f'{rel}: {w}' for w in extracted.warnings],
    }


def _load_part(directory: Path, index: int, entry: Mapping[str, Any]) -> dict[str, Any]:
    path = _part_file(directory, index)
    if path.is_symlink() or not path.is_file() or _sha(path) != entry.get('part_sha256'):
        raise XmlCollectionExportError(f'Completed source checkpoint is missing or corrupted: {path.name}')
    part = json.loads(path.read_text(encoding='utf-8'))
    if part.get('schema') != 'ml-lab.xml-collection-source-part@1' or (
        part.get('relative_path') != entry['relative_path'] or part.get('sha256') != entry['sha256']
    ):
        raise XmlCollectionExportError('Checkpoint part identity is inconsistent')
    return part


def _register(columns: list[dict[str, Any]], headers: OrderedDict[str, tuple[Any, ...]],
              seen: dict[tuple[Any, ...], str], *, child: bool = False) -> None:
    for column in columns:
        name = column['name']
        if name in {'__source_file', '__source_row'} or (child and name == '__parent_record' and column['strategy'] != 'link'):
            raise XmlCollectionExportError(f'Column conflicts with a reserved source identifier: {name}')
        identity = (column['strategy'], column['branch'], tuple(column['field_ids']),
                    tuple(column['element_paths']))
        if name in headers and headers[name] != identity:
            raise XmlCollectionExportError(f'Column {name!r} has conflicting meanings across files')
        if identity in seen and seen[identity] != name:
            raise XmlCollectionExportError(f'Column identity changes names: {name!r}')
        headers.setdefault(name, identity)
        seen[identity] = name


def _write_row(writer: csv.DictWriter, values: Mapping[str, Any]) -> None:
    writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list, tuple)) else v
                     for k, v in values.items()})


def _child_path(target: Path, name: str) -> Path:
    slug = re.sub('[^A-Za-z0-9._-]+', '_', name).strip('._-')
    if not slug:
        raise XmlCollectionExportError(f'Invalid child table name: {name!r}')
    return target.with_name(target.stem + '__' + slug + target.suffix)


def _materialize_xml_collection_scaled_unlocked(
    recipe: Mapping[str, Any], output: str | Path, *, confirm: bool = False,
    resume: bool = False, max_files: int = 80, max_source_bytes: int = 64 * 1024**2,
    max_source_records: int = 10000, max_child_rows_per_source: int = 50000,
    max_total_rows: int = 2_000_000, max_child_rows: int = 1_000_000,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Export with per-source checkpoints and bounded collection-wide memory.

    C7 does NOT raise the 10k extracted-row limit for one source; scaling is
    across multiple sources. Resume re-verifies the recipe, source hashes and
    completed part hashes, then skips trusted completed source extracts.
    """
    if not confirm:
        raise XmlCollectionExportError('Explicit confirmation is required')
    if min(max_files, max_source_bytes, max_source_records, max_child_rows_per_source,
           max_total_rows, max_child_rows) < 1:
        raise XmlCollectionExportError('All resource limits must be positive')
    target, state_path, workdir = _paths(output)
    manifest_path = target.with_suffix(target.suffix + '.collection.json')
    event_path = provenance_path(target)
    # Keep the C6 exporter and C7 exporter mutually exclusive.
    c6_checkpoint = target.with_suffix(target.suffix + '.collection.checkpoint.json')
    if c6_checkpoint.exists() or c6_checkpoint.is_symlink():
        raise XmlCollectionExportError('C6 recovery checkpoint already exists for this output')
    for item in (target, manifest_path, event_path, state_path, workdir):
        if item.is_symlink():
            raise XmlCollectionExportError(f'Unsafe symlink in export path: {item}')
    for item in (target, manifest_path, event_path):
        if item.exists():
            raise FileExistsError(f'Existing collection output must not be replaced: {item}')
    current, context = _verify_recipe(recipe, max_files=max_files, max_source_bytes=max_source_bytes)
    root: Path = context['root']
    membership = [s['relative_path'] for s in current['selected_sources']]
    if not resume:
        if state_path.exists() or workdir.exists():
            raise XmlCollectionExportError('Existing C7 state detected; inspect or explicitly resume it')
        target.parent.mkdir(parents=True, exist_ok=True)
        workdir.mkdir(mode=0o700)
        state = {'schema': STATE_SCHEMA, 'status': 'running', 'output': str(target),
                 'created_at': datetime.now(timezone.utc).isoformat(),
                 'recipe_sha256': _hash(current), 'source_membership': membership,
                 'completed': []}
        try:
            _atomic_json(state_path, state, exclusive=True)
        except Exception:
            workdir.rmdir()
            raise
    else:
        if not state_path.is_file() or not workdir.is_dir() or workdir.is_symlink():
            raise XmlCollectionExportError('Cannot resume without a regular C7 state file and parts directory')
        state = json.loads(state_path.read_text(encoding='utf-8'))
        if (state.get('schema') != STATE_SCHEMA or state.get('output') != str(target) or
            state.get('recipe_sha256') != _hash(current) or state.get('source_membership') != membership):
            raise XmlCollectionExportError('Resume state does not match the current recipe and sources')
        completed = state.get('completed')
        if not isinstance(completed, list) or len(completed) > len(membership):
            raise XmlCollectionExportError('Malformed completed-source checkpoint')
        for i, item in enumerate(completed):
            if item.get('relative_path') != membership[i] or item.get('sha256') != current['selected_sources'][i]['source_fingerprint'].removeprefix('sha256:'):
                raise XmlCollectionExportError('Completed-source sequence changed')
            _load_part(workdir, i, item)
        state['status'] = 'running'
        _atomic_json(state_path, state)
    _emit(progress, state, 'started' if not resume else 'resumed')

    try:
        for i in range(len(state['completed']), len(membership)):
            source = current['selected_sources'][i]
            path = root / source['relative_path']
            part = _extract_part(path, source, current, max_source_bytes=max_source_bytes,
                                 max_source_records=max_source_records,
                                 max_child_rows_per_source=max_child_rows_per_source)
            part_path = _part_file(workdir, i)
            if part_path.is_symlink():
                raise XmlCollectionExportError('Unsafe source part symlink')
            _file_json(part_path, part)
            state['completed'].append({'relative_path': source['relative_path'],
                                       'sha256': part['sha256'], 'part_sha256': _sha(part_path)})
            _atomic_json(state_path, state)
            _emit(progress, state, 'source_completed')

        main_headers: OrderedDict[str, tuple[Any, ...]] = OrderedDict()
        main_seen: dict[tuple[Any, ...], str] = {}
        children: OrderedDict[str, dict[str, Any]] = OrderedDict()
        sources: list[dict[str, Any]] = []
        warnings: list[str] = []
        n_main = 0
        for i, entry in enumerate(state['completed']):
            part = _load_part(workdir, i, entry)
            _register(part['main']['columns'], main_headers, main_seen)
            n_main += len(part['main']['rows'])
            if n_main > max_total_rows:
                raise XmlCollectionExportError('Consolidated main output exceeds maximum rows')
            for table in part['children']:
                ch = children.setdefault(table['name'], {'headers': OrderedDict(), 'seen': {}, 'row_count': 0})
                _register(table['columns'], ch['headers'], ch['seen'], child=True)
                ch['row_count'] += len(table['rows'])
                if ch['row_count'] > max_child_rows:
                    raise XmlCollectionExportError(f'Child table {table["name"]!r} exceeds row limit')
            sources.append(part['source_report'])
            warnings.extend(part['warnings'])
        if n_main < 1:
            raise XmlCollectionExportError('No extracted main rows; nothing published')
        child_paths = {name: _child_path(target, name) for name in children}
        destinations = [target, *child_paths.values(), event_path, manifest_path]
        if len(set(destinations)) != len(destinations):
            raise XmlCollectionExportError('Child output filenames collide')
        for path in destinations:
            if path.exists() or path.is_symlink():
                raise FileExistsError(f'Output path already exists: {path}')
        delimiter = '\t' if target.suffix.lower() == '.tsv' else ','
        # Finished row fragments remain on disk; only one source is in memory.
        with tempfile.TemporaryDirectory(prefix='.xml_c7_publish_', dir=target.parent) as stage_name:
            stage = Path(stage_name)
            staged = {destination: stage / f'{idx:04d}_{destination.name}'
                      for idx, destination in enumerate(destinations)}
            main_columns = ['__source_file', '__source_row', *main_headers]
            child_columns = {name: ['__source_file', *info['headers']] for name, info in children.items()}
            with staged[target].open('w', encoding='utf-8', newline='') as main_stream:
                main_writer = csv.DictWriter(main_stream, fieldnames=main_columns,
                                             delimiter=delimiter, extrasaction='raise')
                main_writer.writeheader()
                child_streams = {}
                child_writers = {}
                try:
                    for name, path in child_paths.items():
                        child_streams[name] = staged[path].open('w', encoding='utf-8', newline='')
                        child_writers[name] = csv.DictWriter(child_streams[name], fieldnames=child_columns[name],
                                                              delimiter=delimiter, extrasaction='raise')
                        child_writers[name].writeheader()
                    for i, entry in enumerate(state['completed']):
                        part = _load_part(workdir, i, entry)
                        rel = part['relative_path']
                        for index, row in enumerate(part['main']['rows'], start=1):
                            _write_row(main_writer, {'__source_file': rel, '__source_row': index, **row})
                        for table in part['children']:
                            for row in table['rows']:
                                _write_row(child_writers[table['name']], {'__source_file': rel, **row})
                finally:
                    for stream in child_streams.values():
                        stream.close()
            manifest = {
                'schema': 'ml-lab.xml-collection-materialization@1',
                'created_at': datetime.now(timezone.utc).isoformat(),
                'root_path': str(root), 'recipe': current,
                'main_dataset': {'path': str(target), 'sha256': _sha(staged[target]),
                                 'columns': main_columns, 'row_count': n_main},
                'child_datasets': [{'name': name, 'path': str(path), 'sha256': _sha(staged[path]),
                                    'columns': child_columns[name], 'row_count': children[name]['row_count']}
                                   for name, path in child_paths.items()],
                'sources': sources, 'warnings': list(dict.fromkeys(warnings)),
                'provenance': {'status': 'not_recorded'},
                'execution': {'mode': 'c7_per_source_checkpointed', 'bounded_rows_per_source': 10000,
                              'resumed': resume, 'completed_sources': len(sources)},
            }
            event = make_collection_event(manifest)
            staged[event_path].write_text(json.dumps(event, sort_keys=True, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
            manifest['provenance'] = {'status': 'recorded', 'schema': event['schema'],
                                      'path': str(event_path), 'event_id': event['event_id'],
                                      'sha256': _sha(staged[event_path])}
            staged[manifest_path].write_text(json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
            published: list[Path] = []
            try:
                for destination in destinations:
                    if destination.exists() or destination.is_symlink():
                        raise FileExistsError(f'Output appeared during publication: {destination}')
                    os.replace(staged[destination], destination)
                    published.append(destination)
            except OSError:
                for destination in published:
                    destination.unlink(missing_ok=True)
                raise
        state['status'] = 'complete'
        _emit(progress, state, 'published')
        state_path.unlink(missing_ok=True)
        shutil.rmtree(workdir)
        return manifest
    except Exception:
        # Keep verified per-source fragments for an explicit retry.
        state['status'] = 'interrupted'
        _atomic_json(state_path, state)
        _emit(progress, state, 'interrupted')
        raise


def materialize_xml_collection_scaled(
    recipe: Mapping[str, Any], output: str | Path, *, confirm: bool = False,
    resume: bool = False, max_files: int = 80, max_source_bytes: int = 64 * 1024**2,
    max_source_records: int = 10000, max_child_rows_per_source: int = 50000,
    max_total_rows: int = 2_000_000, max_child_rows: int = 1_000_000,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run one output at a time with a local advisory process lock."""
    if not confirm:
        raise XmlCollectionExportError('Explicit confirmation is required')
    target, _, _ = _paths(output)
    with _output_lock(target):
        return _materialize_xml_collection_scaled_unlocked(
            recipe, target, confirm=True, resume=resume, max_files=max_files,
            max_source_bytes=max_source_bytes, max_source_records=max_source_records,
            max_child_rows_per_source=max_child_rows_per_source,
            max_total_rows=max_total_rows, max_child_rows=max_child_rows, progress=progress,
        )


def regenerate_xml_collection_scaled(
    provenance: str | Path, output: str | Path, *, confirm: bool = False,
    **limits: Any,
) -> dict[str, Any]:
    """Regenerate via the checkpointed path and compare recorded output hashes."""
    if not confirm:
        raise XmlCollectionExportError('Explicit regeneration confirmation is required')
    audit = verify_collection_event(provenance, verify_sources=True, verify_outputs=False)
    if audit['status'] != 'verified':
        raise XmlCollectionExportError('Sources changed; cannot reproduce the recorded export')
    event = json.loads(Path(provenance).read_text(encoding='utf-8'))
    result = materialize_xml_collection_scaled(
        event['recipe']['snapshot'], output, confirm=True, **limits,
    )
    expected = [(x['role'], x['sha256']) for x in event['outputs']]
    actual = [('consolidated_main', result['main_dataset']['sha256'])] + [
        ('supplemental_child', x['sha256']) for x in result['child_datasets']
    ]
    return {'schema': 'ml-lab.xml-collection-regeneration@1',
            'original_event_id': event['event_id'],
            'status': 'verified_reproduction' if expected == actual else 'different_output',
            'expected': expected, 'actual': actual, 'new_manifest': result}


def describe_sqlite_handoff(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Versioned *logical* catalog metadata, not SQLite database access or DDL.

    Future ML_Lab-owned adapters may map these records into private tables.
    PAH must consume published artifacts/service APIs, not the private DB.
    """
    if manifest.get('schema') != 'ml-lab.xml-collection-materialization@1':
        raise XmlCollectionExportError('SQLite handoff requires a finalized C5/C7 manifest')
    if manifest.get('provenance', {}).get('status') != 'recorded':
        raise XmlCollectionExportError('SQLite handoff requires C6 authoritative provenance')
    event_path = manifest['provenance'].get('path')
    if not event_path or verify_collection_event(event_path)['status'] != 'verified':
        raise XmlCollectionExportError('SQLite handoff requires verified input/output lineage')
    if json.loads(Path(event_path).read_text(encoding='utf-8')).get('event_id') != manifest['provenance'].get('event_id'):
        raise XmlCollectionExportError('SQLite handoff event identity mismatch')
    return {
        'schema': HANDOFF_SCHEMA,
        'storage_owner': 'ml_lab', 'database_created': False,
        'provenance_event_id': manifest['provenance']['event_id'],
        'datasets': [{'path': d['path'], 'sha256': d['sha256'], 'row_count': d['row_count'],
                      'columns': d['columns'], 'role': role}
                     for d, role in [(manifest['main_dataset'], 'consolidated_main'),
                                     *[(child, 'supplemental_child') for child in manifest['child_datasets']]]],
        'sources': [{'relative_path': s['relative_path'], 'sha256': s['sha256'],
                     'output_row_count': s['output_row_count']} for s in manifest['sources']],
        'note': 'Future SQLite materialization is separate mandatory ML_Lab roadmap work.',
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='XML-C7: bounded checkpointed collection export')
    sub = parser.add_subparsers(dest='command', required=True)
    export = sub.add_parser('export', help='Create or resume a checkpointed collection export')
    export.add_argument('--recipe', required=True)
    export.add_argument('--output', required=True)
    export.add_argument('--confirm', action='store_true')
    export.add_argument('--resume', action='store_true')
    export.add_argument('--progress-json', action='store_true',
                        help='Write bounded job events as local stderr JSON lines')
    export.add_argument('--max-files', type=int, default=80)
    export.add_argument('--max-source-bytes', type=int, default=64 * 1024**2)
    export.add_argument('--max-source-records', type=int, default=10000)
    export.add_argument('--max-total-rows', type=int, default=2_000_000)
    export.add_argument('--max-child-rows', type=int, default=1_000_000)
    status = sub.add_parser('status', help='Inspect non-authoritative checkpoint progress')
    status.add_argument('--output', required=True)
    regen = sub.add_parser('regenerate', help='Regenerate a C7 export from C6 lineage')
    regen.add_argument('--provenance', required=True)
    regen.add_argument('--output', required=True)
    regen.add_argument('--confirm', action='store_true')
    handoff = sub.add_parser('sqlite-handoff', help='Describe a verified logical SQLite handoff, without writing a DB')
    handoff.add_argument('--manifest', required=True)
    args = parser.parse_args(argv)
    if args.command == 'status':
        result = inspect_xml_collection_scaled(args.output)
    elif args.command == 'sqlite-handoff':
        data = json.loads(Path(args.manifest).expanduser().read_text(encoding='utf-8'))
        result = describe_sqlite_handoff(data)
    elif args.command == 'regenerate':
        result = regenerate_xml_collection_scaled(args.provenance, args.output, confirm=args.confirm)
    else:
        recipe = json.loads(Path(args.recipe).expanduser().read_text(encoding='utf-8'))
        progress = (lambda event: print(json.dumps(event, sort_keys=True), file=sys.stderr, flush=True)) if args.progress_json else None
        result = materialize_xml_collection_scaled(
            recipe, args.output, confirm=args.confirm, resume=args.resume,
            max_files=args.max_files, max_source_bytes=args.max_source_bytes,
            max_source_records=args.max_source_records, max_total_rows=args.max_total_rows,
            max_child_rows=args.max_child_rows, progress=progress,
        )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if result.get('status') not in {'different_output', 'interrupted'} else 2


if __name__ == '__main__':
    raise SystemExit(main())
