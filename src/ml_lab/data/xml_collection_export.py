"""XML-C5: local multi-source tabular extraction from an explicitly reviewed C4 recipe.

C5 writes a consolidated CSV/TSV, optional child tables, and a descriptive run
manifest. The manifest is NOT a formal provenance event (reserved for C6).
The existing single-source XML extraction API is left unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections import OrderedDict
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .xml_collection import inventory_xml_collection
from .xml_collection_recipe import build_xml_collection_recipe
from .xml_ingest import inspect_xml_structure
from .xml_preview import preview_xml_tabularization
from .xml_similarity import compare_xml_collection
from .xml_collection_provenance import (
    checkpoint_path, make_collection_event, provenance_path, start_checkpoint,
)


class XmlCollectionExportError(ValueError):
    """A recipe, source, or extracted value is unsafe for consolidated export."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _verify_recipe(recipe: Mapping[str, Any], *, max_files: int, max_source_bytes: int) -> tuple[dict[str, Any], dict[str, Any]]:
    if recipe.get('schema') != 'ml-lab.xml-collection-recipe@1' or not recipe.get('structural_recipe_validated') or recipe.get('status') != 'ready_for_collection_preview':
        raise XmlCollectionExportError('C5 requires a C4 validated and ready collection recipe')
    sources = recipe.get('selected_sources')
    if not isinstance(sources, list) or not 2 <= len(sources) <= max_files:
        raise XmlCollectionExportError('The collection must contain 2 or more reviewed sources within the file limit')
    if any(not isinstance(item, dict) for item in sources):
        raise XmlCollectionExportError('Invalid C4 source evidence')
    rels = [item.get('relative_path') for item in sources]
    if any(not isinstance(rel, str) or not rel for rel in rels) or len(set(rels)) != len(rels):
        raise XmlCollectionExportError('Invalid or duplicate source identities')
    root = Path(str(recipe.get('root_path', ''))).expanduser()
    if not root.is_dir() or root.is_symlink() or not root.is_absolute():
        raise XmlCollectionExportError('C4 root directory is unavailable or untrusted')
    root = root.resolve(strict=True)
    for rel, evidence in zip(rels, sources):
        relative = Path(rel)
        if relative.is_absolute() or '..' in relative.parts or relative.as_posix() != rel:
            raise XmlCollectionExportError(f'Invalid source path: {rel}')
        path = root / relative
        if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode) or not path.resolve(strict=True).is_relative_to(root):
            raise XmlCollectionExportError(f'Non-regular or redirected XML source: {rel}')
        if path.stat().st_size > max_source_bytes:
            raise XmlCollectionExportError(f'XML source exceeds C5 size limit: {rel}')
        if _sha(path) != str(evidence.get('source_fingerprint', '')).removeprefix('sha256:'):
            raise XmlCollectionExportError(f'XML source changed since C4 validation: {rel}')
    # Reconstruct the same reviewed semantics; do not trust JSON readiness flags alone.
    inventory = inventory_xml_collection(root, recursive=True, include_hidden=True,
                                         max_files=max(1000, max_files),
                                         max_file_bytes=max_source_bytes)
    report = compare_xml_collection(inventory)
    try:
        current = build_xml_collection_recipe(
            inventory, report, rels, str(recipe.get('record_root_canonical_path', '')),
            recipe.get('selected_field_ids', ()),
            acknowledge_variants=bool(recipe.get('acknowledge_variants')),
            allow_missing_fields=bool(recipe.get('allow_missing_fields')),
            rules=recipe.get('rules', ()), max_source_bytes=max_source_bytes,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise XmlCollectionExportError(f'C4 revalidation failed: {exc}') from exc
    if current['status'] != 'ready_for_collection_preview':
        raise XmlCollectionExportError('C4 recipe is no longer valid: ' + '; '.join(current['blocking_reasons']))
    if current['selected_sources'] != sources:
        raise XmlCollectionExportError('The reviewed source evidence changed; repeat C4 validation')
    return current, {'root': root, 'sources': sources}


def _rules_for(source: Mapping[str, Any], shared_rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    active = set(source['active_repeated_branches'])
    return [rule for rule in shared_rules if rule['branch_canonical_path'] in active]


def _extract(recipe: Mapping[str, Any], *, preview: bool, max_files: int,
             max_source_bytes: int, max_source_records: int,
             max_total_rows: int, max_child_rows: int) -> dict[str, Any]:
    current, context = _verify_recipe(recipe, max_files=max_files, max_source_bytes=max_source_bytes)
    root: Path = context['root']
    headers: OrderedDict[str, tuple[Any, ...]] = OrderedDict()
    identities: dict[tuple[Any, ...], str] = {}
    main_rows: list[dict[str, Any]] = []
    child: OrderedDict[str, dict[str, Any]] = OrderedDict()
    source_reports: list[dict[str, Any]] = []
    warnings: list[str] = []
    limit = 25 if preview else None
    for entry in current['selected_sources']:
        rel = entry['relative_path']
        path = root / rel
        artifact = inspect_xml_structure(path, max_source_bytes=max_source_bytes)
        if artifact.source_fingerprint != entry['source_fingerprint']:
            raise XmlCollectionExportError(f'Source changed during export: {rel}')
        if entry['record_root_occurrences'] > max_source_records:
            raise XmlCollectionExportError(f'{rel}: too many record-root occurrences for C5 in-memory execution ({max_source_records})')
        selected = entry['present_field_ids']
        extracted = preview_xml_tabularization(
            path, artifact, current['record_root_canonical_path'], selected,
            rules=_rules_for(entry, current['rules']), max_rows=limit,
            max_child_rows=100 if preview else None,
            max_output_rows=1000 if preview else 10000,
        )
        if not preview and (extracted.main_table is None or extracted.main_table.truncated or any(t.truncated for t in extracted.child_tables)):
            raise XmlCollectionExportError(f'{rel}: output exceeded C5 limits or was truncated; nothing published')
        if extracted.main_table is None:
            raise XmlCollectionExportError(f'{rel}: no tabular main result')
        # A single output column name must represent the SAME field, strategy and
        # branch in all sources. Different named pivot columns may be aligned.
        def register(table_headers: OrderedDict[str, tuple[Any, ...]], seen: dict[tuple[Any, ...], str], column: Any, *, table: str) -> None:
            name = column.name
            if name in {'__source_file', '__source_row'} or (table != 'main' and name == '__parent_record' and column.strategy != 'link'):
                raise XmlCollectionExportError(f'XML column collides with reserved provenance column: {name}')
            identity = (column.strategy, column.repeated_branch_canonical_path,
                        tuple(column.source_field_ids), tuple(column.source_element_canonical_paths))
            if name in table_headers and table_headers[name] != identity:
                raise XmlCollectionExportError(f'Column {name!r} has conflicting meanings across XML sources; revise recipe')
            if identity in seen and seen[identity] != name:
                raise XmlCollectionExportError(f'Column identity maps to different names ({seen[identity]!r}, {name!r}); revise recipe')
            table_headers.setdefault(name, identity)
            seen[identity] = name
        for column in extracted.main_table.columns:
            register(headers, identities, column, table='main')
        for row_idx, row in enumerate(extracted.main_table.rows, start=1):
            main_rows.append({'__source_file': rel, '__source_row': row_idx, **row})
            if len(main_rows) > max_total_rows:
                raise XmlCollectionExportError(f'Main output exceeds C5 row limit ({max_total_rows})')
        for table in extracted.child_tables:
            section = child.setdefault(table.name, {'headers': OrderedDict(), 'identities': {}, 'rows': []})
            for column in table.columns:
                register(section['headers'], section['identities'], column, table=table.name)
            for row in table.rows:
                section['rows'].append({'__source_file': rel, **row})
                if len(section['rows']) > max_child_rows:
                    raise XmlCollectionExportError(f'Child table {table.name!r} exceeds C5 row limit ({max_child_rows})')
        if _sha(path) != artifact.source_sha256:
            raise XmlCollectionExportError(f'XML source changed during extraction: {rel}')
        source_reports.append({'relative_path': rel, 'sha256': artifact.source_sha256,
                               'root_record_count': entry['record_root_occurrences'],
                               'output_row_count': len(extracted.main_table.rows),
                               'child_row_counts': {t.name: len(t.rows) for t in extracted.child_tables},
                               'preview_signature': extracted.preview_signature,
                               'missing_field_ids': entry['missing_field_ids']})
        warnings.extend(f'{rel}: {w}' for w in extracted.warnings)
    if not main_rows:
        raise XmlCollectionExportError('No main table rows were extracted; nothing published')
    return {
        'schema': 'ml-lab.xml-collection-extraction@1',
        'main_columns': ['__source_file', '__source_row', *headers],
        'main_rows': main_rows,
        'children': {name: {'columns': ['__source_file', *part['headers']], 'rows': part['rows']}
                     for name, part in child.items()},
        'sources': source_reports, 'warnings': list(dict.fromkeys(warnings)),
        'root_path': str(root), 'recipe': current,
        'is_preview': preview,
    }


def preview_xml_collection_export(recipe: Mapping[str, Any], *, max_files: int = 80,
                                  max_source_bytes: int = 64 * 1024**2) -> dict[str, Any]:
    """Revalidate C4 and return a bounded, non-persisted view of the combined table."""
    result = _extract(recipe, preview=True, max_files=max_files,
                      max_source_bytes=max_source_bytes, max_source_records=10000,
                      max_total_rows=1000, max_child_rows=1000)
    return {'schema': 'ml-lab.xml-collection-export-preview@1',
            'columns': result['main_columns'], 'rows': result['main_rows'][:50],
            'sample_row_count': len(result['main_rows']),
            'child_tables': {name: {'columns': child['columns'], 'sample_rows': child['rows'][:10]}
                             for name, child in result['children'].items()},
            'sources': result['sources'], 'warnings': result['warnings'],
            'truncated_by_design': True, 'no_files_written': True}


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]], delimiter: str) -> None:
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter=delimiter, extrasaction='raise')
        writer.writeheader()
        for row in rows:
            writer.writerow({name: (json.dumps(value, sort_keys=True, ensure_ascii=False)
                                     if isinstance(value, (dict, list, tuple)) else value)
                             for name, value in row.items()})


def materialize_xml_collection(recipe: Mapping[str, Any], output: str | Path, *,
                               overwrite: bool = False, confirm: bool = False,
                               max_files: int = 80, max_source_bytes: int = 64 * 1024**2,
                               max_source_records: int = 10000,
                               max_total_rows: int = 100000,
                               max_child_rows: int = 50000,
                               _resume_checkpoint: bool = False) -> dict[str, Any]:
    """Write reviewed sources as one CSV/TSV, optional children, and a C5 manifest.

    All tables are staged first. Existing targets are backed up for rollback when
    overwrite=True. The manifest is published last. C6 will add formal provenance.
    """
    if not confirm:
        raise XmlCollectionExportError('Explicit confirmation is required for consolidated export')
    target = Path(output).expanduser().absolute()
    if target.suffix.lower() not in {'.csv', '.tsv'}:
        raise XmlCollectionExportError('Consolidated output must use .csv or .tsv')
    if target.is_symlink():
        raise XmlCollectionExportError('Refusing to write through an output symlink')
    if not overwrite:
        for initial in (target, target.with_suffix(target.suffix + '.collection.json'), provenance_path(target)):
            if initial.exists():
                raise FileExistsError(f'Output exists (or is not a file): {initial}')
    # Create a durable replay intent *before* the potentially long extraction.
    # Never mistake this checkpoint for authoritative lineage or a completed CSV.
    journal = checkpoint_path(target)
    if journal.is_symlink():
        raise XmlCollectionExportError(f'Recovery checkpoint is a symlink: {journal}')
    if _resume_checkpoint and not journal.is_file():
        raise XmlCollectionExportError('Cannot resume without the existing C6 checkpoint')
    if not _resume_checkpoint:
        start_checkpoint(recipe, target)
    result = _extract(recipe, preview=False, max_files=max_files,
                      max_source_bytes=max_source_bytes, max_source_records=max_source_records,
                      max_total_rows=max_total_rows, max_child_rows=max_child_rows)
    delimiter = '\t' if target.suffix.lower() == '.tsv' else ','
    # Keep deterministic, conflict-free child path names.
    def child_target(name: str) -> Path:
        import re
        slug = re.sub('[^A-Za-z0-9._-]+', '_', name).strip('._-')
        if not slug:
            raise XmlCollectionExportError(f'Invalid child table name: {name!r}')
        return target.with_name(target.stem + '__' + slug + target.suffix)
    children = {name: child_target(name) for name in result['children']}
    if len(set(children.values())) != len(children):
        raise XmlCollectionExportError('Child table filenames collide; rename the extraction rules')
    manifest_path = target.with_suffix(target.suffix + '.collection.json')
    event_path = provenance_path(target)
    expected = [target, *children.values(), event_path, manifest_path]
    if len(set(expected)) != len(expected):
        raise XmlCollectionExportError('Consolidated output paths collide')
    for path in expected:
        if path.is_symlink():
            raise XmlCollectionExportError(f'Output artifact is a symlink: {path}')
        if path.exists() and (not path.is_file() or not overwrite):
            raise FileExistsError(f'Output exists (or is not a file): {path}')
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.xml_collection_', dir=target.parent) as temp:
        stage = Path(temp)
        staged: dict[Path, Path] = {}
        def staged_path(original: Path) -> Path:
            return stage / f'{len(staged):04d}_{original.name}'
        for name, path in [(None, target), *children.items()]:
            temporary = staged_path(path)
            table = {'columns': result['main_columns'], 'rows': result['main_rows']} if name is None else result['children'][name]
            _write_csv(temporary, table['columns'], table['rows'], delimiter)
            staged[path] = temporary
        manifest: dict[str, Any] = {
            'schema': 'ml-lab.xml-collection-materialization@1',
            'created_at': datetime.now(timezone.utc).isoformat(),
            'root_path': result['root_path'],
            'recipe': result['recipe'],
            'main_dataset': {'path': str(target), 'sha256': _sha(staged[target]),
                             'columns': result['main_columns'], 'row_count': len(result['main_rows'])},
            'child_datasets': [{'name': name, 'path': str(path),
                                'sha256': _sha(staged[path]), 'columns': result['children'][name]['columns'],
                                'row_count': len(result['children'][name]['rows'])}
                               for name, path in children.items()],
            'sources': result['sources'], 'warnings': result['warnings'],
            'provenance': {'status': 'not_recorded',
                           'note': 'C5 manifest is a descriptive export record, not authoritative C6 lineage.'},
        }
        # C6 authoritative provenance is prepared from verified staged bytes,
        # not from optimistic source paths or an incompletely published dataset.
        event = make_collection_event(manifest)
        event_temp = staged_path(event_path)
        event_temp.write_text(json.dumps(event, indent=2, sort_keys=True, ensure_ascii=False) + '\n', encoding='utf-8')
        staged[event_path] = event_temp
        manifest['provenance'] = {
            'status': 'recorded', 'schema': event['schema'],
            'path': str(event_path), 'event_id': event['event_id'],
            'sha256': _sha(event_temp),
        }
        manifest_temp = staged_path(manifest_path)
        manifest_temp.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + '\n', encoding='utf-8')
        staged[manifest_path] = manifest_temp
        backups: dict[Path, Path] = {}
        published: list[Path] = []
        try:
            for destination in staged:
                if destination.exists():
                    backup = stage / ('backup_' + str(len(backups)))
                    os.replace(destination, backup)
                    backups[destination] = backup
            for destination, staged_file in staged.items():
                os.replace(staged_file, destination)
                published.append(destination)
        except OSError:
            for destination in published:
                destination.unlink(missing_ok=True)
            for destination, backup in backups.items():
                os.replace(backup, destination)
            raise
        # The manifest is committed last. Only then retire the non-authoritative
        # replay intent; an interrupted run retains a recoverable checkpoint.
        journal.unlink(missing_ok=True)
        return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='XML-C5: consolidated XML collection CSV/TSV export')
    parser.add_argument('--recipe', required=True, help='Validated C4 JSON recipe path')
    parser.add_argument('--output', default='analysis/data/xml_collection.csv')
    parser.add_argument('--preview', action='store_true', help='Inspect sampled rows; do not write files')
    parser.add_argument('--confirm', action='store_true', help='Explicitly approve writing the combined output')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    recipe = json.loads(Path(args.recipe).expanduser().read_text(encoding='utf-8'))
    if args.preview:
        print(json.dumps(preview_xml_collection_export(recipe), indent=2, ensure_ascii=False))
    else:
        print(json.dumps(materialize_xml_collection(recipe, args.output, confirm=args.confirm,
                                                    overwrite=args.overwrite), indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
