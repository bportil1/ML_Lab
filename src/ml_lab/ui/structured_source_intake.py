"""Explicit, project-scoped single-file sources for Structured Uncertainty.

Only reads from the configured project root; no uploads, remote access,
background conversion, SQLite writes, or inferred probabilistic edges.
"""
from __future__ import annotations

import csv
import io
import os
from itertools import islice
from pathlib import Path

from ml_lab.structured import CONTRACT_SCHEMA, StructuredContractError, StructuredObject

SUPPORTED_SUFFIXES = frozenset({'.json', '.xml', '.csv', '.tsv'})


def project_source(root, path):
    """Resolve and normalize project-local paths; reject every symlink component."""
    root = Path(root).resolve()
    if not isinstance(path, str) or not path.strip() or len(path) > 400:
        raise StructuredContractError('Choose an existing file from this project.')
    submitted = Path(path.strip()).expanduser()
    try:
        relative = submitted.relative_to(root) if submitted.is_absolute() else submitted
    except ValueError as exc:
        raise StructuredContractError('File must be inside the configured ML_Lab project directory.') from exc
    if not relative.parts or any(part in ('..', '.') for part in relative.parts):
        raise StructuredContractError('source must be project-relative or a local path inside the project directory.')
    if relative.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise StructuredContractError('Supported single-file sources: .json, .xml, .csv, .tsv.')
    for depth in range(1, len(relative.parts) + 1):
        if root.joinpath(*relative.parts[:depth]).is_symlink():
            raise StructuredContractError('symlink source paths are not allowed.')
    selected = root / relative
    if not selected.resolve().is_relative_to(root) or not selected.is_file():
        raise StructuredContractError('Source is not a readable file in this project.')
    return selected, relative.as_posix()


def browse_project_files(root, *, max_files=120, max_directories=150, max_entries=4000, max_depth=7):
    """Bounded, on-demand project inventory; never descend into symlinked folders."""
    root = Path(root).resolve()
    exclude = {'.git', '.venv', 'venv', '__pycache__', 'node_modules', '.mypy_cache',
               '.pytest_cache', '.tox', 'dist', 'build'}
    result, dirs_seen, entries_seen = [], 0, 0
    queue = [(root, 0)]
    while queue and len(result) < max_files and dirs_seen < max_directories and entries_seen < max_entries:
        directory, depth = queue.pop(0)
        dirs_seen += 1
        try:
            # Stop scanning each directory as well as the total tree: a large
            # generated-output folder must not lock the interactive UI.
            with os.scandir(directory) as scan:
                entries = sorted(islice(scan, min(800, max_entries - entries_seen)),
                                 key=lambda item: item.name.casefold())
        except OSError:
            continue
        for entry in entries:
            entries_seen += 1
            if entries_seen > max_entries:
                break
            if entry.is_symlink():
                continue
            try:
                if entry.is_file(follow_symlinks=False) and Path(entry.name).suffix.lower() in SUPPORTED_SUFFIXES and len(str(Path(entry.path).relative_to(root))) <= 400:
                    result.append(Path(entry.path).relative_to(root).as_posix())
                    if len(result) >= max_files:
                        break
                elif entry.is_dir(follow_symlinks=False) and depth < max_depth and entry.name not in exclude and not entry.name.startswith('.'):
                    queue.append((Path(entry.path), depth + 1))
            except OSError:
                continue
    return tuple(result)


def delimited_to_structured(raw: bytes, *, relative_path: str, sha256: str, delimiter: str):
    """A table has explicit column nodes and aligned *record* rows.

    Preserve CSV cells as strings (including numeric-looking identifiers); never
    infer sampling independence, numeric types, missingness or source dependencies.
    """
    if b'\x00' in raw:
        raise StructuredContractError('Delimited source contains NUL bytes.')
    try:
        data = raw.decode('utf-8-sig')
        reader = csv.reader(io.StringIO(data, newline=''), delimiter=delimiter, strict=True)
        headers = next(reader, None)
        if not headers or any(not name.strip() for name in headers) or len(headers) != len(set(headers)):
            raise StructuredContractError('CSV/TSV requires a nonempty, unique header for every column.')
        nodes = [{'id': 'table', 'kind': 'table', 'label': Path(relative_path).name,
                  'attributes': {'source_type': 'delimited_table'}}]
        edges = []
        for index, name in enumerate(headers):
            node_id = f'column:{index}'
            nodes.append({'id': node_id, 'kind': 'column', 'label': name,
                          'attributes': {'column': name, 'column_index': index}})
            edges.append({'id': f'edge:{index}', 'kind': 'has_column',
                          'source_id': 'table', 'target_id': node_id, 'ordinal': index})
        observations = []
        for i, row in enumerate(reader, start=1):
            if len(row) != len(headers):
                raise StructuredContractError(f'CSV/TSV line {reader.line_num}: expected {len(headers)} columns; got {len(row)}.')
            observations.append({'observation_id': f'row:{i}',
                                 'values': {name: {'status': 'empty' if value == '' else 'present',
                                                   'value': value}
                                            for name, value in zip(headers, row)},
                                 'source_ref': {'line': reader.line_num}})
    except (UnicodeError, csv.Error) as exc:
        raise StructuredContractError(f'Invalid UTF-8 CSV/TSV: {exc}') from exc
    obj = StructuredObject.from_dict({
        'schema': CONTRACT_SCHEMA, 'topology': {'kind': 'tree'},
        'nodes': nodes, 'edges': edges, 'observation_unit': 'record',
        'observations': observations,
        'provenance': {'adapter': 'delimited-file@1', 'source_path': relative_path,
                       'source_sha256': sha256,
                       'column_policy': 'header_names_exact',
                       'cell_policy': 'strings; empty is not automatically missing',
                       'sampling_independence': 'unasserted'},
    })
    return obj, [
        'CSV/TSV values were preserved as strings, not inferred as numeric types.',
        'Empty CSV/TSV fields remain empty cells; sampling independence is not asserted.',
        'The table-to-column edges describe layout, not statistical dependency.',
    ], []
