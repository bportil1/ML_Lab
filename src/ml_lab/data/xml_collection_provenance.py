"""XML-C6: authoritative multi-source lineage and cautious local recovery.

A checkpoint is an *intent*, never a provenance event. Only fully staged,
checksum-identified exports receive an authoritative event, published in the
same rollback transaction as their datasets and C5 materialization manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROVENANCE_SCHEMA = 'ml-lab.xml-collection-provenance@1'
CHECKPOINT_SCHEMA = 'ml-lab.xml-collection-checkpoint@1'


class XmlCollectionProvenanceError(ValueError):
    """The lineage, sources, outputs, or recovery state are untrustworthy."""


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def provenance_path(output: str | Path) -> Path:
    p = Path(output).expanduser().absolute()
    return p.with_suffix(p.suffix + '.provenance.json')


def checkpoint_path(output: str | Path) -> Path:
    p = Path(output).expanduser().absolute()
    return p.with_suffix(p.suffix + '.collection.checkpoint.json')


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise XmlCollectionProvenanceError(f'Not a regular local JSON file: {path}')
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise XmlCollectionProvenanceError(f'Unreadable JSON: {path}: {exc}') from exc
    if not isinstance(value, dict):
        raise XmlCollectionProvenanceError(f'Expected a JSON object: {path}')
    return value


def _atomic_json(path: Path, document: Mapping[str, Any], *, exclusive: bool = False) -> None:
    """Write a status checkpoint locally without exposing partly written JSON."""
    if path.is_symlink():
        raise XmlCollectionProvenanceError(f'Refusing symlink checkpoint: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.xml_c6_', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(document, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            # Atomic create-if-absent: two exporters cannot acquire the same
            # recovery intent or accidentally overwrite each other.
            os.link(name, path)
        else:
            os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def make_collection_event(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Construct immutable recorded multi-parent lineage from staged checksums.

    All values originate in the C5 validated recipe and already staged files;
    a claim about an input missing from the C4 selection is never inferred.
    """
    if manifest.get('schema') != 'ml-lab.xml-collection-materialization@1':
        raise XmlCollectionProvenanceError('Cannot record provenance for unknown C5 manifest')
    recipe = manifest.get('recipe')
    sources = manifest.get('sources')
    main = manifest.get('main_dataset')
    if not isinstance(recipe, dict) or not isinstance(sources, list) or not sources or not isinstance(main, dict):
        raise XmlCollectionProvenanceError('Manifest lacks required source, output, or recipe evidence')
    reviewed = recipe.get('selected_sources') or []
    if [s.get('relative_path') for s in reviewed] != [s.get('relative_path') for s in sources]:
        raise XmlCollectionProvenanceError('Source membership differs from reviewed C4 selection')
    root = Path(str(manifest['root_path']))
    parents = []
    for reviewed_source, source in zip(reviewed, sources):
        rel = str(source['relative_path'])
        if not rel or Path(rel).is_absolute() or '..' in Path(rel).parts:
            raise XmlCollectionProvenanceError('Unsafe parent source identity')
        digest = str(source['sha256'])
        if 'sha256:' + digest != reviewed_source.get('source_fingerprint'):
            raise XmlCollectionProvenanceError('Parent source hash does not match C4 evidence')
        parents.append({
            'path': str(root / rel), 'relative_path': rel,
            'sha256': digest, 'source_fingerprint': 'sha256:' + digest,
            'root_record_count': source['root_record_count'],
            'output_row_count': source['output_row_count'],
            'child_row_counts': source['child_row_counts'],
            'missing_field_ids': source['missing_field_ids'],
        })
    outputs = [dict(main, role='consolidated_main')]
    outputs.extend(dict(child, role='supplemental_child') for child in manifest.get('child_datasets', []))
    for output in outputs:
        if not output.get('path') or not output.get('sha256') or len(str(output['sha256'])) != 64:
            raise XmlCollectionProvenanceError('A staged output lacks a path or SHA-256')
    transformation = {
        'recipe_sha256': _hash(recipe),
        'source_ids': [{'relative_path': p['relative_path'], 'sha256': p['sha256']} for p in parents],
        'output_ids': [{'role': o['role'], 'sha256': o['sha256'], 'path': o['path']} for o in outputs],
    }
    event = {
        'schema': PROVENANCE_SCHEMA,
        'created_at': manifest['created_at'],
        'authoritative': True,
        'relation': 'derived_from_many',
        'operation_kind': 'xml_collection_consolidation',
        'source_root': str(root),
        'sources': parents,
        'source_membership': [p['relative_path'] for p in parents],
        'excluded_sources': {'status': 'not_recorded', 'note': 'C4 records selected sources, not a complete inventory of excluded files'},
        'recipe': {'sha256': _hash(recipe), 'snapshot': recipe},
        'outputs': outputs,
        'manifest_path': str(Path(str(main['path'])).with_suffix(Path(str(main['path'])).suffix + '.collection.json')),
        'warnings': list(manifest.get('warnings') or []),
        'transformation_id': 'transformation:sha256:' + _hash(transformation),
    }
    event['event_id'] = 'event:sha256:' + _hash(event)
    return event


def verify_collection_event(event_or_path: Mapping[str, Any] | str | Path, *,
                            verify_sources: bool = True,
                            verify_outputs: bool = True) -> dict[str, Any]:
    """Check event integrity plus on-disk input/output fingerprints; no writes."""
    event = _read_json(Path(event_or_path)) if isinstance(event_or_path, (str, Path)) else dict(event_or_path)
    if event.get('schema') != PROVENANCE_SCHEMA or event.get('authoritative') is not True:
        raise XmlCollectionProvenanceError('Not an authoritative XML-C6 provenance event')
    expected = 'event:sha256:' + _hash({k: v for k, v in event.items() if k != 'event_id'})
    if event.get('event_id') != expected:
        raise XmlCollectionProvenanceError('Provenance event identity does not match its contents')
    recipe = event.get('recipe') or {}
    if _hash(recipe.get('snapshot')) != recipe.get('sha256'):
        raise XmlCollectionProvenanceError('Recorded recipe checksum mismatch')
    sources = event.get('sources')
    outputs = event.get('outputs')
    if not isinstance(sources, list) or not sources or not isinstance(outputs, list) or not outputs:
        raise XmlCollectionProvenanceError('Provenance has no source or output membership')
    if event.get('source_membership') != [s.get('relative_path') for s in sources]:
        raise XmlCollectionProvenanceError('Recorded source membership is inconsistent')
    if (recipe['snapshot'].get('selected_sources') or []) and (
        [s.get('relative_path') for s in recipe['snapshot']['selected_sources']] != event['source_membership']
    ):
        raise XmlCollectionProvenanceError('Sources differ from recipe selection')
    problems = []
    def check(path_string: str, digest: str, kind: str) -> None:
        path = Path(path_string)
        if path.is_symlink() or not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
            problems.append(f'{kind} missing/unsafe: {path}')
        elif _file_sha(path) != digest:
            problems.append(f'{kind} changed: {path}')
    if verify_sources:
        for source in sources:
            check(source['path'], source['sha256'], 'Source')
    if verify_outputs:
        for output in outputs:
            check(output['path'], output['sha256'], 'Output')
    return {'schema': 'ml-lab.xml-collection-verification@1',
            'event_id': event['event_id'], 'status': 'verified' if not problems else 'mismatch',
            'problems': problems, 'sources_checked': len(sources) if verify_sources else 0,
            'outputs_checked': len(outputs) if verify_outputs else 0}


def start_checkpoint(recipe: Mapping[str, Any], output: str | Path) -> Path:
    path = checkpoint_path(output)
    if path.is_symlink():
        raise XmlCollectionProvenanceError(f'Recovery checkpoint is a symlink: {path}')
    if path.exists():
        raise XmlCollectionProvenanceError(
            f'Previous XML collection checkpoint exists: {path}. Inspect and explicitly recover it first.')
    document = {
        'schema': CHECKPOINT_SCHEMA, 'status': 'pending',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'output': str(Path(output).expanduser().absolute()),
        'recipe_sha256': _hash(recipe), 'recipe': dict(recipe),
        'authoritative': False,
        'note': 'A checkpoint is not a completed export or provenance event.',
    }
    try:
        _atomic_json(path, document, exclusive=True)
    except FileExistsError as exc:
        raise XmlCollectionProvenanceError(
            f'Concurrent or incomplete export checkpoint exists: {path}') from exc
    return path


def inspect_collection_recovery(output: str | Path) -> dict[str, Any]:
    """Report whether safe replay is possible without touching existing outputs."""
    out = Path(output).expanduser().absolute()
    checkpoint = checkpoint_path(out)
    if not checkpoint.exists():
        return {'status': 'no_checkpoint', 'checkpoint': str(checkpoint)}
    payload = _read_json(checkpoint)
    if payload.get('schema') != CHECKPOINT_SCHEMA or payload.get('output') != str(out):
        raise XmlCollectionProvenanceError('Unrecognized checkpoint or output identity')
    if _hash(payload.get('recipe')) != payload.get('recipe_sha256'):
        raise XmlCollectionProvenanceError('Recovery recipe hash mismatch')
    # Never adopt partial files, and never guess whether a partial child table is valid.
    existing = [str(p) for p in (out, out.with_suffix(out.suffix + '.collection.json'),
                                   provenance_path(out)) if p.exists()]
    # A partially published child file is also an obstacle. Since C5 uses table
    # names in its manifest, audit any matching child artifact conservatively.
    existing += [str(p) for p in out.parent.glob(out.stem + '__*' + out.suffix) if p.exists()]
    result = {'schema': 'ml-lab.xml-collection-recovery@1',
              'checkpoint': str(checkpoint), 'output': str(out),
              'status': 'blocked_existing_artifacts' if existing else 'ready_to_replay',
              'existing_artifacts': sorted(set(existing)),
              'source_integrity': 'unchecked'}
    if not existing:
        from .xml_collection_export import _verify_recipe
        try:
            _verify_recipe(payload['recipe'], max_files=80, max_source_bytes=64 * 1024**2)
        except (ValueError, OSError) as exc:
            result['status'] = 'blocked_source_or_recipe'
            result['problems'] = [str(exc)]
        else:
            result['source_integrity'] = 'verified'
    return result


def resume_xml_collection_export(output: str | Path, *, confirm: bool = False) -> dict[str, Any]:
    """Replay checkpoint from the beginning; never adopt partially written files."""
    if not confirm:
        raise XmlCollectionProvenanceError('Explicit recovery confirmation required')
    report = inspect_collection_recovery(output)
    if report.get('status') != 'ready_to_replay':
        raise XmlCollectionProvenanceError(f'Recovery blocked: {report}')
    checkpoint = _read_json(checkpoint_path(output))
    from .xml_collection_export import materialize_xml_collection
    return materialize_xml_collection(checkpoint['recipe'], output, confirm=True, _resume_checkpoint=True)


def regenerate_xml_collection(provenance: str | Path, output: str | Path, *,
                              confirm: bool = False, overwrite: bool = False) -> dict[str, Any]:
    """Re-extract strictly from a recorded recipe; reject changed source content.

    Regeneration does not copy derived CSV rows. If output hashes differ, the
    new artifacts are *not* considered a verified reproduction.
    """
    if not confirm:
        raise XmlCollectionProvenanceError('Explicit regeneration confirmation required')
    event = _read_json(Path(provenance))
    audit = verify_collection_event(event, verify_outputs=False)
    if audit['status'] != 'verified':
        raise XmlCollectionProvenanceError('Cannot regenerate from changed source files: ' + '; '.join(audit['problems']))
    from .xml_collection_export import materialize_xml_collection
    new_manifest = materialize_xml_collection(event['recipe']['snapshot'], output,
                                               confirm=True, overwrite=overwrite)
    expected = [(p['role'], p['sha256']) for p in event['outputs']]
    actual = [('consolidated_main', new_manifest['main_dataset']['sha256'])] + [
        ('supplemental_child', p['sha256']) for p in new_manifest['child_datasets']]
    return {'schema': 'ml-lab.xml-collection-regeneration@1',
            'original_event_id': event['event_id'],
            'status': 'verified_reproduction' if expected == actual else 'different_output',
            'expected': expected, 'actual': actual,
            'new_manifest': new_manifest}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='XML-C6 local provenance, audit and recovery')
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('verify', help='Verify event and current source/output hashes')
    check.add_argument('--provenance', required=True)
    inspect = sub.add_parser('inspect-recovery', help='Inspect leftover checkpoint without writing')
    inspect.add_argument('--output', required=True)
    resume = sub.add_parser('resume', help='Replay a safe, complete recipe from a checkpoint')
    resume.add_argument('--output', required=True)
    resume.add_argument('--confirm', action='store_true')
    regen = sub.add_parser('regenerate', help='Regenerate into an output path from provenance')
    regen.add_argument('--provenance', required=True)
    regen.add_argument('--output', required=True)
    regen.add_argument('--confirm', action='store_true')
    regen.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'verify':
        data = verify_collection_event(args.provenance)
    elif args.command == 'inspect-recovery':
        data = inspect_collection_recovery(args.output)
    elif args.command == 'resume':
        data = resume_xml_collection_export(args.output, confirm=args.confirm)
    else:
        data = regenerate_xml_collection(args.provenance, args.output,
                                         confirm=args.confirm, overwrite=args.overwrite)
    print(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if data.get('status') not in {'mismatch', 'different_output',
                                         'blocked_source_or_recipe', 'blocked_existing_artifacts'} else 2


if __name__ == '__main__':
    raise SystemExit(main())
