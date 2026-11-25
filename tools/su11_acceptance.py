#!/usr/bin/env python3
"""SU-11 local acceptance of explicitly supplied, project-owned exports.

No upload, network access, background service, cache writes, database reads,
or invented statistical samples. Run from an ML_Lab checkout with PYTHONPATH=src.
JSON artifacts use the same read_source guard as the real Structured UI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any

from ml_lab.data.xml_ingest import inspect_xml_structure
from ml_lab.structured import (ObservationBindingPlan, StructuredContractError,
                               source_sha256)
from ml_lab.structured.adapters import XMLStructureAdapter
from ml_lab.ui.structured_workbench import KINDS, read_source
from ml_lab.ui.structured_graph_view import make_shared_graph_view
from ml_lab.ui.structured_sandbox import analyze_sandbox

SCHEMA = 'ml-lab.su11-local-acceptance@1'


def _source_path(root: Path, value: str, *, suffix: str) -> Path:
    """Do not permit traversal or symlinks for user-supplied artifact paths."""
    relative = Path(value)
    if (not value or len(value) > 400 or relative.is_absolute() or
            '..' in relative.parts or relative.suffix.lower() != suffix):
        raise StructuredContractError('expected project-relative ' + suffix + ' path')
    for n in range(1, len(relative.parts) + 1):
        if root.joinpath(*relative.parts[:n]).is_symlink():
            raise StructuredContractError('source symlinks not permitted')
    path = root / relative
    if not path.resolve().is_relative_to(root) or not path.is_file():
        raise StructuredContractError('source must exist within project')
    return path


def _viewer_check(view: dict[str, Any], module: Path | None) -> dict[str, Any]:
    if module is None:
        return {'status': 'skipped', 'reason': 'viewer_module_not_supplied'}
    if not module.is_file() or module.suffix != '.mjs':
        return {'status': 'failed', 'reason': 'viewer module missing or not an .mjs file'}
    # This is a real, local PAH_Shared module import, not a local reimplementation.
    code = '''import {validateInteractiveGraphView,interactiveGraphRows,interactiveGraphLayout} from %s;
let raw=''; for await (const chunk of process.stdin) raw += chunk;
const view=validateInteractiveGraphView(JSON.parse(raw));
const rows=interactiveGraphRows(view);
const layout=interactiveGraphLayout(view);
if(rows.nodes.length!==view.nodes.length || rows.edges.length!==view.edges.length || layout.size!==view.nodes.length)
  throw Error('shared engine does not cover all original identities');
process.stdout.write('validated');''' % json.dumps(module.resolve().as_uri())
    try:
        done = subprocess.run(['node', '--input-type=module', '-e', code],
                              input=json.dumps(view, ensure_ascii=False, allow_nan=False),
                              capture_output=True, text=True, timeout=25, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {'status': 'failed', 'reason': type(exc).__name__}
    return {'status': 'passed' if done.returncode == 0 else 'failed',
            'reason': 'real_shared_viewer_validated' if done.returncode == 0
                      else done.stderr[:300]}


def _inspect(root: Path, kind: str, relative: str, *, viewer: Path | None,
             binding_file: str | None = None, node: str | None = None) -> dict[str, Any]:
    started = perf_counter()
    report: dict[str, Any] = {'source_kind': kind, 'source_path': relative,
                              'status': 'failed', 'gates': {}}
    try:
        if kind == 'xml_raw':
            path = _source_path(root, relative, suffix='.xml')
            if path.stat().st_size > 2 * 1024 * 1024:
                raise StructuredContractError('raw XML exceeds 2 MiB acceptance limit')
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            # Existing XML inspector applies the existing safe parser limits.
            adapted = XMLStructureAdapter().adapt(inspect_xml_structure(path))
            source = adapted.object
            warnings, losses = adapted.warnings, adapted.losses
            assert hashlib.sha256(path.read_bytes()).hexdigest() == before
            revision = before
        else:
            if kind not in KINDS:
                raise StructuredContractError('unsupported adapter: ' + kind)
            source, meta = read_source(root, relative, kind)
            revision = meta['sha256']
            warnings, losses = meta['warnings'], meta['losses']
        report.update({'source_sha256': 'sha256:' + revision,
                       'structured_digest': source_sha256(source),
                       'topology': source.topology['kind'],
                       'nodes': len(source.nodes), 'edges': len(source.edges),
                       'observation_rows': len(source.observations),
                       'warnings': list(warnings), 'losses': list(losses)})
        report['gates']['scientific_source'] = {'status': 'passed',
            'note': 'nodes_are_not_statistical_observations'}
        try:
            view = make_shared_graph_view(source, fingerprint=revision)
        except StructuredContractError as exc:
            # UI size/identifier restrictions do not invalidate the source's
            # scientific representation; never claim a sampled display.
            report['gates']['interactive_graph'] = {'status': 'skipped',
                'reason': 'explicit_viewer_limit', 'detail': str(exc)}
        else:
            report['gates']['interactive_graph'] = {'status': 'passed',
                'scope': view['display']['coverage']['scope']}
            report['gates']['shared_viewer'] = _viewer_check(view, viewer)
        if binding_file is None:
            report['gates']['observation_binding'] = {'status': 'skipped',
                'reason': 'explicit_binding_not_supplied'}
        else:
            if kind == 'xml_raw':
                raise StructuredContractError('raw XML structure profile has no aligned rows to bind')
            binding_path = _source_path(root, binding_file, suffix='.json')
            if binding_path.stat().st_size > 96_000:
                raise StructuredContractError('binding file exceeds acceptance limit')
            plan = ObservationBindingPlan.from_json(binding_path.read_text('utf-8'))
            if node is None:
                raise StructuredContractError('--node is mandatory when --binding is provided')
            analysis = analyze_sandbox(source, source_revision='sha256:' + revision,
                request_data={'source_revision': 'sha256:' + revision,
                              'mode': 'inspect', 'node_id': node,
                              'binding_plan': plan.to_dict()})
            report['gates']['observation_binding'] = {'status': 'passed',
                'scope': analysis['observation_evidence']['scope'],
                'sample_rows': analysis['observation_evidence']['aligned_rows'],
                'binding_digest': analysis['binding_digest']}
        report['status'] = 'failed' if any(gate['status'] == 'failed'
            for gate in report['gates'].values()) else 'passed'
    except (OSError, ValueError, TypeError, UnicodeError, StructuredContractError) as exc:
        report['error'] = str(exc)
    finally:
        report['elapsed_seconds'] = round(perf_counter() - started, 4)
    return report


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path('.'))
    parser.add_argument('--artifact', action='append', metavar='KIND:RELATIVE_PATH',
                        required=True, help='repeat for actual XML/JSON/AST/network exports; xml_raw supports .xml')
    parser.add_argument('--viewer', type=Path,
                        help='installed local PAH_Shared interactive_graph_view.mjs; optional')
    parser.add_argument('--binding', help='project-relative explicit SU-05 binding JSON (single artifact only)')
    parser.add_argument('--node', help='exact source node ID used with --binding')
    parser.add_argument('--output', type=Path, help='explicit local JSON output path; no output by default')
    args = parser.parse_args(argv)
    if args.binding and len(args.artifact) != 1:
        parser.error('--binding requires exactly one --artifact')
    if args.node and not args.binding:
        parser.error('--node requires --binding')
    root = args.project_root.expanduser().resolve()
    if not root.is_dir():
        parser.error('project root must exist')
    outcomes = []
    for entry in args.artifact:
        kind, delim, relative = entry.partition(':')
        if not delim:
            parser.error('--artifact must be KIND:RELATIVE_PATH')
        outcomes.append(_inspect(root, kind, relative, viewer=args.viewer,
                                binding_file=args.binding, node=args.node))
    summary = {'schema': SCHEMA, 'project_root': str(root),
               'status': 'failed' if any(item['status'] == 'failed' for item in outcomes)
                         else 'passed',
               'artifacts': outcomes,
               'environment_gates': {
                   'real_pypique_export': 'requires_explicit_real_export_artifact',
                   'real_code_analyzer_export': 'requires_explicit_real_export_artifact',
                   'live_flask_pah_browser': 'not_executed_by_this_headless_runner',
                   'cross_module_regressions': 'separate_repositories_required',
               }}
    report = json.dumps(summary, indent=2, allow_nan=False, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + '\n', encoding='utf-8')
    print(report)
    return 0 if summary['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(run())
