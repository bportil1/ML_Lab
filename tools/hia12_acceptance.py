#!/usr/bin/env python3
"""Offline HIA-01–12 acceptance runner for the standalone ML_Lab repository.

Runs only local Python/Node executables and files. It never downloads assets,
contacts a service, modifies another repository, or starts PAH instances.
Optional --browser uses a *locally installed* Chromium and the actual sibling
PAH_Shared graph engine in an isolated page (not the live PAH host).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(label: str, command: list[str], env: dict[str, str]) -> bool:
    print(f'\n[HIA-12] {label}', flush=True)
    try:
        return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode == 0
    except FileNotFoundError as exc:
        print(f'[HIA-12] Required local tool unavailable: {exc.filename}', file=sys.stderr)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--browser', action='store_true', help='Run opt-in local Chromium acceptance')
    parser.add_argument('--shared-viewer', type=Path, default=None,
                        help='Standalone PAH_Shared interactive_graph_view.mjs path')
    parser.add_argument('--chromium', type=str, default=None,
                        help='Local Chromium executable (default: Playwright discovery)')
    args = parser.parse_args()
    env = dict(os.environ)
    src = str(ROOT / 'src')
    env['PYTHONPATH'] = src + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    # Avoid accidental on-demand Node tool installation: only Node built-in tests.
    py_tests = sorted(str(path.relative_to(ROOT)) for path in (ROOT / 'tests').glob('test_hia*.py'))
    py_tests += sorted(str(path.relative_to(ROOT)) for path in (ROOT / 'tests').glob('test_su*.py'))
    py_tests += ['tests/test_graph_asset_cache_revision.py']
    js_tests = ['tests/test_hia11_scientific_visuals.mjs']
    scripts = sorted((ROOT / 'src/ml_lab/ui/static').glob('ml_lab_hia*.js'))
    scripts += [ROOT / 'src/ml_lab/ui/static/ml_lab_hia_visuals.mjs']
    success = run('HIA/SU Python regression and acceptance tests',
                  [sys.executable, '-m', 'pytest', '-q', *py_tests], env)
    success = run('Scientific overlay JavaScript tests', ['node', '--test', *js_tests], env) and success
    for path in scripts:
        success = run(f'JS syntax: {path.name}', ['node', '--check', str(path)], env) and success
    if args.browser:
        executable = [sys.executable, 'tests/hia12_local_browser_acceptance.py']
        if args.shared_viewer is not None:
            executable.extend(['--shared-viewer', str(args.shared_viewer.resolve())])
        if args.chromium is not None:
            executable.extend(['--chromium', args.chromium])
        success = run('Local Chromium with actual PAH_Shared engine', executable, env) and success
    print('\n[HIA-12] ' + ('PASS' if success else 'FAILED (see result above)'), flush=True)
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
