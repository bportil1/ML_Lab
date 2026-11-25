"""Graph asset URL must change with the actual installed PAH_Shared module.

A pip/PAH setup may refresh a dependency without changing its package version.
The browser's ES module map is URL keyed; relying on max-age=3600 for a static
module URL leaves the outdated 5,000-node validator in use until expiry.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.resources
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / 'src/ml_lab/ui'
ROUTES = (UI / 'workbench_structured_routes.py').read_text(encoding='utf-8')
BLUEPRINT = (UI / 'blueprint.py').read_text(encoding='utf-8')
TEMPLATE = (UI / 'templates/ml_lab_ui/workbench_structured.html').read_text(encoding='utf-8')


def _isolated_function(source: str, name: str, *, file: Path):
    """Exercise the pure helpers without needing optional Flask dependencies."""
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {'__file__': str(file), 'hashlib': hashlib, 'Path': Path}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(file), 'exec'), namespace)
    return namespace[name]


def test_graph_asset_url_revisions_follow_installed_bytes(monkeypatch, tmp_path):
    assets = tmp_path / 'static'
    assets.mkdir()
    graph = assets / 'interactive_graph_view.mjs'
    graph.write_bytes(b'old graph contract')
    monkeypatch.setattr(importlib.resources, 'files', lambda package: tmp_path)
    revision = _isolated_function(ROUTES, '_shared_graph_asset_revision',
                                  file=UI / 'workbench_structured_routes.py')
    first = revision()
    assert first == hashlib.sha256(b'old graph contract').hexdigest()[:20]
    graph.write_bytes(b'new graph contract, no 5000 maximum')
    assert revision() == hashlib.sha256(graph.read_bytes()).hexdigest()[:20]
    assert revision() != first
    graph.unlink()
    assert revision() is None


def test_controller_revision_follows_local_scripts(tmp_path):
    location = tmp_path / 'ui'
    location.mkdir()
    assets = location / 'static'
    assets.mkdir()
    for name in ('ml_lab_structured_graph.js', 'ml_lab_hia_workspace.js',
                 'ml_lab_hia_atomic.js', 'ml_lab_hia_neighbors.js',
                 'ml_lab_hia_branch.js', 'ml_lab_hia_level.js', 'ml_lab_hia_conditional.js',
                 'ml_lab_hia_factorization.js', 'ml_lab_hia_network.js', 'ml_lab_hia_visuals.mjs', 'ml_lab_hia_visuals.css', 'ml_lab_structured_sandbox.js', 'ml_lab_structured_network.js'):
        (assets / name).write_text('const x = 1;')
    revision = _isolated_function(ROUTES, '_structured_controller_revision',
                                  file=location / 'workbench_structured_routes.py')
    before = revision()
    (assets / 'ml_lab_structured_network.js').write_text('const x = 2;')
    assert revision() != before


def test_all_graph_import_paths_are_content_versioned():
    # Source graph and whole-network graph share the installed renderer.
    for asset in ('interactive_graph_view.css', 'interactive_graph_view.mjs'):
        matches = re.findall(r"url_for\([^\n]*?asset='" + asset + r"'[^\n]*?\)", TEMPLATE)
        assert matches and all('v=shared_graph_revision' in match for match in matches)
    for filename in ('ml_lab_structured_graph.js', 'ml_lab_structured_sandbox.js',
                     'ml_lab_structured_network.js'):
        assert f"filename='{filename}', v=graph_controller_revision" in TEMPLATE
    assert 'shared_graph_revision=_shared_graph_asset_revision()' in ROUTES
    assert 'graph_controller_revision=_structured_controller_revision()' in ROUTES


def test_shared_graph_asset_is_not_cachable_by_browser():
    assert '"Cache-Control": "private, no-store"' in BLUEPRINT
    assert '"X-ML-Lab-Graph-Asset-Revision": revision' in BLUEPRINT
    # Preserve the original strict allowlist; do not serve arbitrary files.
    assert '{"interactive_graph_view.mjs", "interactive_graph_view.css"}' in BLUEPRINT
    assert 'private, max-age=3600' in BLUEPRINT  # unaffected non-graph assets
    controller = (UI / 'static/ml_lab_structured_graph.js').read_text()
    assert 'await import(root.dataset.graphModuleUrl)' in controller
    assert 'array missing or exceeds (5000|15000)' in controller
    assert 'Check the actual served asset' in controller
