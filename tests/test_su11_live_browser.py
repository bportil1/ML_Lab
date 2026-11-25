"""Opt-in SU-11 real Flask + installed PAH_Shared + Chromium acceptance.

Set SU11_BROWSER=1 and install local optional UI/browser dependencies to run.
A simulated document or mock renderer is NOT counted as live-browser proof.
The WSGI server binds only 127.0.0.1; no other HTTP hosts are used.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import threading

import pytest


@pytest.mark.skipif(os.environ.get('SU11_BROWSER') != '1',
                    reason='opt-in live acceptance: SU11_BROWSER=1')
def test_real_flask_browser_graph_evidence_preview_navigation_and_source_integrity(tmp_path):
    flask = pytest.importorskip('flask')
    playwright = pytest.importorskip('playwright.sync_api')
    from werkzeug.serving import make_server
    from ml_lab.ui import create_ui_blueprint
    from ml_lab.ui.workbench_structured_routes import _shared_graph_available

    if not _shared_graph_available():
        pytest.skip('SU-02 PAH_Shared .mjs and .css not installed locally')
    browser_executable = os.environ.get('SU11_CHROMIUM') or shutil.which('chromium') or shutil.which('google-chrome')
    if not browser_executable:
        pytest.skip('local Chromium executable not installed')

    source = Path(__file__).parent/'fixtures/structured_uncertainty/xml_tree.json'
    initial = source.read_bytes()
    local_file = tmp_path/'source.json'
    local_file.write_bytes(initial)

    app = flask.Flask(__name__)
    app.config.update(TESTING=True, SECRET_KEY='su11-test-only')
    app.register_blueprint(create_ui_blueprint(name='su11_live', experiment_project_root=str(tmp_path)),
                           url_prefix='/lab')
    host = make_server('127.0.0.1', 0, app, threaded=True)
    server = threading.Thread(target=host.serve_forever, daemon=True)
    server.start()
    try:
        with playwright.sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, executable_path=browser_executable,
                args=['--no-sandbox', '--disable-dev-shm-usage'])
            context = browser.new_context(accept_downloads=True)
            base = f'http://127.0.0.1:{host.server_port}'
            # Make any unexpected remote request an acceptance failure, not an
            # opportunity to send code or observations off-device.
            unexpected = []
            def local_only(route):
                if not route.request.url.startswith(base + '/'):
                    unexpected.append(route.request.url.split('?', 1)[0])
                    route.abort()
                else:
                    route.continue_()
            context.route('**/*', local_only)
            page = context.new_page()
            errors = []
            page.on('pageerror', lambda err: errors.append(str(err)))
            page.goto(base + '/lab/workbench/structured?section=data')
            page.locator('input[name="path"]').fill('source.json')
            page.locator('button[name="action"][value="select"]').click()
            page.locator('[data-graph-action="open"]').click()
            page.locator('[data-graph-rows] button').first.wait_for(timeout=15000)
            # Choose a real source leaf from the linked original record table.
            entries = page.locator('[data-graph-rows] button')
            assert entries.count() > 1
            entries.nth(1).click()
            page.locator('[data-su10-action="inspect"]').click()
            page.locator('[data-su10-results]:not([hidden])').wait_for(timeout=15000)
            assert 'explicit_SU_05_binding_required' in page.locator('[data-su10-evidence]').inner_text()
            page.locator('[data-su10-action="preview"]').click()
            page.locator('[data-su10-preview]:not([hidden])').wait_for(timeout=15000)
            assert page.locator('[data-su10-comparison]').inner_text().strip()
            page.locator('[data-su10-action="export"]').click()
            # All UI actions must leave original artifact bytes alone.
            assert local_file.read_bytes() == initial
            assert not unexpected, 'Unexpected non-local requests: ' + repr(unexpected)
            assert not errors, 'Browser JavaScript errors: ' + repr(errors)
            for section in ('profiling','representation','comparison','tools'):
                page.goto(base + f'/lab/workbench/structured?section={section}')
                assert page.locator('nav.mlwb-structured-tabs a[aria-current="page"]').count() == 1
            context.close()
            browser.close()
    finally:
        host.shutdown()
        server.join(timeout=5)
