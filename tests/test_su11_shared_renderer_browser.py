"""Actual PAH_Shared SU-02 renderer smoke in a local, isolated Chromium page.

This tests the real JS engine rather than a mock/copy. It is a component-browser
check, NOT the Flask/PAH/pyPIQUE integrated UI gate. All requests are local.
"""
from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import threading

import pytest

from ml_lab.structured import StructuredObject
from ml_lab.ui.structured_graph_view import make_shared_graph_view


def test_real_shared_renderer_browser_component(tmp_path):
    viewer = os.environ.get('SU11_SHARED_VIEWER')
    if not viewer or not Path(viewer).is_file():
        pytest.skip('supply installed SU11_SHARED_VIEWER .mjs for real renderer')
    if os.environ.get('SU11_BROWSER_COMPONENT') != '1':
        pytest.skip('opt-in real browser component: SU11_BROWSER_COMPONENT=1')
    sync = pytest.importorskip('playwright.sync_api')
    chromium = os.environ.get('SU11_CHROMIUM') or shutil.which('chromium') or shutil.which('google-chrome')
    if not chromium:
        pytest.skip('local Chromium not installed')
    source = (Path(__file__).parent/'fixtures/structured_uncertainty/xml_tree.json').read_text()
    obj = StructuredObject.from_json(source)
    view = make_shared_graph_view(obj, fingerprint='a'*64)
    assert view['display']['coverage']['scope'] == 'exhaustive'
    shutil.copy2(viewer, tmp_path/'shared.mjs')
    (tmp_path/'source.json').write_text(json.dumps(view, allow_nan=False), encoding='utf-8')
    (tmp_path/'index.html').write_text('''<!doctype html><html><head><meta charset="utf-8"></head>
      <body><p id="status">before</p><div id="graph" style="width:850px;height:550px"></div>
      <script type="module">
      import {validateInteractiveGraphView,mountInteractiveGraph,interactiveGraphRows} from './shared.mjs';
      try {
        const view=validateInteractiveGraphView(await (await fetch('./source.json')).json());
        const rows=interactiveGraphRows(view);
        if(rows.nodes.length !== view.nodes.length) throw Error('lost node rows');
        const viewer=mountInteractiveGraph(document.getElementById('graph'),view,{
          onSelect:({kind,id})=>{document.getElementById('status').textContent=`selected:${kind}:${id}`;}
        });
        window.acceptanceViewer=viewer;
        viewer.fit();
        document.getElementById('status').textContent='ready';
      } catch(err) { document.getElementById('status').textContent='failed:'+err.message; }
      </script></body></html>''', encoding='utf-8')

    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, directory=str(tmp_path)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync.sync_playwright() as pw:
            browser = pw.chromium.launch(executable_path=chromium, headless=True,
                                         args=['--no-sandbox', '--disable-dev-shm-usage'])
            context = browser.new_context()
            address = f'http://127.0.0.1:{server.server_port}'
            remote = []
            def only_local(route):
                if route.request.url.startswith(address + '/'):
                    route.continue_()
                else:
                    remote.append(route.request.url)
                    route.abort()
            context.route('**/*', only_local)
            page = context.new_page()
            errors=[]
            page.on('pageerror', lambda err: errors.append(str(err)))
            try:
                page.goto(address+'/index.html')
            except sync.Error as exc:
                if 'ERR_BLOCKED_BY_ADMINISTRATOR' in str(exc):
                    pytest.skip('environment policy blocks localhost Chromium navigation')
                raise
            page.wait_for_function("document.querySelector('#status')?.textContent !== 'before'",timeout=15000)
            assert page.locator('#status').inner_text() == 'ready', page.locator('#status').inner_text()
            assert page.locator('#graph svg').count() >= 1
            page.evaluate("window.acceptanceViewer.select('node', 'section')")
            assert page.locator('#status').inner_text() == 'selected:node:section'
            page.evaluate("window.acceptanceViewer.zoom(1.25); window.acceptanceViewer.fit();")
            page.evaluate("window.acceptanceViewer.overview(); window.acceptanceViewer.context();")
            assert not errors
            assert not remote
            context.close()
            browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)
