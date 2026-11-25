"""HIA-12 opt-in *local-only* Chromium smoke using the actual PAH_Shared renderer.

Run from the standalone ML_Lab checkout:

    python tests/hia12_local_browser_acceptance.py \
      --shared-viewer ../PAH_Shared/src/pah_shared/static/interactive_graph_view.mjs

Requires a preinstalled Playwright Python package and Chromium; installs
nothing and never opens an HTTP connection. Exercises viewer+HIA-11 overlay
interaction in one DOM, not the real PAH-hosted routes (live checklist in docs).
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / 'src/ml_lab/ui/templates/ml_lab_ui/workbench_structured.html'
OVERLAY = ROOT / 'src/ml_lab/ui/static/ml_lab_hia_visuals.mjs'


def _markup():
    text = TEMPLATE.read_text(encoding='utf-8')
    start = text.index('  <section class="mlwb-hia-visuals"')
    end = text.index('  <section class="mlwb-hia-conditional"', start)
    return ('''<!DOCTYPE html><html><head><meta charset="utf-8"></head><body>
      <main data-mlwb-shared-graph><div data-graph-surface></div>'''
            + text[start:end] + '''</main></body></html>''')


def run(shared_path: Path, chromium: str | None):
    from playwright.sync_api import sync_playwright

    code = shared_path.read_text(encoding='utf-8')
    overlay = OVERLAY.read_text(encoding='utf-8')
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, executable_path=chromium,
                                     args=['--no-sandbox', '--disable-gpu'])
        page = browser.new_page(accept_downloads=True)
        try:
            page.route('http://**/*', lambda route: route.abort())
            page.route('https://**/*', lambda route: route.abort())
            page.goto('about:blank')
            page.set_content(_markup())
            page.add_script_tag(type='module', content=overlay)
            page.evaluate('''async (code) => {
              const blob = new Blob([code], {type: 'text/javascript'});
              const url = URL.createObjectURL(blob);
              try {
                const {mountInteractiveGraph} = await import(url);
                const root = document.querySelector('[data-mlwb-shared-graph]');
                const surface = root.querySelector('[data-graph-surface]');
                const view = {
                  schema:'pah-shared.graph-view@1',view_id:'hia12-real-shared',
                  source_revision:'sha256:fixture',graph_role:'source_structure',
                  title:'Local HIA acceptance graph',
                  nodes:[{id:'root',label:'root',kind:'atom',group:'depth-0'},
                         {id:'a',label:'a',kind:'atom',group:'depth-1'},
                         {id:'unmapped',label:'unmapped',kind:'atom',group:'depth-1'}],
                  edges:[{id:'e0',source:'root',target:'a',directed:true,traceable:true},
                         {id:'e1',source:'root',target:'unmapped',directed:true,traceable:true}],
                  display:{layout:'layers',coverage:{scope:'exhaustive',total_nodes:3,total_edges:2}},
                  overlays:{nodes:{},edges:{}}
                };
                window.hia12Input = JSON.stringify(view);
                window.hia12Viewer = mountInteractiveGraph(surface,view,{
                  onSelect:detail=>root.dispatchEvent(new CustomEvent('mlwb:graph-select',{detail})),
                  onViewportChange:detail=>root.dispatchEvent(new CustomEvent('mlwb:graph-viewport',{detail}))
                });
                root.dispatchEvent(new CustomEvent('mlwb:source-open',{detail:{
                  payload:{view,binding_contract:{source_digest:'digest'}},viewer:window.hia12Viewer}}));
                root.dispatchEvent(new CustomEvent('mlwb:hia-network-report',{detail:{
                  source_revision:'sha256:fixture',report:{
                    source_digest:'digest',binding_digest:'bound',source_topology:'tree',
                    network_information:{status:'ok',support:40,levels:[
                      {level:0,mapped_node_ids:['root'],mapped_node_count:1,structural_node_count:1,
                       level_joint_entropy_bits:1,incremental_joint_entropy_given_prior_bits:1,
                       information_shared_with_prior_bits:0,within_level_total_correlation_bits:0},
                      {level:1,mapped_node_ids:['a'],mapped_node_count:1,structural_node_count:2,
                       level_joint_entropy_bits:1,incremental_joint_entropy_given_prior_bits:.5,
                       information_shared_with_prior_bits:.5,within_level_total_correlation_bits:0}
                    ],cross_level_pairs:[{left_level:0,right_level:1,mutual_information_bits:.5,support:40}]}
                  }}}));
              } finally {URL.revokeObjectURL(url);}
            }''', code)
            page.wait_for_function('''() => document.querySelectorAll('[data-hia11-overlay="measured"]').length === 2''')
            assert page.locator('[data-graph-surface] [data-node-id="unmapped"] [data-hia11-overlay]').count() == 0
            assert page.locator('[data-hia-visuals-comparison] table tr').count() >= 5
            old_camera = page.evaluate('window.hia12Viewer.camera.zoom')
            assert page.evaluate("window.hia12Viewer.select('node','a',{recenter:true})") is True
            assert page.evaluate('window.hia12Viewer.selection.id') == 'a'
            assert page.evaluate('window.hia12Viewer.camera.zoom') == old_camera
            assert page.locator('[data-graph-surface] [data-hia11-overlay="measured"]').count() == 2
            for action, suffix in [('csv','.csv'),('svg','.svg'),('png','.png')]:
                with page.expect_download(timeout=15000) as info:
                    page.locator(f'[data-hia-visuals-action="{action}"]').click()
                assert info.value.suggested_filename.endswith(suffix)
            page.evaluate('''() => document.querySelector('[data-mlwb-shared-graph]').dispatchEvent(
              new CustomEvent('mlwb:hia-network-invalidated'))''')
            page.wait_for_function('''() => document.querySelectorAll('[data-hia11-overlay="measured"]').length === 0''')
            assert page.evaluate('JSON.stringify({input:JSON.parse(window.hia12Input),selected:window.hia12Viewer.selection.id})')
            page.evaluate('''() => document.querySelector('[data-mlwb-shared-graph]').dispatchEvent(
              new CustomEvent('mlwb:source-reset'))''')
            assert page.locator('[data-hia-visuals]').is_hidden()
            print('HIA-12 local Chromium: REAL PAH_Shared mount/selection, overlay, unmapped, '
                  'matched comparison, camera, CSV/SVG/PNG, invalidation/reset: PASS')
        finally:
            browser.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shared-viewer', type=Path,
        default=ROOT.parent/'PAH_Shared/src/pah_shared/static/interactive_graph_view.mjs')
    parser.add_argument('--chromium', default=None,
        help='Optional path to a locally installed Chromium binary')
    args = parser.parse_args()
    if not args.shared_viewer.is_file():
        parser.error('Standalone PAH_Shared viewer not found; pass --shared-viewer with its path')
    try:
        run(args.shared_viewer, args.chromium)
    except ModuleNotFoundError as exc:
        parser.error(f'Optional local browser dependency not installed: {exc.name}')


if __name__ == '__main__':
    main()
