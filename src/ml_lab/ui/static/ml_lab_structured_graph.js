/* SU-04: opt-in ML_Lab owner/controller; PAH_Shared draws the actual graph.
 * Local-only, no scientific computation or persistence. Every load verifies
 * the source's original fingerprint again on the ML_Lab route.
 */
(() => {
  'use strict';
  const root = document.querySelector('[data-mlwb-shared-graph]');
  if (!root) return;
  const button = root.querySelector('[data-graph-action="open"]');
  const surface = root.querySelector('[data-graph-surface]');
  const status = root.querySelector('[data-graph-status]');
  const inspector = root.querySelector('[data-graph-inspector]');
  const rows = root.querySelector('[data-graph-rows]');
  const controls = root.querySelector('[data-graph-controls]');
  const recordSearch = root.querySelector('[data-graph-row-search]');
  const recordCount = root.querySelector('[data-graph-row-count]');
  let viewer = null;
  let snapshot = null;
  let graphModule = null;
  let busy = false;
  let recordPage = 0;
  const RECORD_PAGE_SIZE = 80;

  const say = text => { status.textContent = text; };
  const reset = () => {
    if (viewer) viewer.destroy();
    viewer = null;
    snapshot = null;
    graphModule = null;
    surface.replaceChildren();
    inspector.textContent = 'Select a graph node or edge to inspect its original metadata.';
    rows.replaceChildren();
    controls.hidden = true;
    recordPage = 0;
    if(recordSearch) recordSearch.value = '';
    root.dispatchEvent(new CustomEvent('mlwb:source-reset'));
  };
  function inspect({source_revision, kind, id}) {
    if (!snapshot || source_revision !== snapshot.view.source_revision) {
      inspector.textContent = 'Source revision changed; reload the original structure.';
      return;
    }
    root.dispatchEvent(new CustomEvent('mlwb:graph-select', {detail: {source_revision, kind, id}}));
    const record = snapshot.inspector?.[kind === 'node' ? 'nodes' : 'edges']?.[id];
    inspector.textContent = record ? JSON.stringify(record, null, 2).slice(0, 8000)
      : 'This display summary has no original source record.';
  }
  function renderRows(_module, view) {
    // Do not allocate another full copy of the graph's records just to render
    // the selector. A page and optional search expose the entire source set.
    rows.replaceChildren();
    const query = (recordSearch?.value || '').trim().toLocaleLowerCase();
    const sources = [
      ['node', view.nodes || []], ['edge', view.edges || []],
    ];
    const matches = (kind, record) => !query || [kind,record.id,record.label,record.kind,
      record.source,record.target].some(value => String(value??'').toLocaleLowerCase().includes(query));
    let count=0;
    for (const [kind, collection] of sources) for(const entry of collection) {
      if(matches(kind,entry)) count++;
    }
    const pages=Math.max(1,Math.ceil(count/RECORD_PAGE_SIZE));
    recordPage=Math.min(recordPage,pages-1);
    const first=recordPage*RECORD_PAGE_SIZE,last=first+RECORD_PAGE_SIZE;
    let position=0;
    for (const [kind, collection] of sources) {
      for(const record of collection) {
        if(!matches(kind,record)) continue;
        const index=position++;
        if(index<first || index>=last) continue;
        const entry=document.createElement('button');
        entry.type='button';entry.className='button secondary mlwb-graph-row';
        entry.textContent=`${kind} · ${record.label||record.kind||record.id} (${record.id})`;
        entry.addEventListener('click',()=>viewer?.select(kind,record.id,{recenter:true}));
        rows.appendChild(entry);
      }
    }
    recordCount.textContent = `Source records ${count?first+1:0}–${Math.min(last,count)} of ${count} · page ${recordPage+1}/${pages}. No records discarded.`;
    for(const [name,disabled] of [['previous-records',recordPage===0],['next-records',recordPage+1>=pages]]) {
      const button=root.querySelector(`[data-graph-action="${name}"]`);
      if(button) button.disabled=disabled;
    }
  }

  async function open() {
    if (busy) return;
    busy = true;
    button.disabled = true;
    reset();
    say('Verifying selected source and loading shared graph…');
    try {
      const response = await fetch(root.dataset.graphPayloadUrl, {
        method: 'GET', cache: 'no-store', credentials: 'same-origin',
        headers: {'Accept': 'application/json'}
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || 'Selected structure no longer available');
      const module = await import(root.dataset.graphModuleUrl);
      try {
        module.validateInteractiveGraphView(payload.view);
      } catch (error) {
        // This is a signature of the pre-GRAPH-SCALE-01 PAH_Shared module.
        // Tell the user which *served* module to inspect. A Git/source or
        // PAH host-venv version check cannot verify the separate ML_Lab runtime.
        if (/array missing or exceeds (5000|15000)/.test(String(error?.message))) {
          throw new Error(`ML_Lab loaded an obsolete PAH_Shared graph module at ${root.dataset.graphModuleUrl}. Check the actual served asset and restart the ML_Lab runtime, not the scientific job.`);
        }
        throw error;
      }
      snapshot = payload;
      graphModule = module;
      viewer = module.mountInteractiveGraph(surface, payload.view, {
        onSelect: inspect,
        onViewportChange: ({coverage}) => {
          const page = coverage.page_count > 1 ? ` · group page ${coverage.page_index+1}/${coverage.page_count}` : '';
          const edgePage = coverage.edge_page_count > 1 ? ` · edge page ${coverage.edge_page_index+1}/${coverage.edge_page_count}` : '';
          say(`${coverage.scope} display · ${coverage.rendered_nodes} visual nodes / ${coverage.total_nodes} original nodes · ${coverage.rendered_edges} visual edges / ${coverage.total_edges} original edges${page}${edgePage}. No source records discarded.`);
        },
      });
      controls.hidden = false;
      root.dispatchEvent(new CustomEvent('mlwb:source-open', {detail: {payload, module}}));
      renderRows(module, payload.view);
      viewer.fit();
      say(`Verified ${payload.view.source_revision.slice(0, 24)}… · ${payload.notice}`);
    } catch (error) {
      reset();
      say(`Graph preview unavailable: ${error.message}`);
    } finally {
      busy = false;
      button.disabled = false;
    }
  }
  root.addEventListener('mlwb:sandbox-overlay', event => {
    const data = event.detail;
    if (!viewer || !snapshot || !graphModule ||
        data?.source_revision !== snapshot.view.source_revision ||
        !snapshot.inspector?.nodes?.[data.node_id]) return;
    // The original graph and revision are untouched. This is a detached,
    // owner-calculated statistical node overlay accepted by PAH_Shared.
    const view = {...snapshot.view, overlays: {
      nodes: {[data.node_id]: data.graph_overlay.overlay}, edges: {},
    }};
    graphModule.validateInteractiveGraphView(view);
    viewer.update(view, {preserveSelection:true});
    viewer.select('node', data.node_id);
  });
  root.addEventListener('mlwb:source-focus', event => {
    if (viewer && event.detail?.node_id) viewer.select('node', event.detail.node_id, {recenter:true});
  });
  button.addEventListener('click', open);
  recordSearch?.addEventListener('input',()=>{
    if (!snapshot) return;
    recordPage=0;renderRows(graphModule,snapshot.view);
  });
  root.addEventListener('click', event => {
    const action = event.target.closest('[data-graph-action]')?.dataset.graphAction;
    if (!viewer || !action || action === 'open') return;
    if (action === 'fit') viewer.fit();
    if (action === 'zoom-in') viewer.zoom(1.25);
    if (action === 'zoom-out') viewer.zoom(0.8);
    if (action === 'overview') viewer.overview();
    if (action === 'next-page') viewer.nextPage();
    if (action === 'previous-page') viewer.previousPage();
    if (action === 'previous-records') {recordPage=Math.max(0,recordPage-1);renderRows(graphModule,snapshot.view);}
    if (action === 'next-records') {recordPage++;renderRows(graphModule,snapshot.view);}
    if (action === 'context') viewer.context();
    if (action === 'close') { reset(); say('Graph preview closed. No source data changed.'); }
  });
})();
