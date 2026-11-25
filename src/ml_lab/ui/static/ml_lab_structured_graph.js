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
  let viewer = null;
  let snapshot = null;
  let graphModule = null;
  let busy = false;

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
  function renderRows(module, view) {
    rows.replaceChildren();
    const table = module.interactiveGraphRows(view);
    const all = [...table.nodes.map(row => ({...row, record_type:'node'})),
      ...table.edges.map(row => ({...row, record_type:'edge'}))];
    const visible = all.slice(0, 80);
    for (const record of visible) {
      const entry = document.createElement('button');
      entry.type = 'button';
      entry.className = 'button secondary mlwb-graph-row';
      entry.textContent = `${record.record_type} · ${record.label || record.kind || record.id} (${record.id})`;
      entry.addEventListener('click', () => viewer?.select(record.record_type, record.id, {recenter:true}));
      rows.appendChild(entry);
    }
    const note = root.querySelector('[data-graph-row-count]');
    note.textContent = `Linked record controls: ${visible.length} of ${all.length} (bounded preview).`;
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
      module.validateInteractiveGraphView(payload.view);
      snapshot = payload;
      graphModule = module;
      viewer = module.mountInteractiveGraph(surface, payload.view, {
        onSelect: inspect,
        onViewportChange: ({coverage}) => {
          say(`${coverage.scope} display · ${coverage.rendered_nodes} of ${coverage.total_nodes} nodes · ${coverage.rendered_edges} of ${coverage.total_edges} edges. Original graph unchanged.`);
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
  root.addEventListener('click', event => {
    const action = event.target.closest('[data-graph-action]')?.dataset.graphAction;
    if (!viewer || !action || action === 'open') return;
    if (action === 'fit') viewer.fit();
    if (action === 'zoom-in') viewer.zoom(1.25);
    if (action === 'zoom-out') viewer.zoom(0.8);
    if (action === 'overview') viewer.overview();
    if (action === 'context') viewer.context();
    if (action === 'close') { reset(); say('Graph preview closed. No source data changed.'); }
  });
})();
