/* HIA-03: opt-in hierarchical navigation over the existing PAH_Shared viewer.
 * HIA-02 assigns levels; this controller never guesses statistical layers,
 * estimates MI/CMI/entropy, or mutates original graph/source artifacts.
 */
(() => {
  'use strict';
  const root = document.querySelector('[data-mlwb-shared-graph]');
  const panel = root?.querySelector('[data-hia-workspace]');
  if (!root || !panel) return;
  const nodeId = panel.querySelector('[data-hia-selected-node]');
  const levelInput = panel.querySelector('[data-hia-level]');
  const levelSummary = panel.querySelector('[data-hia-level-count]');
  const status = panel.querySelector('[data-hia-status]');
  const info = panel.querySelector('[data-hia-inspector]');
  const members = panel.querySelector('[data-hia-members]');
  const pageInfo = panel.querySelector('[data-hia-page-info]');
  const controls = panel.querySelector('[data-hia-controls]');
  let graph = null, viewer = null, overview = null, report = null;
  let selectedNode = null, selection = null, offset = 0, epoch = 0;
  let preservingScopeMember = false;
  const action = name => panel.querySelector(`[data-hia-action="${name}"]`);
  const say = message => { status.textContent = message; };
  const scope = (kind, id = null) => ({kind, node_ids: id === null ? [] : [id], level: null});
  const download = (filename, content, type) => {
    const blob = new Blob([content], {type});
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url; link.download = filename; document.body.append(link); link.click();
    link.remove();
    // Deferral allows browsers to finish reading the object URL before revocation.
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  };
  const reset = () => {
    epoch++;
    graph = null; viewer = null; overview = null; report = null;
    selectedNode = null; selection = null; offset = 0;
    nodeId.textContent = 'None'; levelSummary.textContent = '';
    pageInfo.textContent = '';
    panel.hidden = true; controls.hidden = true;
    members.replaceChildren();
    info.textContent = 'Choose a verified structural scope to inspect its membership.';
    say('No hierarchy selected.');
  };
  function highlight() {
    if (!viewer || !report) return;
    const included = new Set(report.selection.node_ids);
    // Only currently rendered nodes exist as SVG DOM elements in the scalable
    // PAH_Shared viewer. The complete original selection remains in the report.
    for (const item of root.querySelectorAll('[data-graph-surface] [data-node-id]')) {
      item.classList.toggle('mlwb-hia-in-scope', included.has(item.getAttribute('data-node-id')));
    }
  }
  function render(result) {
    report = result;
    const page = result.member_page;
    offset = page.offset;
    members.replaceChildren();
    for (const id of page.node_ids) {
      const button = document.createElement('button');
      button.type = 'button'; button.className = 'button secondary mlwb-hia-member';
      button.textContent = id;
      button.title = `Locate original source node ${id}`;
      button.dataset.hiaMember = id;
      members.append(button);
    }
    pageInfo.textContent = `Original source identities ${page.total ? page.offset + 1 : 0}–${page.offset + page.node_ids.length} of ${page.total}. Display page only; no nodes discarded.`;
    action('previous').disabled = !page.has_previous;
    action('next').disabled = !page.has_next;
    const selected = result.selection;
    const display = {
      scope: selected.scope,
      original_node_count: selected.node_count,
      structural_levels: selected.levels,
      source_roots: selected.roots,
      direct_context: result.node_context,
      mapped_observation_variables: selected.mapped_variable_ids,
      statistical_estimates: 'Not computed in HIA-03; HIA-04–10 will add opt-in estimators',
      scientific_metric_eligibility: selected.metrics,
      structural_edges_are_not_statistical_dependencies: true,
    };
    info.textContent = JSON.stringify(display, null, 2);
    action('export-scope').disabled = false;
    say(`${selected.scope.kind} selection · ${selected.node_count} original nodes · HIA-02 ${result.level_policy}. No numerical estimates computed.`);
    highlight();
    root.dispatchEvent(new CustomEvent('mlwb:hia-scope', {detail: {
      selection: selected, source_revision: result.source_revision, source_digest: result.source_digest,
    }}));
  }
  async function select(next, pageOffset = 0) {
    if (!graph || !overview?.available) return;
    if (next.kind === 'branch' && !overview.branch_selection_allowed) {
      say('Branch selection requires unique-parent tree or forest topology; use a node or level for DAGs.');
      return;
    }
    const token = ++epoch;
    selection = next;
    offset = pageOffset;
    say(`Inspecting ${next.kind} in verified source…`);
    try {
      const response = await fetch(root.dataset.hiaInspectUrl, {
        method: 'POST', credentials: 'same-origin', cache: 'no-store',
        headers: {'Content-Type':'application/json', 'Accept':'application/json',
                  'X-ML-Lab-HIA':'explicit-inspection'},
        body: JSON.stringify({source_revision:graph.view.source_revision, scope:next, offset:pageOffset}),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Hierarchy selection unavailable');
      if (token !== epoch || data.source_revision !== graph.view.source_revision ||
          data.source_digest !== overview.source_digest) return;
      render(data);
    } catch (error) {
      if (token !== epoch) return;
      report = null;
      members.replaceChildren();
      action('export-scope').disabled = true;
      say(`Hierarchy inspection unavailable: ${error.message}`);
    }
  }
  root.addEventListener('mlwb:source-reset', reset);
  root.addEventListener('mlwb:source-open', async event => {
    graph = event.detail?.payload;
    viewer = event.detail?.viewer || null;
    if (!graph || !viewer) return;
    const token = ++epoch;
    try {
      const response = await fetch(root.dataset.hiaOverviewUrl, {
        credentials:'same-origin', cache:'no-store', headers:{'Accept':'application/json'},
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'Hierarchy unavailable');
      if (token !== epoch) return;
      if (result.source_revision !== graph.view.source_revision ||
          result.source_digest !== graph.binding_contract?.source_digest)
        throw new Error('HIA hierarchy source fingerprint does not match the displayed graph');
      overview = result;
      controls.hidden = false;
      action('export-graph').disabled = false;
      action('export-svg').disabled = false;
      panel.hidden = false;
      if (!result.available) {
        levelInput.disabled = true;
        action('level').disabled = true;
        action('branch').disabled = true;
        action('node').disabled = true;
        action('network').disabled = true;
        say('This verified source is not a tree, forest, or DAG; hierarchy levels are unavailable. Original graph controls remain active.');
        return;
      }
      levelInput.disabled = false;
      levelInput.min = '0';
      levelInput.max = String(Math.max(0, result.levels.length - 1));
      levelInput.value = '0';
      levelSummary.textContent = `${result.levels.length} levels · ${result.node_count} original nodes · ${result.level_policy}`;
      action('level').disabled = result.levels.length === 0;
      action('branch').disabled = true; // Enable only after selecting an original node.
      action('node').disabled = true;
      action('network').disabled = result.node_count === 0;
      action('export-scope').disabled = true;
      say('Hierarchy ready: click an original node, select a level, or inspect the whole network.');
    } catch (error) {
      if (token !== epoch) return;
      panel.hidden = false;
      controls.hidden = false;
      say(`Hierarchy unavailable: ${error.message}. The original graph remains usable.`);
    }
  });
  root.addEventListener('mlwb:graph-select', event => {
    const detail = event.detail;
    if (!overview?.available || detail?.kind !== 'node' ||
        detail.source_revision !== graph?.view.source_revision) return;
    selectedNode = detail.id;
    nodeId.textContent = selectedNode;
    action('node').disabled = false;
    action('branch').disabled = !overview.branch_selection_allowed;
    if (!preservingScopeMember) select(scope('node', selectedNode));
  });
  // Reapply semantic selection after PAH_Shared redraws a page or viewport.
  root.addEventListener('mlwb:graph-viewport', highlight);
  members.addEventListener('click', event => {
    const id = event.target.closest('[data-hia-member]')?.dataset.hiaMember;
    if (id && viewer) {
      // Navigating within a level/branch must not discard that chosen scope.
      preservingScopeMember = true;
      try { viewer.select('node', id, {recenter:true}); }
      finally { preservingScopeMember = false; }
    }
  });
  panel.addEventListener('click', event => {
    const name = event.target.closest('[data-hia-action]')?.dataset.hiaAction;
    if (!name || !graph || !viewer) return;
    if (name === 'node' && selectedNode) select(scope('node', selectedNode));
    if (name === 'branch' && selectedNode) select(scope('branch', selectedNode));
    if (name === 'network') select(scope('network'));
    if (name === 'level') {
      const depth = Number(levelInput.value);
      if (Number.isSafeInteger(depth) && depth >= 0 && depth < overview.levels.length)
        select({kind:'level',node_ids:[],level:depth});
      else say('Choose an existing integer structural depth.');
    }
    if (name === 'previous' && report) select(selection, Math.max(0, offset - report.member_page.page_size));
    if (name === 'next' && report) select(selection, offset + report.member_page.page_size);
    if (name === 'fullscreen') {
      if (document.fullscreenElement === root) document.exitFullscreen?.();
      else root.requestFullscreen?.().catch(error=>say(`Fullscreen unavailable: ${error.message}`));
    }
    if (name === 'popout') {
      const url = new URL(window.location.href);
      url.searchParams.set('section','data'); url.searchParams.set('hia_popout','1');
      url.hash = 'mlwb-hia-graph';
      window.open(url.toString(), '_blank', 'noopener');
    }
    if (name === 'export-graph') {
      download('ml-lab-source-graph-complete.json', JSON.stringify({
        view:graph.view, original_records:graph.inspector, hierarchy_overview:overview,
        exports_are_original_source_records:true,
      }, null, 2), 'application/json');
    }
    if (name === 'export-scope' && report) {
      download('ml-lab-hierarchical-scope.json', JSON.stringify(report, null, 2),'application/json');
    }
    if (name === 'export-svg' && viewer.svg) {
      const svg = viewer.svg.cloneNode(true);
      svg.setAttribute('xmlns','http://www.w3.org/2000/svg');
      // Make the visible-page SVG portable: external UI stylesheets do not
      // accompany a file opened by itself in Firefox or an image viewer.
      const style = document.createElementNS('http://www.w3.org/2000/svg','style');
      style.textContent = `
        .pah-ig-node-bg{fill:#fff;stroke:#6a86a2;stroke-width:1.8}
        .pah-ig-node-label{fill:#17324d;font:600 12.5px system-ui,sans-serif;paint-order:stroke;stroke:#fff;stroke-width:3px;stroke-linejoin:round}
        .pah-ig-edge-line{fill:none;stroke:#5f7993;stroke-width:2}
        .pah-ig-arrow-head{fill:#5f7993}
        .pah-ig-node.is-selected .pah-ig-node-bg{stroke:#096fa8;stroke-width:3.5}
        .pah-ig-node.mlwb-hia-in-scope .pah-ig-node-bg{stroke:#dd7722;stroke-width:3}
        .pah-ig-edge.is-selected .pah-ig-edge-line{stroke:#096fa8;stroke-width:4}
        .pah-ig-node.is-dimmed,.pah-ig-edge.is-dimmed{opacity:.2}`;
      svg.insertBefore(style,svg.firstChild);
      const background = document.createElementNS('http://www.w3.org/2000/svg','rect');
      background.setAttribute('width','100%'); background.setAttribute('height','100%');
      background.setAttribute('fill','#f8fbff');
      svg.insertBefore(background,style.nextSibling);
      download('ml-lab-current-graph-page.svg', new XMLSerializer().serializeToString(svg), 'image/svg+xml');
    }
  });
  if (new URLSearchParams(window.location.search).get('hia_popout') === '1') {
    document.body.classList.add('mlwb-hia-popout');
    // The new window is a standalone browser document; reuse the same project-
    // local SHA-verified route and graph controller, never copy graph data in URL.
    requestAnimationFrame(() => root.querySelector('[data-graph-action="open"]')?.click());
  }
})();
