/* Whole-network empirical information controller; scientific analysis is owned
 * by ML_Lab, visualization by the single installed PAH_Shared graph engine.
 * All results are ephemeral and source SHA-bound. Never infer structural,
 * probabilistic, causal, or propagation edges from empirical MI/CMI.
 */
(() => {
  'use strict';
  const panel = document.querySelector('[data-mlwb-information-network]');
  if (!panel) return;
  const el = name => panel.querySelector(`[data-network-${name}]`);
  const action = name => panel.querySelector(`[data-network-action="${name}"]`);
  const status = text => { el('status').textContent = text; };
  const names = value => value.split(',').map(x => x.trim()).filter(Boolean);
  let result = null, module = null, viewer = null, busy = false;
  let pairLookup = new Map(), nodeLookup = new Map();
  const sourceRevision = panel.dataset.revision;

  function reset() {
    if (viewer) viewer.destroy();
    viewer = null;
    el('surface').replaceChildren();
    el('graph-controls').hidden = true;
    el('results').hidden = true;
    el('matrix').replaceChildren();
    el('inspector').textContent = 'Select a graph node, link or matrix cell.';
    pairLookup.clear(); nodeLookup.clear();
    result = null; action('export').disabled = true;
  }
  function pretty(value) { return JSON.stringify(value, null, 2); }
  function inspect(kind, id) {
    const data = kind === 'edge' ? pairLookup.get(id) : nodeLookup.get(id);
    if (!data) return;
    el('inspector').textContent = pretty({
      record_type: kind === 'edge' ? 'empirical_association' : 'observed_variable',
      ...data,
      limitations: kind === 'edge'
        ? ['Observational, undirected association; not a causal, structural, or learned probabilistic edge.']
        : ['Original observed column, not necessarily an original structural node.'],
    });
  }
  function filteredView(threshold) {
    const original = result.view;
    const edges = original.edges.filter(edge => {
      const pair = pairLookup.get(edge.id);
      return pair && pair.status === 'ok' && pair.value_bits >= threshold;
    });
    const keys = new Set(edges.map(edge => edge.id));
    return {...original, edges,
      display:{...original.display, coverage:{scope:'filtered',
        total_nodes:original.nodes.length, total_edges:original.edges.length}},
      overlays:{nodes:original.overlays.nodes,
        edges:Object.fromEntries(Object.entries(original.overlays.edges)
          .filter(([key]) => keys.has(key)))}};
  }
  function renderGraph() {
    if (!result) return;
    if (!module) {
      el('surface').textContent = 'PAH_Shared interactive graph viewer unavailable. The complete measured matrix remains available below. Install the shared SU-02 renderer locally for graph visualization.';
      return;
    }
    const threshold = Number(el('threshold').value);
    if (!Number.isFinite(threshold) || threshold < 0) {
      status('Display threshold must be a nonnegative finite number of bits.'); return;
    }
    const view = filteredView(threshold);
    try {
      module.validateInteractiveGraphView(view);
      if (!viewer) {
        viewer = module.mountInteractiveGraph(el('surface'), view, {
          onSelect: ({kind, id, source_revision}) => {
            if (source_revision === sourceRevision) inspect(kind, id);
          },
        });
        viewer.fit();
      } else viewer.update(view, {preserveSelection:true});
      el('graph-controls').hidden = false;
      status(`Empirical network ready: showing ${view.edges.length} of ${result.view.edges.length} measured associations at ≥ ${threshold} bits. Matrix includes every attempted pair.`);
    } catch (error) {
      status(`Shared graph rejected a view: ${error.message}`);
    }
  }
  function renderMatrix() {
    const report = result.report, table = el('matrix');
    const nodes = report.nodes, pairs = report.pairs;
    const max = Math.max(0.00001, ...pairs.filter(p => p.value_bits !== null)
      .map(p => p.value_bits));
    const byPair = new Map();
    for (const pair of pairs) {
      byPair.set(`${pair.source_id}/${pair.target_id}`, pair);
      byPair.set(`${pair.target_id}/${pair.source_id}`, pair);
    }
    const head = document.createElement('thead'), header = document.createElement('tr');
    const corner = document.createElement('th'); corner.scope = 'col';
    corner.textContent = 'Variable / variable'; header.append(corner);
    for (const node of nodes) {
      const cell = document.createElement('th'); cell.scope = 'col';
      cell.textContent = node.variable_id;
      cell.title = node.variable_id; header.append(cell);
    }
    head.append(header); table.append(head);
    const body = document.createElement('tbody');
    for (const a of nodes) {
      const row = document.createElement('tr');
      const label = document.createElement('th'); label.scope = 'row';
      label.textContent = a.variable_id; row.append(label);
      for (const b of nodes) {
        const cell = document.createElement('td');
        const same = a.id === b.id;
        const pair = same ? null : byPair.get(`${a.id}/${b.id}`);
        const value = same ? a.entropy_bits : pair?.value_bits;
        const id = same ? a.id : pair?.id;
        const kind = same ? 'node' : 'edge';
        const button = document.createElement('button');
        button.type = 'button'; button.className = 'mlwb-network-matrix-cell';
        button.textContent = value === null || value === undefined ? '—' : Number(value).toFixed(3);
        button.title = same ? `${a.variable_id}: H(X) · n=${a.support}`
          : `${a.variable_id} ↔ ${b.variable_id} · n=${pair?.support ?? 0} · ${pair?.status ?? 'unavailable'}`;
        button.addEventListener('click', () => {
          inspect(kind, id);
          if (viewer && (same || result.view.edges.some(e => e.id === id))) {
            const visible = same || filteredView(Number(el('threshold').value)).edges.some(e => e.id === id);
            if (visible) viewer.select(kind, id, {recenter:true});
          }
        });
        if (value === null || value === undefined) button.dataset.tone = 'dark';
        else if (same) button.dataset.tone = 'light';
        else {
          const opacity = Math.min(0.72, 0.12 + 0.60 * value / max);
          button.style.backgroundColor = `rgba(35, 105, 180, ${opacity})`;
          button.dataset.tone = opacity >= 0.42 ? 'light' : 'dark';
        }
        cell.append(button); row.append(cell);
      }
      body.append(row);
    }
    table.append(body);
  }
  async function compute() {
    if (busy) return;
    busy = true; action('compute').disabled = true; reset();
    status('Computing all requested observed-variable pairs from the verified source. No sampling, edits, or jobs are started…');
    try {
      const bins = JSON.parse(el('bins').value || '{}');
      const payload = {
        source_revision: sourceRevision,
        variables: names(el('variables').value),
        conditioning: names(el('conditioning').value), bins,
        missing_policy: el('missing').value,
        min_samples: Number(el('min-samples').value),
        min_stratum_samples: Number(el('min-stratum').value),
      };
      const response = await fetch(panel.dataset.networkUrl, {
        method:'POST', credentials:'same-origin', cache:'no-store',
        headers:{'Content-Type':'application/json', 'Accept':'application/json',
                 'X-ML-Lab-Network':'explicit-analysis'},
        body:JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `Network analysis failed (${response.status})`);
      if (data.source_revision !== sourceRevision || data.view.source_revision !== sourceRevision)
        throw new Error('Source changed while computing the network; reopen the file.');
      result = data;
      pairLookup = new Map(data.report.pairs.map(x => [x.id, x]));
      nodeLookup = new Map(data.report.nodes.map(x => [x.id, x]));
      el('results').hidden = false;
      el('summary').textContent = `${data.report.support.total_variables} variables · ${data.report.support.pairs_requested} pairs attempted · ${data.report.support.pairs_measured} measured · ${data.report.support.aligned_rows} aligned rows · ${data.report.metric} · source ${data.report.source_digest.slice(0, 24)}…`;
      const joint = data.report.joint_dependency;
      el('summary').textContent += joint.status === 'ok'
        ? ` · full-joint H=${joint.joint_entropy_bits.toFixed(3)} bits · total correlation=${joint.total_correlation_bits.toFixed(3)} bits (n=${joint.support})`
        : ` · full-joint dependency unavailable (${joint.reason})`;
      el('metadata').textContent = pretty({estimator:data.report.estimator,
        conditioning:data.report.conditioning, joint_dependency:data.report.joint_dependency,
        support:data.report.support,
        limitations:data.report.limitations, source_digest:data.report.source_digest});
      renderMatrix();
      if (panel.dataset.sharedGraphUrl) {
        try { module = await import(panel.dataset.sharedGraphUrl); }
        catch (error) { module = null; status(`Shared graph unavailable: ${error.message}; full matrix available.`); }
      }
      renderGraph();
      action('export').disabled = false;
    } catch (error) { reset(); status(`Network unavailable: ${error.message}`); }
    finally { busy = false; action('compute').disabled = false; }
  }
  action('compute').addEventListener('click', compute);
  el('threshold').addEventListener('change', renderGraph);
  action('fit').addEventListener('click', () => viewer?.fit());
  action('overview').addEventListener('click', () => viewer?.overview());
  action('clear').addEventListener('click', () => {
    el('inspector').textContent = 'Select a graph node, link or matrix cell.';
    if (viewer) viewer.update(filteredView(Number(el('threshold').value)), {preserveSelection:false});
  });
  action('export').addEventListener('click', () => {
    if (!result) return;
    const data = new Blob([JSON.stringify({source_revision:result.source_revision,
      report:result.report}, null, 2)], {type:'application/json'});
    const url = URL.createObjectURL(data);
    const a = document.createElement('a'); a.href = url;
    a.download = 'ml_lab_whole_information_network.json'; a.click();
    URL.revokeObjectURL(url);
  });
})();
