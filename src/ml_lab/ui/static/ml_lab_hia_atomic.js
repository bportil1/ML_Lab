/* HIA-04: opt-in atomic source-node profile. Declared SU-05 mapping only.
 * Read-only SHA-pinned POST; never calculate on hover or assume a graph edge
 * implies a statistical variable. Original HIA-03 and SU-10 stay untouched.
 */
(() => {
  'use strict';
  const root = document.querySelector('[data-mlwb-shared-graph]');
  const panel = root?.querySelector('[data-hia-atomic]');
  if (!root || !panel) return;
  const $ = selector => panel.querySelector(`[data-hia-atomic-${selector}]`);
  const column = $('column');
  const nodeOutput = $('node');
  const status = $('status');
  const results = $('results');
  const summary = $('summary');
  const raw = $('report');
  const compute = panel.querySelector('[data-hia-atomic-action="compute"]');
  const exportButton = panel.querySelector('[data-hia-atomic-action="export"]');
  let graph = null, selectedNode = null, report = null, token = 0;
  const say = message => { status.textContent = message; };
  const invalidate = () => {
    token++;
    report = null; results.hidden = true; summary.replaceChildren(); raw.textContent = '';
    exportButton.disabled = true;
    compute.disabled = !graph || !selectedNode || !column.value;
  };
  const reset = () => {
    graph = null; selectedNode = null; column.replaceChildren(); nodeOutput.textContent = 'None';
    panel.hidden = true; invalidate();
    say('Select a source node and explicitly declare the corresponding observed column.');
  };
  root.addEventListener('mlwb:source-reset', reset);
  root.addEventListener('mlwb:source-open', event => {
    graph = event.detail?.payload || null;
    selectedNode = null; nodeOutput.textContent = 'None';
    column.replaceChildren();
    const empty = document.createElement('option'); empty.value = '';
    empty.textContent = 'Select a column…'; column.append(empty);
    const names = graph?.binding_contract?.observed_columns || [];
    for (const name of names) {
      const option = document.createElement('option'); option.value = name;
      option.textContent = name; column.append(option);
    }
    panel.hidden = false; invalidate();
    say(names.length ? 'Select an original node, then confirm its observed column. No mapping is inferred.'
      : 'This source has no aligned observation rows. Structural node identities are available, but empirical statistics cannot be estimated.');
  });
  root.addEventListener('mlwb:graph-select', event => {
    const d = event.detail;
    if (d?.kind !== 'node' || !graph || d.source_revision !== graph.view.source_revision) return;
    selectedNode = d.id; nodeOutput.textContent = selectedNode;
    // Never reuse a previous node's mapping or cached statistics on a new node.
    column.value = ''; invalidate();
    say('Original node selected. Explicitly identify its aligned observation column before computing.');
  });
  for (const element of [column, $('occurrences'), $('bin-mode'), $('bin-count'), $('min-samples')]) {
    element.addEventListener('change', invalidate);
  }
  const labelMetric = (parent, label, value) => {
    const item = document.createElement('div');
    const title = document.createElement('span'); title.textContent = label;
    const content = document.createElement('strong'); content.textContent = String(value);
    item.append(title, content); parent.append(item);
  };
  compute.addEventListener('click', async () => {
    if (!graph || !selectedNode || !column.value) return;
    const selectedRevision = graph.view.source_revision;
    const currentNode = selectedNode;
    const currentVariable = column.value;
    const minSamples = Number($('min-samples').value);
    const binCount = Number($('bin-count').value);
    const mode = $('bin-mode').value;
    if (!Number.isSafeInteger(minSamples) || minSamples < 2 ||
        (mode !== 'none' && (!Number.isSafeInteger(binCount) || binCount < 2 || binCount > 128))) {
      say('Minimum rows must be at least 2; explicit bin count must be from 2 through 128.'); return;
    }
    const requestToken = ++token;
    compute.disabled = true; exportButton.disabled = true; results.hidden = true;
    say('Computing observed statistics with the exact declared SU-05 node/column binding…');
    try {
      const response = await fetch(root.dataset.hiaAtomicUrl, {
        method:'POST', credentials:'same-origin', cache:'no-store',
        headers:{'Content-Type':'application/json','Accept':'application/json',
                 'X-ML-Lab-HIA':'explicit-atomic-statistics'},
        body:JSON.stringify({source_revision:selectedRevision, node_id:currentNode,
          variable_id:currentVariable, occurrence_policy:$('occurrences').value,
          bins:mode === 'none' ? null : {mode,bins:binCount}, min_samples:minSamples}),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || `Atomic analysis failed (${response.status})`);
      if (requestToken !== token || !graph || graph.view.source_revision !== payload.source_revision ||
          selectedNode !== currentNode || column.value !== currentVariable ||
          payload.report.node_id !== currentNode || payload.report.variable_id !== currentVariable ||
          payload.report.source_digest !== graph.binding_contract.source_digest) return;
      report = payload.report;
      const statistics = report.numeric?.statistics;
      const metrics = document.createElement('div'); metrics.className = 'mlwb-hia-atomic-metrics';
      labelMetric(metrics,'Aligned observation rows',report.support.aligned_rows);
      labelMetric(metrics,'Observed scalar rows',report.support.observed_scalar_rows);
      labelMetric(metrics,'Missing/unresolved rows',report.support.unavailable_rows);
      labelMetric(metrics,'Empirical entropy', report.entropy.status === 'ok'
        ? `${report.entropy.bits.toFixed(5)} bits` : `Unavailable: ${report.entropy.status}`);
      labelMetric(metrics,'Numeric mean',statistics ? statistics.mean : 'Unavailable');
      labelMetric(metrics,'Sample variance',statistics?.sample_variance ?? 'Unavailable');
      labelMetric(metrics,'Structural depth',report.node_context.level);
      summary.replaceChildren(metrics);
      raw.textContent = JSON.stringify(report, null, 2);
      results.hidden = false; exportButton.disabled = false;
      say(`Measured ${currentVariable} at ${currentNode}. ${report.entropy.status === 'ok' ? 'Entropy estimated.' : 'Entropy ' + report.entropy.status + '.'} No statistical dependence inferred from source topology.`);
    } catch (error) {
      if (requestToken === token) say(`Atomic statistics unavailable: ${error.message}`);
    } finally {
      if (requestToken === token) compute.disabled = !graph || !selectedNode || !column.value;
    }
  });
  exportButton.addEventListener('click', () => {
    if (!report) return;
    const blob = new Blob([JSON.stringify(report, null, 2)], {type:'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url;
    a.download = 'ml-lab-hia-atomic-node-statistics.json'; document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  });
})();
