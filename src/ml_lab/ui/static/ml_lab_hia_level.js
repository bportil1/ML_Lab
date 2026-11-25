/* HIA-07: exact structural-level inventory and opt-in empirical level statistics.
 * No observations are guessed from graph topology. The PAH_Shared viewer stays
 * the only graph engine. No computation occurs on source open or selection.
 */
(() => {
  'use strict';
  const root = document.querySelector('[data-mlwb-shared-graph]');
  const panel = root?.querySelector('[data-hia-level-stats]');
  if (!root || !panel) return;
  const $ = name => panel.querySelector(`[data-hia-level-${name}]`);
  const action = name => panel.querySelector(`[data-hia-level-action="${name}"]`);
  const tell = message => { $('status').textContent = message; };
  const member = $('node'), column = $('column'), mapped = $('mappings');
  const results = $('results'), table = $('table');
  let graph = null, level = null, membership = new Set();
  let assignments = [], report = null, epoch = 0;
  const fill = names => {
    column.replaceChildren();
    const first = document.createElement('option'); first.value = '';
    first.textContent = 'Select observed column…'; column.append(first);
    for (const name of names) {
      const option = document.createElement('option');
      option.value = name; option.textContent = name; column.append(option);
    }
  };
  const invalidate = () => {
    epoch++;
    report = null; results.hidden = true; table.replaceChildren();
    $('report').textContent = '';
    action('export').disabled = true;
    action('compute').disabled = level === null || !assignments.length;
  };
  const reset = () => {
    graph = null; level = null; membership = new Set(); assignments = [];
    panel.hidden = true; member.value = ''; $('depth').textContent = 'None';
    $('coverage').textContent = ''; mapped.replaceChildren(); fill([]);
    invalidate(); tell('Inspect a structural level, then explicitly map observed columns.');
  };
  const renderMappings = () => {
    mapped.replaceChildren();
    for (const [idx, assignment] of assignments.entries()) {
      const row = document.createElement('div'); row.className = 'mlwb-hia-neighbor-mapping';
      const label = document.createElement('span');
      label.textContent = `${assignment.node_id} → ${assignment.variable_id}`;
      const remove = document.createElement('button'); remove.type = 'button';
      remove.className = 'button secondary'; remove.textContent = 'Remove mapping';
      remove.addEventListener('click', () => {
        assignments.splice(idx, 1); renderMappings(); invalidate();
      });
      row.append(label, remove); mapped.append(row);
    }
    if (level !== null) $('coverage').textContent =
      `Level ${level}: ${membership.size} original nodes · ${assignments.length} explicitly mapped · ` +
      `${membership.size - assignments.length} unmapped in this analysis.`;
  };
  root.addEventListener('mlwb:source-reset', reset);
  root.addEventListener('mlwb:source-open', event => {
    reset();
    graph = event.detail?.payload || null;
    fill(graph?.binding_contract?.observed_columns || []);
  });
  root.addEventListener('mlwb:hia-scope', event => {
    const detail = event.detail || {};
    const selected = detail.selection;
    if (!graph || selected?.scope?.kind !== 'level' ||
        detail.source_revision !== graph.view.source_revision ||
        detail.source_digest !== graph.binding_contract?.source_digest) {
      level = null; membership = new Set(); assignments = [];
      mapped.replaceChildren(); panel.hidden = true; invalidate(); return;
    }
    const depth = selected.scope.level;
    if (!Number.isSafeInteger(depth) || depth < 0 || !Array.isArray(selected.node_ids)) return;
    if (level === depth && membership.size === selected.node_ids.length) return;
    level = depth; membership = new Set(selected.node_ids);
    assignments = []; member.value = ''; $('depth').textContent = String(level);
    panel.hidden = false; renderMappings(); invalidate();
    tell('Select original level members, then explicitly associate their observed columns.');
  });
  // Clicking an original member navigates the graph and offers the exact ID,
  // but does not authorize a statistical binding or run an estimator.
  root.addEventListener('click', event => {
    const id = event.target.closest('[data-hia-member]')?.dataset.hiaMember;
    if (id && level !== null && membership.has(id)) member.value = id;
  });
  action('add').addEventListener('click', () => {
    const id = member.value.trim(), name = column.value;
    if (level === null || !membership.has(id)) {
      tell('Select an exact original source ID from this level.'); return;
    }
    if (!name) { tell('Select an observed column for this member.'); return; }
    if (assignments.some(a => a.node_id === id || a.variable_id === name)) {
      tell('A mapped level node and observed column can each be used only once.'); return;
    }
    assignments.push({node_id:id, variable_id:name});
    renderMappings(); invalidate();
    tell(`Declared ${assignments.length} mappings; Compute must be pressed to estimate.`);
  });
  for (const name of ['occurrences','bins','min-samples'])
    $(name).addEventListener('change', invalidate);
  const metric = (value, label) => {
    const div = document.createElement('div'); div.className = 'mlwb-hia-branch-metric';
    const desc = document.createElement('span'); desc.textContent = label;
    const strong = document.createElement('strong');
    strong.textContent = value === null || !Number.isFinite(value)
      ? 'Unavailable' : value.toFixed(5) + ' bits';
    div.append(desc, strong); return div;
  };
  const download = content => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(content, null, 2)], {type:'application/json'}));
    const link = document.createElement('a'); link.href = url;
    link.download = `ml-lab-hia-level-${level}-information.json`;
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  };
  action('export').addEventListener('click', () => { if (report) download(report); });
  action('compute').addEventListener('click', async () => {
    if (!graph || level === null || !assignments.length) return;
    let bins;
    try {
      bins = JSON.parse($('bins').value || '{}');
      if (!bins || Array.isArray(bins) || typeof bins !== 'object')
        throw new Error('Expected a JSON object');
    } catch (error) { tell(`Invalid discretization JSON: ${error.message}`); return; }
    const minimum = Number($('min-samples').value);
    if (!Number.isSafeInteger(minimum) || minimum < 2) {
      tell('Minimum complete rows must be an integer ≥ 2.'); return;
    }
    const token = ++epoch, revision = graph.view.source_revision, selectedDepth = level;
    action('compute').disabled = true; results.hidden = true;
    action('export').disabled = true;
    tell(`Estimating level ${level} information for ${assignments.length} declared observed variables…`);
    try {
      const response = await fetch(root.dataset.hiaLevelUrl, {
        method:'POST', credentials:'same-origin', cache:'no-store',
        headers:{'Content-Type':'application/json','Accept':'application/json',
                 'X-ML-Lab-HIA':'explicit-level-information'},
        body: JSON.stringify({source_revision:revision, level:selectedDepth,
          mappings:assignments, occurrence_policy:$('occurrences').value,
          bins, min_samples:minimum}),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `Level analysis failed (${response.status})`);
      if (token !== epoch || !graph || level !== selectedDepth ||
          revision !== graph.view.source_revision || data.source_revision !== revision ||
          data.report?.source_digest !== graph.binding_contract?.source_digest) return;
      report = data.report;
      const info = report.level_information;
      const summary = $('summary'); summary.replaceChildren();
      const description = document.createElement('p');
      description.textContent = `Level ${level} (${report.level_policy}) · ` +
        `${report.mapped_node_count}/${report.level_node_count} source nodes mapped · ` +
        `${info.support}/${report.aligned_rows} complete rows · ${info.status}`;
      summary.append(description);
      summary.append(metric(info.joint_entropy_bits, 'Joint level entropy H(L)'));
      summary.append(metric(info.sum_atomic_entropy_bits, 'Sum of mapped-node entropies'));
      summary.append(metric(info.total_correlation_bits, 'Within-level total correlation TC(L)'));
      table.replaceChildren();
      for (const item of info.per_mapped_node) {
        const tr = document.createElement('tr');
        for (const value of [item.node_id, item.variable_id, item.entropy_bits,
          item.conditional_entropy_given_other_mapped_members_bits,
          item.mi_with_other_mapped_members_bits]) {
          const td = document.createElement('td');
          td.textContent = typeof value === 'number' ? value.toFixed(5) : value;
          tr.append(td);
        }
        table.append(tr);
      }
      $('report').textContent = JSON.stringify(report, null, 2);
      results.hidden = false; action('export').disabled = false;
      tell(`Level ${level}: ${info.status}. ${report.statistical_scope}; ` +
        `earlier-level conditional information is not computed in HIA-07.`);
    } catch (error) {
      if (token === epoch) tell(`Level estimates unavailable: ${error.message}`);
    } finally {
      if (token === epoch) action('compute').disabled = level === null || !assignments.length;
    }
  });
})();
