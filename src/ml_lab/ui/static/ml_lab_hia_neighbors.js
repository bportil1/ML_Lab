/* HIA-05: opt-in observed associations between explicitly mapped source neighbors.
 * The existing shared graph owns rendering, pan/zoom, selection and exports.
 * This controller neither infers bindings from structural links nor computes
 * numerical estimates on hover/select, and never persists scientific results.
 */
(() => {
  'use strict';
  const root = document.querySelector('[data-mlwb-shared-graph]');
  const panel = root?.querySelector('[data-hia-neighbors]');
  if (!root || !panel) return;
  const $ = name => panel.querySelector(`[data-hia-neighbor-${name}]`);
  const action = name => panel.querySelector(`[data-hia-neighbor-action="${name}"]`);
  const say = message => { $('status').textContent = message; };
  const center = $('center-column'), choice = $('choice'), column = $('column');
  const givenNode = $('given-node'), givenColumn = $('given-column');
  const table = $('table'), results = $('results'), mappings = $('mappings');
  let graph = null, node = null, neighbors = [], declared = [], report = null, token = 0;
  const fill = (selector, names, placeholder) => {
    selector.replaceChildren();
    const blank = document.createElement('option'); blank.value = ''; blank.textContent = placeholder;
    selector.append(blank);
    for (const name of names) {
      const option = document.createElement('option'); option.value = name;
      option.textContent = name; selector.append(option);
    }
  };
  const invalidate = () => {
    token++;
    report = null; results.hidden = true; table.replaceChildren();
    $('report').textContent = ''; action('export').disabled = true;
    action('compute').disabled = !graph || !node || !center.value || !declared.length;
  };
  const reset = () => {
    graph = null; node = null; neighbors = []; declared = [];
    panel.hidden = true; $('node').textContent = 'None'; mappings.replaceChildren();
    for (const select of [center, choice, column, givenNode, givenColumn]) fill(select, [], 'Choose…');
    invalidate(); say('Select a source node and explicitly declare the neighboring observed columns.');
  };
  const renderDeclared = () => {
    mappings.replaceChildren();
    for (const [index, item] of declared.entries()) {
      const row = document.createElement('div'); row.className = 'mlwb-hia-neighbor-mapping';
      const desc = document.createElement('span');
      desc.textContent = `${item.node_id} → observed column ${item.variable_id}`;
      const remove = document.createElement('button'); remove.type = 'button';
      remove.className = 'button secondary'; remove.textContent = 'Remove mapping';
      remove.setAttribute('aria-label', `Remove ${item.node_id}`);
      remove.addEventListener('click', () => {
        declared.splice(index, 1); renderDeclared(); invalidate();
      });
      row.append(desc, remove); mappings.append(row);
    }
  };
  root.addEventListener('mlwb:source-reset', reset);
  root.addEventListener('mlwb:source-open', event => {
    graph = event.detail?.payload || null;
    node = null; neighbors = []; declared = []; mappings.replaceChildren();
    const names = graph?.binding_contract?.observed_columns || [];
    fill(center, names, 'Choose selected node column…');
    fill(column, names, 'Choose neighbor column…');
    fill(givenColumn, names, 'No conditioning');
    fill(givenNode, graph?.view?.nodes?.map(item => item.id) || [], 'No conditioning');
    fill(choice, [], 'Choose adjacent node…');
    panel.hidden = false; $('node').textContent = 'None'; invalidate();
    say(names.length ? 'Select an original graph node and explicitly map its observed columns.'
      : 'No aligned observations were found in this source; numerical associations are unavailable.');
  });
  root.addEventListener('mlwb:graph-select', event => {
    const detail = event.detail;
    if (!graph || detail?.kind !== 'node' || detail.source_revision !== graph.view.source_revision) return;
    node = detail.id;
    neighbors = [...new Set((graph.view.edges || []).flatMap(edge =>
      edge.source === node ? [edge.target] : edge.target === node ? [edge.source] : []))].sort();
    $('node').textContent = node; declared = []; center.value = ''; column.value = '';
    fill(choice, neighbors, 'Choose adjacent node…');
    renderDeclared(); invalidate();
    say(neighbors.length ? `${neighbors.length} original structural neighbors available. Their edges do not assert statistical dependence.`
      : 'No directly adjacent structural neighbors. Select a different node.');
  });
  action('add').addEventListener('click', () => {
    if (!node || !choice.value || !column.value || !center.value) {
      say('Choose the selected-node column, adjacent source node, and its observed column.'); return;
    }
    if (column.value === center.value || declared.some(item => item.node_id === choice.value || item.variable_id === column.value)) {
      say('Each original node and observed column must be unique within this comparison.'); return;
    }
    declared.push({node_id:choice.value, variable_id:column.value});
    renderDeclared(); invalidate();
    say(`Declared ${declared.length} original neighbor mappings. Compute only when ready.`);
  });
  for (const input of [center, givenNode, givenColumn, $('occurrences'), $('min-samples'), $('min-stratum'), $('bins')]) {
    input.addEventListener('change', invalidate);
  }
  const format = (metric, field, digits = 5) =>
    metric?.status === 'ok' && metric[field] !== null && Number.isFinite(metric[field])
      ? metric[field].toFixed(digits) : `— (${metric?.status || 'unavailable'})`;
  const exportJSON = () => {
    if (!report) return;
    const blob = new Blob([JSON.stringify(report, null, 2)], {type:'application/json'});
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a'); link.href = url;
    link.download = 'ml-lab-hia-neighbor-relationships.json';
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  };
  action('export').addEventListener('click', exportJSON);
  action('compute').addEventListener('click', async () => {
    if (!graph || !node || !center.value || !declared.length) return;
    let bins;
    try {
      bins = JSON.parse($('bins').value || '{}');
      if (!bins || typeof bins !== 'object' || Array.isArray(bins)) throw new Error('Object required');
    } catch (error) { say(`Invalid discretization JSON: ${error.message}`); return; }
    const min = Number($('min-samples').value), stratum = Number($('min-stratum').value);
    if (!Number.isSafeInteger(min) || min < 2 || !Number.isSafeInteger(stratum) || stratum < 1) {
      say('Minimum paired rows must be ≥ 2; stratum rows must be ≥ 1.'); return;
    }
    const conditioning = [];
    if (givenNode.value || givenColumn.value) {
      if (!givenNode.value || !givenColumn.value) { say('Declare both the conditioner node and observed column.'); return; }
      conditioning.push({node_id:givenNode.value, variable_id:givenColumn.value});
    }
    const assignments = [{node_id:node, variable_id:center.value}, ...declared, ...conditioning];
    if (new Set(assignments.map(item => item.node_id)).size !== assignments.length ||
        new Set(assignments.map(item => item.variable_id)).size !== assignments.length) {
      say('Conditioning, selected node, and neighbors must all use distinct node and observation-column IDs.'); return;
    }
    const current = node, variable = center.value, revision = graph.view.source_revision;
    const stamp = ++token;
    action('compute').disabled = true; action('export').disabled = true; results.hidden = true;
    say(`Computing ${declared.length} explicitly bound neighbor comparisons from aligned rows…`);
    try {
      const response = await fetch(root.dataset.hiaNeighborsUrl, {
        method:'POST', credentials:'same-origin', cache:'no-store',
        headers:{'Content-Type':'application/json','Accept':'application/json',
                 'X-ML-Lab-HIA':'explicit-neighbor-statistics'},
        body:JSON.stringify({source_revision:revision,node_id:current,variable_id:variable,
          neighbors:declared,conditioning,occurrence_policy:$('occurrences').value,
          bins,min_samples:min,min_stratum_samples:stratum}),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `Neighbor analysis failed (${response.status})`);
      if (stamp !== token || !graph || node !== current || center.value !== variable ||
          data.source_revision !== graph.view.source_revision ||
          data.report?.source_digest !== graph.binding_contract?.source_digest) return;
      report = data.report; table.replaceChildren();
      for (const item of report.relationships) {
        const row = document.createElement('tr');
        const cells = [item.node_id, format(item.mutual_information, 'mi_bits'),
          format(item.mutual_information, 'h_x_given_y_bits'),
          format(item.mutual_information, 'h_y_given_x_bits'),
          item.conditional_mutual_information ? format(item.conditional_mutual_information, 'mi_bits') : 'Not requested',
          format(item.correlations, 'pearson'), format(item.correlations, 'spearman'),
          String(item.mutual_information.support)];
        for (const value of cells) {
          const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
        }
        table.append(row);
      }
      $('report').textContent = JSON.stringify(report, null, 2);
      results.hidden = false; action('export').disabled = false;
      say(`Completed ${report.relationships.length} explicit comparisons. Displayed values are empirical associations, not interpretations of structural edges.`);
    } catch (error) {
      if (stamp === token) say(`Neighbor statistics unavailable: ${error.message}`);
    } finally {
      if (stamp === token) action('compute').disabled = !graph || !node || !center.value || !declared.length;
    }
  });
})();
