/* HIA-06: opt-in branch information, exact source IDs and SU-05 column mappings.
 * The existing PAH_Shared graph remains the only renderer. No estimates run on
 * graph open, hover or selection; no source mutation or cross-module writes.
 */
(() => {
  'use strict';
  const root = document.querySelector('[data-mlwb-shared-graph]');
  const panel = root?.querySelector('[data-hia-branch]');
  if (!root || !panel) return;
  const $ = name => panel.querySelector(`[data-hia-branch-${name}]`);
  const action = name => panel.querySelector(`[data-hia-branch-action="${name}"]`);
  const status = value => { $('status').textContent = value; };
  const mapped = $('mappings'), table = $('table'), results = $('results');
  const member = $('node'), column = $('column'), contextNode = $('context-node');
  const contextColumn = $('context-column');
  let graph = null, branch = null, membership = new Set();
  let assignments = [], report = null, epoch = 0;
  const fill = (selector, names, empty) => {
    selector.replaceChildren();
    const first = document.createElement('option'); first.value = ''; first.textContent = empty;
    selector.append(first);
    for (const name of names) {
      const option = document.createElement('option'); option.value = name;
      option.textContent = name; selector.append(option);
    }
  };
  const invalidate = () => {
    epoch++;
    report = null;
    results.hidden = true;
    table.replaceChildren();
    $('report').textContent = '';
    action('export').disabled = true;
    action('compute').disabled = !branch || !assignments.length;
  };
  const reset = () => {
    graph = null; branch = null; membership = new Set(); assignments = [];
    panel.hidden = true; member.value = ''; contextNode.value = '';
    $('root').textContent = 'None'; $('coverage').textContent = '';
    mapped.replaceChildren(); fill(column, [], 'Select column…');
    fill(contextColumn, [], 'No context');
    invalidate(); status('Select Inspect branch, then explicitly map observed columns.');
  };
  const renderMappings = () => {
    mapped.replaceChildren();
    for (const [idx, mapping] of assignments.entries()) {
      const row = document.createElement('div'); row.className = 'mlwb-hia-neighbor-mapping';
      const label = document.createElement('span');
      label.textContent = `${mapping.node_id} → ${mapping.variable_id}`;
      const remove = document.createElement('button'); remove.type = 'button';
      remove.className = 'button secondary'; remove.textContent = 'Remove mapping';
      remove.addEventListener('click', () => {
        assignments.splice(idx, 1); renderMappings(); invalidate();
      });
      row.append(label, remove); mapped.append(row);
    }
    if (branch) $('coverage').textContent =
      `${membership.size} source branch nodes · ${assignments.length} explicitly mapped · ` +
      `${membership.size - assignments.length} unobserved/unmapped in this analysis.`;
  };
  root.addEventListener('mlwb:source-reset', reset);
  root.addEventListener('mlwb:source-open', event => {
    reset();
    graph = event.detail?.payload || null;
    const columns = graph?.binding_contract?.observed_columns || [];
    fill(column, columns, 'Select observed column…');
    fill(contextColumn, columns, 'No context');
  });
  root.addEventListener('mlwb:hia-scope', event => {
    const detail = event.detail || {};
    const scope = detail.selection?.scope;
    if (!graph || scope?.kind !== 'branch' || detail.source_revision !== graph.view.source_revision ||
        detail.source_digest !== graph.binding_contract?.source_digest) {
      branch = null; membership = new Set(); assignments = []; mapped.replaceChildren();
      panel.hidden = true; invalidate(); return;
    }
    const next = scope.node_ids?.[0];
    if (!next || !Array.isArray(detail.selection?.node_ids)) return;
    if (branch === next && membership.size === detail.selection.node_ids.length) return; // same page, same branch
    branch = next; membership = new Set(detail.selection.node_ids);
    assignments = []; member.value = ''; contextNode.value = ''; contextColumn.value = '';
    $('root').textContent = branch;
    panel.hidden = false;
    renderMappings(); invalidate();
    status('Choose original branch members and explicitly map their observed columns. No estimates have run.');
  });
  // Clicking a paged HIA-03 member offers its exact ID for mapping; it does not
  // automatically declare a mapping or launch an estimator.
  root.addEventListener('click', event => {
    const id = event.target.closest('[data-hia-member]')?.dataset.hiaMember;
    if (id && branch && membership.has(id)) member.value = id;
  });
  action('add').addEventListener('click', () => {
    const id = member.value, name = column.value;
    if (!membership.has(id)) { status('Choose an exact original node ID from the selected branch.'); return; }
    if (!name) { status('Choose an explicit observed column for this node.'); return; }
    if (assignments.some(item => item.node_id === id || item.variable_id === name)) {
      status('Mapped branch nodes and observed columns must be unique.'); return;
    }
    assignments.push({node_id:id, variable_id:name});
    renderMappings(); invalidate();
    status(`Declared ${assignments.length} branch mappings. Estimates run only when Compute is pressed.`);
  });
  for (const name of ['context-node','context-column','occurrences','bins','min-samples','min-stratum'])
    $(name).addEventListener('change', invalidate);
  const metric = (value, label) => {
    const div = document.createElement('div'); div.className = 'mlwb-hia-branch-metric';
    const desc = document.createElement('span'); desc.textContent = label;
    const strong = document.createElement('strong');
    strong.textContent = value === null || !Number.isFinite(value) ? 'Unavailable' : value.toFixed(5) + ' bits';
    div.append(desc, strong); return div;
  };
  const download = content => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(content, null, 2)], {type:'application/json'}));
    const link = document.createElement('a'); link.href = url;
    link.download = 'ml-lab-hia-branch-information.json';
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  };
  action('export').addEventListener('click', () => { if (report) download(report); });
  action('compute').addEventListener('click', async () => {
    if (!graph || !branch || !assignments.length) return;
    let bins;
    try {
      bins = JSON.parse($('bins').value || '{}');
      if (!bins || Array.isArray(bins) || typeof bins !== 'object') throw new Error('JSON object required');
    } catch (error) { status(`Discretization JSON unavailable: ${error.message}`); return; }
    const minimum = Number($('min-samples').value), minStratum = Number($('min-stratum').value);
    if (!Number.isSafeInteger(minimum) || minimum < 2 || !Number.isSafeInteger(minStratum) || minStratum < 1) {
      status('Minimum rows must be ≥ 2 and context-stratum rows ≥ 1.'); return;
    }
    const node = contextNode.value.trim(), name = contextColumn.value;
    if (Boolean(node) !== Boolean(name)) { status('Both external context node ID and column are required.'); return; }
    if (node && (membership.has(node) || !graph.view.nodes.some(item => item.id === node))) {
      status('External context must be an existing original node outside this branch.'); return;
    }
    const context = node ? {node_id:node,variable_id:name} : null;
    if (context && (assignments.some(item => item.node_id === node || item.variable_id === name))) {
      status('Context mapping must not reuse a branch column or node.'); return;
    }
    const version = ++epoch;
    const revision = graph.view.source_revision, selectedRoot = branch;
    action('compute').disabled = true; results.hidden = true; action('export').disabled = true;
    status(`Computing explicit observed-variable information for ${assignments.length} mapped branch nodes…`);
    try {
      const response = await fetch(root.dataset.hiaBranchUrl, {
        method:'POST', credentials:'same-origin', cache:'no-store',
        headers:{'Content-Type':'application/json','Accept':'application/json',
                 'X-ML-Lab-HIA':'explicit-branch-information'},
        body:JSON.stringify({source_revision:revision,branch_root_id:selectedRoot,
          mappings:assignments,context,occurrence_policy:$('occurrences').value,
          bins,min_samples:minimum,min_stratum_samples:minStratum}),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `Branch analysis failed (${response.status})`);
      if (version !== epoch || branch !== selectedRoot || !graph ||
          revision !== graph.view.source_revision || data.source_revision !== revision ||
          data.report?.source_digest !== graph.binding_contract?.source_digest) return;
      report = data.report;
      const branchInfo = report.branch_information, contextInfo = report.context_information;
      const summary = $('summary'); summary.replaceChildren();
      const coverage = document.createElement('p');
      coverage.textContent = `${report.mapped_node_count}/${report.branch_node_count} mapped original branch nodes · ` +
        `${branchInfo.support}/${report.aligned_rows} complete branch rows · ${branchInfo.status}`;
      summary.append(coverage);
      summary.append(metric(branchInfo.joint_entropy_bits, 'Joint mapped-branch entropy H(B)'));
      summary.append(metric(branchInfo.total_correlation_bits, 'Internal total correlation TC(B)'));
      if (contextInfo) {
        const contextLabel = document.createElement('p');
        contextLabel.textContent = `Context: ${contextInfo.context_node_id} (${contextInfo.context_variable_id}) ` +
          `· ${contextInfo.support} complete rows · ${contextInfo.status} · ` +
          `${contextInfo.context_is_direct_source_parent ? 'direct source parent' : 'external selected source context'}`;
        summary.append(contextLabel);
        summary.append(metric(contextInfo.branch_conditional_entropy_bits, 'H(B | context)'));
        summary.append(metric(contextInfo.branch_context_mutual_information_bits, 'I(B; context)'));
        summary.append(metric(contextInfo.conditional_total_correlation_bits, 'Conditional TC(B | context)'));
      }
      table.replaceChildren();
      for (const item of branchInfo.per_mapped_node) {
        const tr = document.createElement('tr');
        for (const value of [item.node_id, item.variable_id, item.entropy_bits,
          item.conditional_entropy_given_other_mapped_members_bits, item.mi_with_other_mapped_members_bits]) {
          const td = document.createElement('td');
          td.textContent = typeof value === 'number' ? value.toFixed(5) : value;
          tr.append(td);
        }
        table.append(tr);
      }
      $('report').textContent = JSON.stringify(report, null, 2);
      results.hidden = false; action('export').disabled = false;
      status(`Branch estimates: ${branchInfo.status}. Scope is ${report.statistical_scope}; context uses a separately labelled complete-case cohort.`);
    } catch (error) {
      if (version === epoch) status(`Branch statistics unavailable: ${error.message}`);
    } finally {
      if (version === epoch) action('compute').disabled = !branch || !assignments.length;
    }
  });
})();
