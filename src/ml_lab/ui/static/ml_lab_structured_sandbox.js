/* SU-10: ML_Lab's owner/controller consumes the SAME PAH_Shared graph engine
 * used by the SU-04 original graph. No renderer, remote access or persistence.
 * Evidence and transformations run only upon an explicit button press.
 */
(() => {
  'use strict';
  const root = document.querySelector('[data-mlwb-shared-graph]');
  const panel = root?.querySelector('[data-su10-sandbox]');
  if (!panel) return;
  const field = name => panel.querySelector(`[data-su10-${name}]`);
  const action = name => panel.querySelector(`[data-su10-action="${name}"]`);
  const message = field('status');
  const selection = field('selection');
  const originalFocus = node_id => root.dispatchEvent(new CustomEvent('mlwb:source-focus', {detail:{node_id}}));
  const list = name => field(name).value.split(',').map(s => s.trim()).filter(Boolean);
  const readJson = (text, fallback) => text.trim() ? JSON.parse(text) : fallback;
  let source = null, module = null, selected = null, preview = null, result = null;
  let busy = false;

  function state(text) { message.textContent = text; }
  function clearPreview() {
    preview?.destroy(); preview = null;
    field('preview-surface').replaceChildren();
    field('preview').hidden = true;
  }
  function clearReports({keepPreview=false} = {}) {
    result = null;
    if (!keepPreview) clearPreview();
    field('results').hidden = true;
    field('node-overlay').replaceChildren();
    field('removed').replaceChildren();
    action('export').disabled = true;
  }
  function setSelection(node_id) {
    if (node_id === selected) return; // Inspector refocus must not discard evidence.
    const keepPreview = Boolean(node_id && preview);
    selected = node_id;
    selection.textContent = selected || 'No node selected';
    action('inspect').disabled = !selected;
    action('preview').disabled = !selected;
    clearReports({keepPreview});
    if (selected) state(keepPreview
      ? `Selected ${selected}. Previous candidate preview is pinned to the earlier branch; rerun to update.`
      : `Selected ${selected}. Submit an explicit evidence or preview request.`);
  }
  root.addEventListener('mlwb:source-reset', () => {
    source = module = null; field('column').replaceChildren(); setSelection(null); clearReports();
    state('Explore the original source graph to begin.');
  });
  root.addEventListener('mlwb:source-open', event => {
    source = event.detail.payload; module = event.detail.module;
    const columns = source.binding_contract?.observed_columns || [];
    field('column').replaceChildren();
    for (const name of columns) {
      const option = document.createElement('option');
      option.value = name; option.textContent = name; field('column').append(option);
    }
    state(columns.length
      ? 'Source verified. Select a node; explicitly map observed columns before requesting empirical evidence.'
      : 'Source verified. No aligned observations were supplied; statistical measures remain unavailable.');
  });
  root.addEventListener('mlwb:graph-select', event => {
    if (!source || event.detail.source_revision !== source.view.source_revision) return;
    setSelection(event.detail.kind === 'node' ? event.detail.id : null);
  });

  action('map').addEventListener('click', () => {
    if (!source || !selected || !field('column').value) {
      state('Select a source node and an actual observed column first.'); return;
    }
    try {
      const contract = source.binding_contract;
      const current = readJson(field('binding').value, {
        schema: contract.schema, source_digest: contract.source_digest,
        observation_unit: contract.observation_unit, variables:[], group_variables:[], weight_variable:null,
      });
      if (current.source_digest !== contract.source_digest ||
          current.schema !== contract.schema || current.observation_unit !== contract.observation_unit ||
          !Array.isArray(current.variables)) throw new Error('Binding plan identity does not match the selected source.');
      if (current.variables.some(v => v.variable_id === field('column').value)) {
        throw new Error('Column already bound. Edit the plan explicitly to alter or merge its mapping.');
      }
      current.variables.push({variable_id:field('column').value, node_ids:[selected],
                              occurrence_policy:'preserve', merged_sources:false});
      current.group_variables = list('groups');
      field('binding').value = JSON.stringify(current, null, 2);
      clearReports();
      state(`Added explicit mapping: ${field('column').value} → ${selected}.`);
    } catch (error) { state(`Mapping not added: ${error.message}`); }
  });

  function payload(mode) {
    if (!source || !selected) throw new Error('Select an original source node.');
    const binding = readJson(field('binding').value, null);
    const data = {
      source_revision:source.view.source_revision, mode, node_id:selected,
      binding_plan:binding,
      uncertainty_targets:list('targets'), group_levels:list('groups'),
      information_target:field('info-target').value.trim() || null,
      retained_variables:list('retained'),
      include_information:field('information').checked,
      information_bins:readJson(field('bins').value, []),
      missing_policy:'complete_case',
    };
    if (mode === 'preview') {
      data.operation = field('operation').value;
      data.max_depth = ['aggregate','depth_reduce'].includes(data.operation) ? Number(field('depth').value) : null;
      data.observation_policy = field('observation-policy').value;
      if (field('compare').checked) {
        data.retention = {
          predictors:list('predictors'), kind:field('kind').value,
          target:field('study-target').value.trim() || null,
          variance_variables:list('variance'), reconstruction_variables:list('reconstruction'),
          group_variable:field('holdout-group').value.trim() || null,
          acknowledge_independent_rows:field('independent').checked,
          seed:17, holdout_fraction:0.25,
        };
      }
    }
    return data;
  }
  function pretty(obj, max=16000) {
    const string = JSON.stringify(obj, null, 2);
    return string.length > max ? `${string.slice(0, max)}\n… Display truncated; export full JSON.` : string;
  }
  function linkedChip(container, id, label, removed) {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'button secondary mlwb-su10-state ' + (removed ? 'mlwb-su10-removed' : 'mlwb-su10-retained');
    chip.textContent = label;
    chip.title = `Focus original source node ${id}`;
    chip.addEventListener('click', () => originalFocus(id));
    container.append(chip);
  }
  function renderEvidence(data) {
    field('results').hidden = false;
    field('evidence').textContent = pretty({
      selected_source_node_id:data.node_id,
      observations:data.observation_evidence,
      descriptive_variance:data.uncertainty,
      empirical_entropy_MI_CMI:data.information,
      limitations:data.limitations,
    });
    const overlay = field('node-overlay'); overlay.replaceChildren();
    if (data.information?.status === 'available') {
      const node = data.information.node;
      const scope = node?.scope_variable_ids || [];
      const label = `Node ${data.node_id} · ${node?.information?.status || 'unavailable'} · ${scope.length} mapped scope columns`;
      linkedChip(overlay, data.node_id, label, false);
    }
  }
  function renderComparison(data) {
    const candidate = data.candidate;
    const compare = field('comparison');
    const removed = field('removed'); removed.replaceChildren();
    clearPreview();
    if (candidate.status !== 'available') {
      compare.textContent = `No candidate: ${candidate.reason}`; return;
    }
    const manifest = candidate.manifest;
    compare.textContent = pretty({
      operation:manifest.plan.operation, original_source_digest:manifest.source_digest,
      preview_digest:manifest.output_digest,
      structural_loss:manifest.structural_loss,
      feature_impact:manifest.feature_impact,
      observed_values_unchanged:!manifest.feature_impact.observed_cell_values_changed,
      selected_node_retained:candidate.selection_survives,
      utility_comparison:data.retention,
      notes:manifest.notes,
    }, 16000);
    const removedIDs = candidate.highlight.removed_node_ids || [];
    for (const id of removedIDs.slice(0, 60)) linkedChip(removed, id, `Removed: ${id}`, true);
    if (removedIDs.length > 60) {
      const note = document.createElement('span');
      note.textContent = `… ${removedIDs.length - 60} more removed nodes in the exported report`;
      removed.append(note);
    }
    for (const [id, into] of Object.entries(candidate.highlight.collapsed_into || {}).slice(0, 40)) {
      linkedChip(removed, into, `Collapsed: ${id} → ${into}`, false);
    }
    if (candidate.selection_survives) linkedChip(removed, data.node_id, `Retained: ${data.node_id}`, false);
    field('preview').hidden = false;
    module.validateInteractiveGraphView(candidate.view);
    preview = module.mountInteractiveGraph(field('preview-surface'), candidate.view, {
      onSelect: ({kind, id}) => {
        if (kind === 'node' && source.inspector?.nodes?.[id]) originalFocus(id);
      },
    });
    preview.fit();
  }
  async function run(mode) {
    if (busy) return;
    busy = true; action('inspect').disabled = action('preview').disabled = true;
    clearReports(); state('Running explicitly requested local analysis on verified source…');
    try {
      const body = payload(mode);
      const response = await fetch(panel.dataset.sandboxUrl, {
        method:'POST', cache:'no-store', credentials:'same-origin',
        headers:{'Content-Type':'application/json', 'Accept':'application/json',
                 'X-ML-Lab-Sandbox':'explicit-analysis'},
        body:JSON.stringify(body),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `Analysis unavailable (HTTP ${response.status})`);
      if (data.source_revision !== source?.view.source_revision || data.node_id !== selected)
        throw new Error('Graph selection changed during analysis; discard stale report.');
      result = data;
      renderEvidence(data);
      root.dispatchEvent(new CustomEvent('mlwb:sandbox-overlay', {detail:data}));
      if (mode === 'preview') renderComparison(data);
      else field('comparison').textContent = 'Candidate preview not requested; no transformation attempted.';
      action('export').disabled = false;
      state('Evidence computed from verified aligned records. Original source and graph remain unchanged.');
    } catch (error) { state(`Sandbox unavailable: ${error.message}`); }
    finally {
      busy = false;
      action('inspect').disabled = action('preview').disabled = !selected;
    }
  }
  action('inspect').addEventListener('click', () => run('inspect'));
  action('preview').addEventListener('click', () => run('preview'));
  action('export').addEventListener('click', () => {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], {type:'application/json'});
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url; anchor.download = 'ml_lab_structural_sandbox_report.json';
    anchor.click(); URL.revokeObjectURL(url);
  });
})();
