/* HIA-08: opt-in empirical information between mapped levels, prior levels and
 * an optional target. Keeps the original PAH_Shared graph intact. No guessed
 * node/column bindings, automatic work, probabilistic edge or factorization.
 */
(() => {
  'use strict';
  const root = document.querySelector('[data-mlwb-shared-graph]');
  const panel = root?.querySelector('[data-hia-conditional]');
  if (!root || !panel) return;
  const $ = name => panel.querySelector(`[data-hia-conditional-${name}]`);
  const action = name => panel.querySelector(`[data-hia-conditional-action="${name}"]`);
  const say = value => { $('status').textContent = value; };
  const memberships = {level:new Set()};
  const records = {level:[], prior:[], target:null};
  let graph = null, depth = null, report = null, epoch = 0;

  const usable = () => depth !== null && records.level.length && (depth === 0 || records.prior.length);
  function invalidate() {
    epoch++; report = null; $('results').hidden = true;
    $('summary').replaceChildren(); $('table').replaceChildren(); $('report').textContent = '';
    action('compute').disabled = !usable(); action('export').disabled = true;
  }
  function reset() {
    depth = null; graph = null; memberships.level.clear();
    records.level = []; records.prior = []; records.target = null;
    panel.hidden = true; $('depth').textContent = 'None'; $('node').value = '';
    $('coverage').textContent = ''; $('mappings').replaceChildren();
    $('column').replaceChildren(); invalidate();
    say('Select a structural level and explicitly map level/prior variables.');
  }
  function updateMappings() {
    const display = $('mappings'); display.replaceChildren();
    const all = [...records.level.map(x=>['Level',x]),
                 ...records.prior.map(x=>['Prior',x]),
                 ...(records.target ? [['Target',records.target]] : [])];
    for (const [role,item] of all) {
      const row = document.createElement('div'); row.className = 'mlwb-hia-neighbor-mapping';
      const desc = document.createElement('span');
      desc.textContent = `${role}: ${item.node_id} → ${item.variable_id}`;
      const remove = document.createElement('button'); remove.type='button';
      remove.className='button secondary'; remove.textContent='Remove';
      remove.addEventListener('click', () => {
        if (role === 'Target') records.target = null;
        else {
          const list = role === 'Level' ? records.level : records.prior;
          list.splice(list.indexOf(item),1);
        }
        updateMappings(); invalidate();
      });
      row.append(desc,remove); display.append(row);
    }
    if (depth !== null) $('coverage').textContent =
      `Level ${depth}: ${memberships.level.size} source nodes (${records.level.length} mapped), ` +
      `${records.prior.length} explicitly mapped earlier-level nodes. ` +
      'Statistics describe mapped subsets only.';
  }
  root.addEventListener('mlwb:source-reset', reset);
  root.addEventListener('mlwb:source-open', event => {
    reset(); graph = event.detail?.payload || null;
    const first = document.createElement('option'); first.value='';
    first.textContent='Select observed column…'; $('column').append(first);
    for (const name of graph?.binding_contract?.observed_columns || []) {
      const option = document.createElement('option');
      option.value=name; option.textContent=name; $('column').append(option);
    }
  });
  root.addEventListener('mlwb:hia-scope', event => {
    const detail=event.detail || {};
    const selected=detail.selection;
    if (!graph || selected?.scope?.kind !== 'level' ||
        detail.source_revision !== graph.view.source_revision ||
        detail.source_digest !== graph.binding_contract?.source_digest) {
      depth=null; panel.hidden=true; memberships.level.clear();
      records.level=[]; records.prior=[];
      records.target=null; updateMappings(); invalidate(); return;
    }
    const next=selected.scope.level;
    if (!Number.isSafeInteger(next) || next < 0 || !Array.isArray(selected.node_ids)) return;
    if (next===depth && selected.node_ids.length===memberships.level.size) return;
    depth=next; memberships.level=new Set(selected.node_ids);
    // Earlier levels are validated against exact source depth *server-side*;
    // the client never infers an ancestor from a visual edge.
    records.level=[]; records.prior=[];
    records.target=null; $('node').value=''; $('depth').textContent=String(depth);
    panel.hidden=false; updateMappings(); invalidate();
    say('Add exact level members and prior-level variables, then press Compute.');
  });
  root.addEventListener('click', event => {
    const id=event.target.closest('[data-hia-member]')?.dataset.hiaMember;
    if (id && depth !== null && memberships.level.has(id)) {
      $('node').value=id; $('role').value='level';
    }
  });
  action('add').addEventListener('click', () => {
    const role=$('role').value, id=$('node').value.trim(), name=$('column').value;
    if (!id || !name) { say('Exact source node ID and observed column are required.'); return; }
    if (role==='level' && !memberships.level.has(id)) {
      say('The selected-level node must be an exact member of this level.'); return;
    }
    if (role==='prior' && (depth===0 || memberships.level.has(id))) {
      say('A preceding-level node must be outside this level; the server will verify its exact depth.'); return;
    }
    if (role==='target' && memberships.level.has(id)) {
      say('Target must be outside the selected level.'); return;
    }
    const all=[...records.level,...records.prior,...(records.target?[records.target]:[])];
    if (all.some(x=>x.node_id===id || x.variable_id===name)) {
      say('Each source node and observed column may be assigned only once.'); return;
    }
    if (role==='target') records.target={node_id:id,variable_id:name};
    else records[role].push({node_id:id,variable_id:name});
    $('node').value=''; updateMappings(); invalidate();
    say('Mapping declared. No statistical estimate has run.');
  });
  for (const name of ['occurrences','bins','min-samples','min-stratum'])
    $(name).addEventListener('change',invalidate);
  const metric=(label,value) => {
    const box=document.createElement('div'); box.className='mlwb-hia-branch-metric';
    const heading=document.createElement('span'); heading.textContent=label;
    const number=document.createElement('strong');
    number.textContent=typeof value==='number' && Number.isFinite(value)
      ? `${value.toFixed(5)} bits` : 'Unavailable';
    box.append(heading,number); return box;
  };
  action('export').addEventListener('click', () => {
    if (!report) return;
    const url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));
    const link=document.createElement('a'); link.href=url;
    link.download=`ml-lab-hia-level-${depth}-conditional.json`;
    document.body.append(link); link.click(); link.remove();
    setTimeout(()=>URL.revokeObjectURL(url),5000);
  });
  action('compute').addEventListener('click', async () => {
    if (!graph || !usable()) return;
    let bins;
    try {
      bins=JSON.parse($('bins').value || '{}');
      if (!bins || Array.isArray(bins) || typeof bins!=='object')
        throw new Error('Expected a JSON object');
    } catch (error) { say(`Invalid discretization: ${error.message}`); return; }
    const minimum=Number($('min-samples').value);
    const stratum=Number($('min-stratum').value);
    if (!Number.isSafeInteger(minimum) || minimum<2 ||
        !Number.isSafeInteger(stratum) || stratum<1) {
      say('Minimum complete rows must be ≥2 and stratum support ≥1 (integers).'); return;
    }
    const token=++epoch, revision=graph.view.source_revision, selectedLevel=depth;
    action('compute').disabled=true; $('results').hidden=true; action('export').disabled=true;
    say('Estimating joint empirical conditional information on aligned observations…');
    try {
      const response=await fetch(root.dataset.hiaConditionalUrl,{
        method:'POST',credentials:'same-origin',cache:'no-store',
        headers:{'Content-Type':'application/json','Accept':'application/json',
                 'X-ML-Lab-HIA':'explicit-conditional-level-information'},
        body:JSON.stringify({source_revision:revision,level:selectedLevel,
          mappings:records.level,preceding_mappings:records.prior,target:records.target,
          occurrence_policy:$('occurrences').value,bins,min_samples:minimum,
          min_stratum_samples:stratum}),
      });
      const data=await response.json();
      if (!response.ok) throw new Error(data.error || `Conditional analysis failed (${response.status})`);
      if (token!==epoch || !graph || selectedLevel!==depth ||
          revision!==graph.view.source_revision || data.source_revision!==revision ||
          data.report?.source_digest!==graph.binding_contract?.source_digest) return;
      report=data.report;
      const base=report.level_conditioned_information;
      const summary=$('summary'); summary.replaceChildren();
      const coverage=document.createElement('p');
      coverage.textContent=`Level ${depth} · ${report.mapped_level_node_count}/${report.level_node_count} mapped level nodes · `+
        `${report.mapped_preceding_node_count}/${report.preceding_level_node_count} mapped prior nodes · `+
        `${base.support}/${report.aligned_rows} complete level+prior rows · ${base.status}`;
      summary.append(coverage,
        metric('H(L | mapped prior levels)',base.level_conditional_entropy_bits),
        metric('I(L; mapped prior levels)',base.level_prior_mutual_information_bits),
        metric('Empirical TC(L | mapped prior levels)',base.conditional_total_correlation_bits));
      if (report.target_increment_information) {
        const target=report.target_increment_information;
        const note=document.createElement('p');
        note.textContent=`Optional target: ${target.status} on its separate ${target.support}/${report.aligned_rows} complete-case cohort. Do not subtract cross-cohort estimates.`;
        summary.append(note,
          metric('I(target; L | mapped prior levels)',target.target_incremental_information_bits),
          metric('H(target | mapped prior levels)',target.target_entropy_given_prior_bits),
          metric('H(target | L, mapped prior levels)',target.target_entropy_given_level_and_prior_bits));
      }
      $('table').replaceChildren();
      for (const item of base.per_mapped_level_node) {
        const tr=document.createElement('tr');
        for (const value of [item.node_id,item.variable_id,item.entropy_given_prior_bits,
          item.mi_with_other_level_members_given_prior_bits]) {
          const td=document.createElement('td'); td.textContent=typeof value==='number'
            ? value.toFixed(5) : value; tr.append(td);
        }
        $('table').append(tr);
      }
      $('report').textContent=JSON.stringify(report,null,2);
      $('results').hidden=false; action('export').disabled=false;
      say(`${base.status}: empirical joint conditioning only; `+
          `${report.statistical_scope}. No assumed conditional independence.`);
    } catch (error) {
      if (token===epoch) say(`Conditional information unavailable: ${error.message}`);
    } finally {
      if (token===epoch) action('compute').disabled=!usable();
    }
  });
})();
