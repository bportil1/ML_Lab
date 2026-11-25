/* HIA-10: opt-in, matched-cohort information accounting across all mapped levels.
 * No graph rendering, original IDs, or statistical data are changed.
 */
(() => {
  'use strict';
  const root=document.querySelector('[data-mlwb-shared-graph]');
  const panel=root?.querySelector('[data-hia-network]');
  if (!root || !panel) return;
  const $=name=>panel.querySelector(`[data-hia-network-${name}]`);
  const action=name=>panel.querySelector(`[data-hia-network-action="${name}"]`);
  const say=value=>{$('status').textContent=value;};
  let graph=null, report=null, epoch=0, mappings=[], target=null;
  const fmt=value=>typeof value==='number' && Number.isFinite(value) ? value.toFixed(5) : '—';
  function invalidate() {
    ++epoch; report=null; root.dispatchEvent(new CustomEvent('mlwb:hia-network-invalidated')); $('results').hidden=true;
    $('summary').replaceChildren(); $('level-table').replaceChildren();
    $('pair-table').replaceChildren(); $('report').textContent='';
    action('compute').disabled=!graph || !mappings.length;
    action('export').disabled=true;
  }
  function redraw() {
    const list=$('mappings'); list.replaceChildren();
    for (const [kind,mapping] of [...mappings.map(m=>['Mapped',m]), ...(target?[['Target',target]]:[])]) {
      const row=document.createElement('div'); row.className='mlwb-hia-neighbor-mapping';
      const name=document.createElement('span');
      name.textContent=`${kind}: ${mapping.node_id} → ${mapping.variable_id}`;
      const remove=document.createElement('button'); remove.type='button';
      remove.className='button secondary';remove.textContent='Remove';
      remove.addEventListener('click',()=>{
        if(kind==='Target') target=null;
        else mappings=mappings.filter(x=>x!==mapping);
        redraw();invalidate();
      });row.append(name,remove);list.append(row);
    }
    $('coverage').textContent=`${mappings.length} explicitly mapped source nodes`+
      (target ? ' · 1 optional target' : ' · no target');
  }
  function reset() {
    graph=null; mappings=[];target=null;panel.hidden=true;
    $('node').value=''; $('column').replaceChildren();redraw();invalidate();
    say('Select an original source graph and map observed variables.');
  }
  root.addEventListener('mlwb:source-reset',reset);
  root.addEventListener('mlwb:source-open',event=>{
    reset();graph=event.detail?.payload||null;
    for(const name of ['Select an observed column…',...(graph?.binding_contract?.observed_columns||[])]) {
      const option=document.createElement('option');
      option.value=name==='Select an observed column…'?'':name;
      option.textContent=name;$('column').append(option);
    }
    panel.hidden=!graph;
    invalidate();
  });
  // The exact ID is the source ID, never a renderer index or truncated label.
  root.addEventListener('mlwb:hia-scope',event=>{
    const info=event.detail;
    if (!graph || info?.source_revision!==graph.view.source_revision) return;
    const scope=info.selection?.scope;
    if(scope?.kind==='node' && typeof scope.node_id==='string') $('node').value=scope.node_id;
  });
  root.addEventListener('click',event=>{
    const id=event.target.closest('[data-hia-member]')?.dataset.hiaMember;
    if(id && graph) $('node').value=id;
  });
  for(const kind of ['add','target']) action(kind).addEventListener('click',()=>{
    const id=$('node').value.trim(), name=$('column').value;
    if(!id || !name) {say('Select an exact original node ID and observed column.');return;}
    if([...mappings,...(target?[target]:[])].some(x=>x.node_id===id || x.variable_id===name)) {
      say('Each source node and observed column may be used only once.');return;
    }
    const mapping={node_id:id,variable_id:name};
    if(kind==='target') {
      if(target){say('Remove the previous target before selecting another.');return;}
      target=mapping;
    } else mappings.push(mapping);
    $('node').value='';redraw();invalidate();
    say('Explicit mapping recorded; no estimate has run.');
  });
  for(const name of ['occurrences','bins','min-samples']) $(name).addEventListener('change',invalidate);
  function writeRows(destination, records, keys) {
    destination.replaceChildren();
    for(const record of records) {
      const tr=document.createElement('tr');
      for(const key of keys) {
        const cell=document.createElement('td');
        const value=key==='coverage' ? `${record.mapped_node_count}/${record.structural_node_count}` : record[key];
        cell.textContent=typeof value==='number' && !Number.isInteger(value) ? fmt(value) :
          value===null || value===undefined ? '—' : String(value);
        tr.append(cell);
      } destination.append(tr);
    }
  }
  action('export').addEventListener('click',()=>{
    if(!report)return;
    const url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));
    const link=document.createElement('a');link.href=url;
    link.download='ml-lab-hia10-network-information.json';
    document.body.append(link);link.click();link.remove();
    setTimeout(()=>URL.revokeObjectURL(url),5000);
  });
  action('compute').addEventListener('click',async()=>{
    if(!graph || !mappings.length)return;
    let bins;
    try {bins=JSON.parse($('bins').value||'{}');
      if(!bins || Array.isArray(bins) || typeof bins!=='object')throw Error('expected object');
    }catch(error){say(`Invalid discretization: ${error.message}`);return;}
    const minimum=Number($('min-samples').value);
    if(!Number.isSafeInteger(minimum) || minimum<2){say('Minimum complete rows must be ≥2.');return;}
    const revision=graph.view.source_revision,token=++epoch;
    const originalMapping=mappings.map(x=>({...x}));
    action('compute').disabled=true;action('export').disabled=true;$('results').hidden=true;
    say('Computing cumulative and cross-level information on one matched cohort…');
    try {
      const response=await fetch(root.dataset.hiaNetworkUrl,{
        method:'POST',credentials:'same-origin',cache:'no-store',
        headers:{'Content-Type':'application/json','Accept':'application/json',
                 'X-ML-Lab-HIA':'explicit-network-information-accounting'},
        body:JSON.stringify({source_revision:revision,mappings:originalMapping,target,
          occurrence_policy:$('occurrences').value,bins,min_samples:minimum}),
      });
      const data=await response.json();
      if(!response.ok)throw Error(data.error||`Network information failed (${response.status})`);
      if(token!==epoch || !graph || revision!==graph.view.source_revision ||
         data.source_revision!==revision ||
         data.report?.source_digest!==graph.binding_contract?.source_digest)return;
      report=data.report;const info=report.network_information;
      root.dispatchEvent(new CustomEvent('mlwb:hia-network-report',{detail:{report,source_revision:revision}}));
      const summary=$('summary'); summary.replaceChildren();
      const p=document.createElement('p');
      p.textContent=`${info.status} · ${info.support}/${report.aligned_rows} complete rows · `+
        `${report.mapped_node_count}/${report.structural_node_count} mapped nodes · `+
        `H(mapped network)=${fmt(info.joint_entropy_bits)} bits · `+
        `TC=${fmt(info.total_correlation_bits)} bits`;
      summary.append(p);
      writeRows($('level-table'),info.levels,[
        'level','coverage','level_joint_entropy_bits',
        'incremental_joint_entropy_given_prior_bits','information_shared_with_prior_bits',
        'target_cumulative_mutual_information_bits','target_incremental_information_bits']);
      writeRows($('pair-table'),info.cross_level_pairs,[
        'left_level','right_level','mutual_information_bits','support']);
      $('report').textContent=JSON.stringify(report,null,2);
      $('results').hidden=false;action('export').disabled=false;
      say(info.status==='ok'?'Whole-network information computed. Pairwise MI is not additive.':
        `Information unavailable: ${info.status}. Inspect the complete report.`);
    } catch(error){if(token===epoch)say(error.message);}finally{
      if(token===epoch)action('compute').disabled=!mappings.length;
    }
  });
  reset();
})();
