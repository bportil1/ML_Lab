/* HIA-09: separate opt-in model comparison, reusing HIA-08's exact mappings.
 * Does not modify the PAH_Shared graph, infer parents from edges, or run on
 * selection. No analysis occurs until the explicitly acknowledged click.
 */
(() => {
  'use strict';
  const root=document.querySelector('[data-mlwb-shared-graph]');
  const controls=root?.querySelector('[data-hia-conditional]');
  const panel=controls?.querySelector('[data-hia-factorization-options]');
  if (!root || !controls || !panel) return;
  const $=name=>panel.querySelector(`[data-hia-factorization-${name}]`);
  const action=name=>panel.querySelector(`[data-hia-factorization-action="${name}"]`);
  const setting=name=>controls.querySelector(`[data-hia-conditional-${name}]`);
  const say=value=>$('status').textContent=value;
  let graph=null, selection=null, report=null, epoch=0;
  function valid() {
    return graph && selection && Number.isSafeInteger(selection.level) &&
      selection.mappings.length > 0 &&
      (selection.level === 0 || selection.preceding_mappings.length > 0) &&
      $('confirm').checked && selection.source_revision === graph.view.source_revision &&
      selection.source_digest === graph.binding_contract?.source_digest;
  }
  function invalidate() {
    epoch++; report=null; $('results').hidden=true;
    $('summary').replaceChildren(); $('table').replaceChildren(); $('report').textContent='';
    action('compute').disabled=!valid(); action('export').disabled=true;
  }
  function reset() {
    graph=null; selection=null; $('confirm').checked=false;
    panel.open=false; invalidate();
    say('Model comparison is opt-in and requires explicit confirmation.');
  }
  root.addEventListener('mlwb:source-reset',reset);
  root.addEventListener('mlwb:source-open',event=>{
    reset(); graph=event.detail?.payload || null;
  });
  root.addEventListener('mlwb:hia-conditional-mappings',event=>{
    selection=event.detail || null;
    invalidate();
  });
  for (const name of ['confirm','mode','parents']) $(name).addEventListener('change',invalidate);
  for (const name of ['occurrences','bins','min-samples','min-stratum'])
    setting(name).addEventListener('change',invalidate);
  const metric=(label,value)=>{
    const div=document.createElement('div');div.className='mlwb-hia-branch-metric';
    const caption=document.createElement('span');caption.textContent=label;
    const val=document.createElement('strong');
    val.textContent=typeof value==='number' && Number.isFinite(value)
      ? `${value.toFixed(5)} bits` : 'Unavailable';
    div.append(caption,val);return div;
  };
  action('export').addEventListener('click',()=>{
    if (!report) return;
    const url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));
    const a=document.createElement('a');a.href=url;
    a.download=`ml-lab-hia-level-${report.level}-factorization.json`;
    document.body.append(a);a.click();a.remove();
    setTimeout(()=>URL.revokeObjectURL(url),5000);
  });
  action('compute').addEventListener('click',async()=>{
    if (!valid()) {say('Declare mappings and explicitly confirm the assumption first.');return;}
    let parents,bins;
    try {
      parents=JSON.parse($('parents').value || '{}');
      bins=JSON.parse(setting('bins').value || '{}');
      if (!parents || Array.isArray(parents) || typeof parents!=='object' ||
          !bins || Array.isArray(bins) || typeof bins!=='object')
        throw new Error('Parent sets and discretization must be JSON objects.');
    } catch (error) {say(`Invalid JSON: ${error.message}`);return;}
    const minSamples=Number(setting('min-samples').value);
    const minStratum=Number(setting('min-stratum').value);
    if (!Number.isSafeInteger(minSamples) || minSamples<2 ||
        !Number.isSafeInteger(minStratum) || minStratum<1) {
      say('Minimum rows and prior stratum support must be valid integers.');return;
    }
    const current=selection, rev=current.source_revision, stamp=++epoch;
    action('compute').disabled=true;action('export').disabled=true;$('results').hidden=true;
    say('Fitting explicit factor tables and comparing them with the empirical joint cohort…');
    try {
      const response=await fetch(root.dataset.hiaFactorizationUrl,{
        method:'POST',credentials:'same-origin',cache:'no-store',
        headers:{'Content-Type':'application/json','Accept':'application/json',
                 'X-ML-Lab-HIA':'explicit-conditional-factorization'},
        body:JSON.stringify({source_revision:rev,level:current.level,
          mappings:current.mappings,preceding_mappings:current.preceding_mappings,
          target:current.target,occurrence_policy:setting('occurrences').value,
          bins,min_samples:minSamples,min_stratum_samples:minStratum,
          factorization_mode:$('mode').value,parent_sets:parents,
          assumption_confirmed:$('confirm').checked}),
      });
      const data=await response.json();
      if (!response.ok) throw new Error(data.error || `Factorization failed (${response.status})`);
      if (stamp!==epoch || selection!==current || !graph ||
          data.source_revision!==rev || rev!==graph.view.source_revision ||
          data.report?.source_digest!==graph.binding_contract?.source_digest) return;
      report=data.report;
      const model=report.model, summary=$('summary');summary.replaceChildren();
      const details=document.createElement('p');
      details.textContent=`Level ${report.level} · ${report.mapped_level_node_count}/${report.level_node_count} mapped members · `+
        `${model.support} complete cases · ${model.status} · ${report.factorization_mode} · in-sample fit`;
      summary.append(details,
        metric('Empirical H(level | full prior)',model.empirical_joint_conditional_entropy_bits),
        metric('Factorized model cross-entropy',model.factorized_cross_entropy_bits),
        metric('KL: empirical joint vs factorized model',model.empirical_model_kl_divergence_bits),
        metric('Residual within-level dependence TC(level | full prior)',model.empirical_conditional_total_correlation_bits),
        metric('Information omitted by chosen parents',model.omitted_prior_information_bits));
      $('table').replaceChildren();
      for (const item of model.per_mapped_level_node) {
        const tr=document.createElement('tr');
        for (const value of [item.node_id,item.declared_parent_node_ids.join(', ') || '(none)',
          item.model_conditional_entropy_bits,item.omitted_prior_information_bits]) {
          const td=document.createElement('td');
          td.textContent=typeof value==='number'?value.toFixed(5):value;
          tr.append(td);
        }
        $('table').append(tr);
      }
      $('report').textContent=JSON.stringify(report,null,2);
      $('results').hidden=false;action('export').disabled=false;
      say(`${model.status}: model independence is assumed, not established; `+
          'KL is in-sample, not a held-out test or causal result.');
    } catch(error) {
      if (stamp===epoch) say(`Model comparison unavailable: ${error.message}`);
    } finally {
      if (stamp===epoch) action('compute').disabled=!valid();
    }
  });
})();
