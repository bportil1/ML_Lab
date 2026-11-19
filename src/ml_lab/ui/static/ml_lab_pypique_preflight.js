/* Read-only managed artifact preflight, modeled on pyPIQUE's workflow status.
   URLs are supplied by Flask's url_for; never construct a mount prefix here. */
(() => {
  const panel = document.querySelector('[data-mlwb-preflight-url]');
  if (!panel || typeof fetch !== 'function') return;
  const state = panel.querySelector('.mlwb-preflight-state');
  const steps = panel.querySelector('.mlwb-preflight-steps');
  const reuse = panel.querySelector('.mlwb-preflight-list');
  let requestSequence = 0;
  const stepLabel = {complete:'Existing dataset active',ready:'Ready to use',blocked:'Needs dataset'};
  function show(data) {
    steps.replaceChildren();reuse.replaceChildren();
    state.textContent = data.active
      ? `Active: ${data.active.path} · ${data.active.rows} rows`
      : 'No active dataset — checking saved work';
    (data.steps || []).forEach(step => {
      const line = document.createElement('div');line.className='mlwb-preflight-step';
      const led=document.createElement('span');
      led.className=`mlwb-preflight-led ${['complete','ready','blocked'].includes(step.state)?step.state:'unknown'}`;
      led.setAttribute('aria-hidden','true');
      const label=document.createElement('span');label.textContent=`${step.label} · ${stepLabel[step.state]||'Unknown'}`;
      line.append(led,label);steps.append(line);
    });
    const available=(data.saved||[]).filter(item=>['ready','found'].includes(item.status) && typeof item.url==='string');
    const derived=(data.derived||[]).filter(item=>typeof item.url==='string');
    if(available.length||derived.length){
      const heading=document.createElement('p');heading.className='mlwb-preflight-note';
      heading.textContent='Reuse existing outputs (no recomputation):';reuse.append(heading);
      available.forEach(item=>{
        const link=document.createElement('a');link.href=item.url;
        link.textContent=item.status==='ready' ? `${item.name} · rev ${item.revision} · ${item.rows} rows` : `Verify saved: ${item.name} · rev ${item.revision}`;
        link.title=item.path;reuse.append(link);
      });
      derived.forEach(item=>{
        const link=document.createElement('a');link.href=item.url;
        link.textContent=`Inspect derived table: ${item.path}`;
        link.title='Detected on disk; verify before activation';reuse.append(link);
      });
    }
    if(!data.active && !available.length && !derived.length){
      const note=document.createElement('p');note.className='mlwb-preflight-note';
      note.textContent='No saved dataset found. Use Data to create or select one.';reuse.append(note);
    }
    (data.recent_runs||[]).filter(item=>typeof item.url==='string').forEach(item=>{
      const link=document.createElement('a');link.href=item.url;
      link.textContent=`Recorded run · ${item.status} · rev ${item.revision}`;
      link.title='Open saved run status and verified output artifacts';reuse.append(link);
    });
    const unavailable=(data.saved||[]).filter(item=>item.status==='stale').length;
    if(unavailable){
      const warning=document.createElement('p');warning.className='mlwb-preflight-note';
      warning.textContent=`${unavailable} saved dataset(s) could not be verified and are not offered for reuse.`;
      reuse.append(warning);
    }
  }
  async function refresh(){
    if(document.visibilityState==='hidden')return;
    const seq=++requestSequence;
    try {
      const response=await fetch(panel.dataset.mlwbPreflightUrl,
        {credentials:'same-origin',cache:'no-store',headers:{Accept:'application/json'}});
      if(!response.ok)throw new Error(`HTTP ${response.status}`);
      const data=await response.json();
      if(seq===requestSequence)show(data);
    } catch(error) {
      if(seq===requestSequence){
        state.textContent='Saved-work precheck unavailable; existing work was not changed.';
        steps.replaceChildren();reuse.replaceChildren();
      }
    }
  }
  refresh();
  window.addEventListener('pageshow',refresh);
  document.addEventListener('visibilitychange',()=>{if(document.visibilityState==='visible')refresh();});
})();
