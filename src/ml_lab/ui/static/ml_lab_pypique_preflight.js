/* Read-only managed artifact preflight, modeled on pyPIQUE's workflow status.
   URLs are supplied by Flask's url_for; never construct a mount prefix here. */
(() => {
  const panel = document.querySelector('[data-mlwb-preflight-url]');
  if (!panel || typeof fetch !== 'function') return;
  const state = panel.querySelector('.mlwb-preflight-state');
  const steps = panel.querySelector('.mlwb-preflight-steps');
  const reuse = panel.querySelector('.mlwb-preflight-list');
  const datasetPanel = document.querySelector('[data-mlwb-dataset-summary]');
  const datasetName = datasetPanel && datasetPanel.querySelector('[data-mlwb-dataset-name]');
  const datasetDetail = datasetPanel && datasetPanel.querySelector('[data-mlwb-dataset-detail]');
  const stages = new Map(Array.from(document.querySelectorAll('[data-mlwb-stage]'))
    .filter(node => node.dataset.mlwbStage).map(node => [node.dataset.mlwbStage, node]));
  let requestSequence = 0;

  // Status comes exclusively from the read-only, instance-scoped preflight API.
  // This presentation layer never guesses that a model ran or a stage completed.
  const knownStates = new Set(['complete', 'ready', 'blocked']);
  function sourceName(path) {
    return String(path || '').split(/[\\/]/).filter(Boolean).pop() || 'Selected dataset';
  }
  function displayStage(step, data) {
    const node = stages.get(step.id);
    if (!node) return;
    const summary = node.querySelector('[data-mlwb-step-summary]');
    const hasDataset = !!(data.active && data.active_status === 'ready');
    const stale = data.active_status === 'stale';
    const originalState = knownStates.has(step.state) ? step.state : 'unknown';
    // Data itself can always create/select a source. General ML Tools also
    // contains independent synthetic tools, even without selected tabular data.
    const displayedState = stale ? 'blocked' :
      (!hasDataset && step.id === 'data' && originalState === 'blocked') ? 'ready' :
      (!hasDataset && step.id === 'tools' && originalState === 'blocked') ? 'partial' :
      originalState;
    node.dataset.state = displayedState;
    let description = 'Readiness unavailable';
    if (stale) {
      description = step.id === 'data' ? 'Source changed · reselect dataset' :
        'Reselect changed dataset in Data';
    } else if (hasDataset) {
      const messages = {
        data: `Selected · ${sourceName(data.active.path)}`,
        profiling: originalState === 'complete' ? 'Saved profile available for this dataset' :
          'Dataset ready · run profiling',
        representation: 'Dataset ready · choose a method',
        comparison: 'Dataset ready · choose a comparison',
        tools: 'Dataset ready · configure tools',
      };
      description = messages[step.id] || description;
    } else if (step.id === 'data') {
      description = originalState === 'ready' ? 'Saved data found · select in Data' :
        'Ready · create or select a dataset';
    } else if (step.id === 'tools') {
      description = 'Data-linked tools need dataset';
    } else if (knownStates.has(originalState)) {
      description = 'Select a dataset in Data first';
    }
    summary.textContent = description;
    summary.title = description;
  }
  function displayDataset(data) {
    if (!datasetPanel || !datasetName || !datasetDetail) return;
    const active = data.active_status === 'ready' ? data.active : null;
    if (active) {
      const name = sourceName(active.path);
      datasetPanel.dataset.state = 'complete';
      datasetName.textContent = name;
      datasetName.title = active.path || name;
      datasetDetail.textContent = `${active.rows} rows · ${active.columns} columns · verified selection`;
    } else if (data.active_status === 'stale') {
      datasetPanel.dataset.state = 'blocked';
      datasetName.textContent = 'Selected source changed';
      datasetName.removeAttribute('title');
      datasetDetail.textContent = 'Reselect a valid dataset in Data';
    } else {
      const reusable = (data.saved || []).some(item => ['ready', 'found'].includes(item.status)) ||
        (data.derived || []).length > 0;
      datasetPanel.dataset.state = reusable ? 'ready' : 'unknown';
      datasetName.textContent = 'No active dataset';
      datasetName.removeAttribute('title');
      datasetDetail.textContent = reusable ? 'Saved data available · select in Data' :
        'Choose a dataset in Data to bind the workflow';
    }
  }
  function displayUnavailable() {
    if (datasetPanel && datasetName && datasetDetail) {
      datasetPanel.dataset.state = 'unknown';
      datasetName.textContent = 'Selection status unavailable';
      datasetName.removeAttribute('title');
      datasetDetail.textContent = 'Reopen a page to retry the read-only check';
    }
    stages.forEach(node => {
      node.dataset.state = 'unknown';
      const summary = node.querySelector('[data-mlwb-step-summary]');
      summary.textContent = 'Readiness check unavailable';
      summary.title = summary.textContent;
    });
  }
  const stepLabel = {complete:'Existing dataset active',ready:'Ready to use',blocked:'Needs dataset'};
  function show(data) {
    steps.replaceChildren();reuse.replaceChildren();
    displayDataset(data);
    (data.steps || []).forEach(step => displayStage(step, data));
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
        displayUnavailable();
      }
    }
  }
  refresh();
  window.addEventListener('pageshow',refresh);
  document.addEventListener('visibilitychange',()=>{if(document.visibilityState==='visible')refresh();});
})();
