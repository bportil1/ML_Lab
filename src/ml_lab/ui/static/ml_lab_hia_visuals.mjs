/* HIA-11: scientific overlays and matched-cohort level comparisons.
 * Scientific data comes ONLY from explicit, SHA-verified HIA-04/HIA-07/HIA-10
 * reports. No estimator runs here; no source graph/view object is mutated.
 * PAH_Shared remains the only graph renderer, layout and navigation engine.
 */
const palettes = Object.freeze([
  {fill:'#eaf2fa',text:'#17324d',outline:'#b4cce0'},
  {fill:'#d4e7f4',text:'#17324d',outline:'#8ab4d2'},
  {fill:'#a5cde6',text:'#17324d',outline:'#6da5c9'},
  {fill:'#5c9dc4',text:'#102b46',outline:'#397ea9'},
  {fill:'#205e89',text:'#ffffff',outline:'#174567'},
]);
const record=x=>x!==null&&typeof x==='object'&&!Array.isArray(x);
const finite=x=>typeof x==='number'&&Number.isFinite(x);
const METRICS=Object.freeze({
  network: Object.freeze({
    level_joint_entropy_bits:'Level H(L), bits',
    incremental_joint_entropy_given_prior_bits:'Level H(L | earlier), bits',
    information_shared_with_prior_bits:'Level I(L; earlier), bits',
    within_level_total_correlation_bits:'Level TC(L), bits',
    target_incremental_information_bits:'Level I(target; L | earlier), bits',
    support:'Level matched-cohort support, rows',
  }),
  level: Object.freeze({
    entropy_bits:'Node H(X), bits',
    conditional_entropy_given_other_mapped_members_bits:'Node H(X | mapped peers), bits',
    mi_with_other_mapped_members_bits:'Node I(X; mapped peers), bits',
    support:'Level matched-cohort support, rows',
  }),
  atomic: Object.freeze({
    entropy_bits:'Node H(X), bits',
    support:'Observed scalar rows',
  }),
});
export const HIA11_METRICS=METRICS;
export function overlayRows(report,kind,metric) {
  if(!record(report) || !Object.hasOwn(METRICS,kind) || !Object.hasOwn(METRICS[kind],metric))
    throw new Error('Unsupported overlay report or metric');
  let source=[];
  if(kind==='network') {
    const info=report.network_information;
    if(!record(info) || info.status!=='ok' || !Array.isArray(info.levels)) return [];
    for(const level of info.levels) {
      const value=metric==='support'?info.support:level[metric];
      if(!finite(value))continue;
      // These are LEVEL aggregates, repeated on mapped members only for
      // visualization. They must NEVER be represented as atomic statistics.
      for(const id of level.mapped_node_ids??[]) source.push({node_id:id,level:level.level,
        variable_id:null,value,metric,scope:'level-aggregate',support:info.support});
    }
  } else if(kind==='level') {
    const info=report.level_information;
    if(!record(info)||info.status!=='ok'||!Array.isArray(info.per_mapped_node))return [];
    for(const item of info.per_mapped_node){
      const value=metric==='support'?info.support:item[metric];
      if(!finite(value))continue;
      source.push({node_id:item.node_id,variable_id:item.variable_id,level:report.level,
        value,metric,scope:'atomic-observation',support:info.support});
    }
  } else {
    const value=metric==='support'?report.support?.observed_scalar_rows:
      report.entropy?.status==='ok'?report.entropy.bits:null;
    if(finite(value))source=[{node_id:report.node_id,variable_id:report.variable_id,
      level:report.node_context?.level??null,value,metric,
      scope:'atomic-observation',support:report.support?.observed_scalar_rows??null}];
  }
  const seen=new Set();
  for(const row of source) {
    if(typeof row.node_id!=='string'||seen.has(row.node_id)||!finite(row.value)||row.value<0)
      throw new Error('Invalid or ambiguous measured overlay node');
    seen.add(row.node_id);
  }
  return source;
}
export function metricTone(value,maximum) {
  if(!finite(value)||value<0||!finite(maximum)||maximum<0)return null;
  const ratio=maximum===0?0:Math.min(1,value/maximum);
  return palettes[Math.min(4,Math.floor(ratio*4.999))];
}
export function compareMatchedLevels(report,first,second) {
  const info=report?.network_information;
  if(!record(info)||info.status!=='ok'||!Array.isArray(info.levels))
    throw new Error('A successful HIA-10 matched-cohort report is required');
  if(!Number.isSafeInteger(first)||!Number.isSafeInteger(second)||first===second)
    throw new Error('Choose two distinct existing structural levels');
  const left=info.levels.find(x=>x.level===first);
  const right=info.levels.find(x=>x.level===second);
  if(!left||!right)throw new Error('Selected level is not in the measured subset');
  const pair=(info.cross_level_pairs||[]).find(x=>
    (x.left_level===first&&x.right_level===second)||
    (x.left_level===second&&x.right_level===first));
  return {schema:'ml-lab.hia-level-comparison@1',source_digest:report.source_digest,
    binding_digest:report.binding_digest,source_topology:report.source_topology,
    cohort_support:info.support,first:left,second:right,
    pairwise_cross_level_mi_bits:pair?.mutual_information_bits??null,
    interpretation:'Descriptive matched-cohort comparison; pairwise MI is not additive or causal.'};
}
const quote=x=>'"'+String(x??'').replaceAll('"','""')+'"';
export function overlayCsv(rows,{source_digest='',source_revision='',basis='',metric=''}={}) {
  const keys=['source_revision','source_digest','basis','metric','node_id','variable_id',
    'structural_level','value','support','statistical_scope'];
  const lines=[keys.map(quote).join(',')];
  for(const row of rows) {
    const values=[source_revision,source_digest,basis,metric,row.node_id,
      row.variable_id,row.level,row.value,row.support,row.scope];
    lines.push(values.map(quote).join(','));
  }
  return lines.join('\r\n')+'\r\n';
}

if(typeof document!=='undefined') {
  const root=document.querySelector('[data-mlwb-shared-graph]');
  const panel=root?.querySelector('[data-hia-visuals]');
  if(root&&panel) {
    const $=name=>panel.querySelector(`[data-hia-visuals-${name}]`);
    const button=name=>panel.querySelector(`[data-hia-visuals-action="${name}"]`);
    const surface=root.querySelector('[data-graph-surface]');
    const state={graph:null,viewer:null,network:null,level:null,atomic:null,source:'network'};
    let rows=[],rowMap=new Map(),max=0,observer=null,scheduled=false;
    const message=s=>{$('status').textContent=s;};
    const fmt=v=>finite(v)?v.toFixed(5):'Unavailable';
    const options=()=>Array.from($('basis').options);
    function chooseAvailable() {
      for(const option of options())option.disabled=!state[option.value];
      if(!state[$('basis').value]) {
        const next=options().find(o=>!o.disabled);
        if(next)$('basis').value=next.value;
      }
      configureMetrics();
    }
    function configureMetrics() {
      const kind=$('basis').value;
      const former=$('metric').value;
      $('metric').replaceChildren();
      for(const [value,label] of Object.entries(METRICS[kind])) {
        const opt=document.createElement('option');opt.value=value;opt.textContent=label;
        $('metric').append(opt);
      }
      if(Object.hasOwn(METRICS[kind],former))$('metric').value=former;
      else $('metric').value='entropy_bits' in METRICS[kind]?'entropy_bits':
        'level_joint_entropy_bits' in METRICS[kind]?'level_joint_entropy_bits':Object.keys(METRICS[kind])[0];
    }
    function stripOverlay() {
      for(const group of surface.querySelectorAll('.pah-ig-node[data-node-id]')) {
        const bg=group.querySelector('.pah-ig-node-bg');
        const label=group.querySelector('.pah-ig-node-label');
        if(bg)bg.style.removeProperty('fill');
        if(label){label.style.removeProperty('fill');label.style.removeProperty('stroke');label.style.removeProperty('stroke-width');}
        group.removeAttribute('data-hia11-overlay');
      }
    }
    function renderOverlay() {
      scheduled=false;
      if(!state.graph||!surface)return;
      for(const group of surface.querySelectorAll('.pah-ig-node[data-node-id]')) {
        const id=group.getAttribute('data-node-id');
        const item=rowMap.get(id);
        const tone=item?metricTone(item.value,max):null;
        const bg=group.querySelector('.pah-ig-node-bg');
        const label=group.querySelector('.pah-ig-node-label');
        if(!tone) {
          if(bg)bg.style.removeProperty('fill');
          if(label){label.style.removeProperty('fill');label.style.removeProperty('stroke');label.style.removeProperty('stroke-width');}
          group.removeAttribute('data-hia11-overlay');continue;
        }
        if(bg)bg.style.fill=tone.fill;
        if(label){label.style.fill=tone.text;label.style.stroke=tone.text==='#ffffff'?'#15415f':'#ffffff';label.style.strokeWidth='2px';}
        group.setAttribute('data-hia11-overlay','measured');
      }
    }
    function schedule() {
      if(scheduled)return;
      scheduled=true;requestAnimationFrame(renderOverlay);
    }
    if(surface) {
      observer=new MutationObserver(schedule);
      observer.observe(surface,{childList:true,subtree:true});
    }
    function paint() {
      const kind=$('basis').value, report=state[kind], metric=$('metric').value;
      const candidate=report?overlayRows(report,kind,metric):[];
      const validIds=new Set(state.graph?.view.nodes?.map(node=>node.id)||[]);
      rows=candidate.filter(item=>validIds.has(item.node_id));
      rowMap=new Map(rows.map(item=>[item.node_id,item]));
      max=0;for(const item of rows)if(item.value>max)max=item.value;
      const caption=METRICS[kind][metric],levelWide=kind==='network';
      $('legend').textContent=rows.length?`${caption} · ${rows.length} explicitly mapped original nodes · ${fmt(0)} to ${fmt(max)} · `+
        `${levelWide?'One level-wide statistic repeated on that level’s mapped members':'Atomic statistics'} · `+
        'Unmapped and unmeasured nodes are neutral. Source topology is unchanged.' :
        'No eligible measured nodes for this source/metric. Unmapped nodes remain neutral.';
      const chips=$('swatches');chips.replaceChildren();
      const steps=max===0?[0]:[0,1,2,3,4];
      for(const i of steps) {
        const cell=document.createElement('span');cell.className='mlwb-hia11-swatch';
        cell.style.backgroundColor=palettes[i].fill;
        cell.style.color=palettes[i].text;
        cell.textContent=max===0?'Measured 0':`${Math.round(i*max/4*100)/100}`;
        chips.append(cell);
      }
      button('csv').disabled=rows.length===0;
      button('svg').disabled=rows.length===0||!state.viewer;
      button('png').disabled=rows.length===0||!state.viewer;
      $('scopelabel').textContent=levelWide?'Level aggregation, shown on mapped members':
        'Atomic observations, mapped nodes only';
      schedule();
    }
    function redrawComparison() {
      const report=state.network,info=report?.network_information;
      const available=info?.status==='ok' && info.levels?.length>=2;
      $('compare').hidden=!available;
      if(!available){$('comparison').replaceChildren();return;}
      const a=$('left').value,b=$('right').value;
      for(const select of [$('left'),$('right')]) {
        select.replaceChildren();
        for(const level of info.levels) {
          const opt=document.createElement('option');opt.value=String(level.level);
          opt.textContent=`Level ${level.level} · ${level.mapped_node_count}/${level.structural_node_count} mapped`;
          select.append(opt);
        }
      }
      if(info.levels.some(l=>String(l.level)===a))$('left').value=a;
      if(info.levels.some(l=>String(l.level)===b))$('right').value=b;
      if($('left').value===$('right').value)$('right').selectedIndex=1;
      renderComparison();
    }
    function renderComparison() {
      const dest=$('comparison');dest.replaceChildren();
      let compare;
      try {compare=compareMatchedLevels(state.network,Number($('left').value),Number($('right').value));}
      catch(error){dest.textContent=error.message;return;}
      const properties=[
        ['level_joint_entropy_bits','Joint entropy H(L), bits'],
        ['incremental_joint_entropy_given_prior_bits','H(L | earlier), bits'],
        ['information_shared_with_prior_bits','I(L; earlier), bits'],
        ['within_level_total_correlation_bits','Within-level TC(L), bits'],
        ['target_incremental_information_bits','Incremental target information, bits'],
        ['mapped_node_count','Mapped source nodes'],
      ];
      const table=document.createElement('table');table.className='mlwb-hia-neighbor-table';
      const head=document.createElement('thead'),tr=document.createElement('tr');
      for(const text of ['Matched-cohort measure',`Level ${compare.first.level}`,`Level ${compare.second.level}`]) {
        const cell=document.createElement('th');cell.textContent=text;tr.append(cell);
      }head.append(tr);table.append(head);
      const body=document.createElement('tbody');
      for(const [field,label] of properties) {
        const row=document.createElement('tr'),th=document.createElement('th');
        th.textContent=label;row.append(th);
        for(const part of [compare.first,compare.second]) {
          const td=document.createElement('td');td.textContent=fmt(part[field]);row.append(td);
        }body.append(row);
      }
      table.append(body);dest.append(table);
      const note=document.createElement('p');note.className='tiny';
      note.textContent=`Matched cohort: ${compare.cohort_support} complete rows · Pairwise I(L${compare.first.level}; L${compare.second.level})=${fmt(compare.pairwise_cross_level_mi_bits)} bits. `+
        compare.interpretation;dest.append(note);
    }
    const download=(name,content,type)=>{
      const url=URL.createObjectURL(new Blob([content],{type}));
      const link=document.createElement('a');link.href=url;link.download=name;
      document.body.append(link);link.click();link.remove();
      setTimeout(()=>URL.revokeObjectURL(url),5000);
    };
    function svgMarkup() {
      const source=surface.querySelector('svg');
      if(!source)throw Error('No visible graph page to export');
      // Vector export is an honest snapshot of only the CURRENT graph page.
      // Browser SVG can otherwise depend on an external stylesheet.
      const svg=source.cloneNode(true);
      svg.setAttribute('xmlns','http://www.w3.org/2000/svg');
      const style=document.createElementNS('http://www.w3.org/2000/svg','style');
      style.textContent=`.pah-ig-node-bg{fill:#fff;stroke:#6a86a2;stroke-width:1.8}
      .pah-ig-node-label{fill:#17324d;font:600 12.5px system-ui,sans-serif;paint-order:stroke;stroke:#fff;stroke-width:3px}
      .pah-ig-edge-line{fill:none;stroke:#5f7993;stroke-width:2}
      .pah-ig-arrow-head{fill:#5f7993}`;
      svg.insertBefore(style,svg.firstChild);
      const backdrop=document.createElementNS('http://www.w3.org/2000/svg','rect');
      backdrop.setAttribute('width','100%');backdrop.setAttribute('height','100%');backdrop.setAttribute('fill','#f8fbff');
      svg.insertBefore(backdrop,style.nextSibling);
      return new XMLSerializer().serializeToString(svg);
    }
    button('csv').addEventListener('click',()=>{
      if(!rows.length)return;
      download('ml-lab-hia11-scientific-overlay.csv',overlayCsv(rows,{
        source_digest:state.graph.binding_contract.source_digest,
        source_revision:state.graph.view.source_revision,
        basis:$('basis').value,metric:$('metric').value}), 'text/csv;charset=utf-8');
    });
    button('svg').addEventListener('click',()=>{
      try {download('ml-lab-hia11-visible-graph.svg',svgMarkup(),'image/svg+xml');}
      catch(error){message(error.message);}
    });
    button('png').addEventListener('click',async()=>{
      if(!rows.length)return;
      try {
        const svg=surface.querySelector('svg');if(!svg)throw Error('No graph page to export');
        const view=svg.viewBox.baseVal;
        const width=Math.max(1,Math.round(view.width||svg.clientWidth||900));
        const height=Math.max(1,Math.round(view.height||svg.clientHeight||600));
        const canvas=document.createElement('canvas');canvas.width=width;canvas.height=height;
        const ctx=canvas.getContext('2d');if(!ctx)throw Error('Canvas rendering unavailable');
        const url=URL.createObjectURL(new Blob([svgMarkup()],{type:'image/svg+xml'}));
        try {
          const image=new Image();
          await new Promise((resolve,reject)=>{image.onload=resolve;image.onerror=()=>reject(Error('SVG rasterization failed'));image.src=url;});
          ctx.fillStyle='#f8fbff';ctx.fillRect(0,0,width,height);ctx.drawImage(image,0,0,width,height);
        } finally {URL.revokeObjectURL(url);}
        const blob=await new Promise(resolve=>canvas.toBlob(resolve,'image/png'));
        if(!blob)throw Error('PNG encoding unavailable');
        const objectURL=URL.createObjectURL(blob),link=document.createElement('a');
        link.href=objectURL;link.download='ml-lab-hia11-visible-graph.png';document.body.append(link);link.click();link.remove();
        setTimeout(()=>URL.revokeObjectURL(objectURL),5000);
      }catch(error){message(`PNG export unavailable: ${error.message}`);}
    });
    for(const name of ['left','right'])$(name).addEventListener('change',renderComparison);
    $('basis').addEventListener('change',()=>{configureMetrics();paint();});
    $('metric').addEventListener('change',paint);
    button('clear').addEventListener('click',()=>{
      rows=[];rowMap.clear();max=0;stripOverlay();$('legend').textContent='Overlay cleared. Original graph preserved.';
      $('swatches').replaceChildren();button('csv').disabled=true;button('svg').disabled=true;button('png').disabled=true;
    });
    root.addEventListener('mlwb:source-reset',()=>{
      state.graph=null;state.viewer=null;state.network=null;state.level=null;state.atomic=null;
      panel.hidden=true;rowMap.clear();rows=[];max=0;stripOverlay();message('No analysis run. Open a source graph first.');
    });
    root.addEventListener('mlwb:source-open',event=>{
      state.graph=event.detail?.payload??null;state.viewer=event.detail?.viewer??null;
      state.network=null;state.level=null;state.atomic=null;panel.hidden=!state.graph;
      chooseAvailable();rows=[];rowMap.clear();max=0;stripOverlay();
      $('legend').textContent='Unmapped and unmeasured nodes remain neutral.';
      $('swatches').replaceChildren();$('compare').hidden=true;$('comparison').replaceChildren();
      for(const name of ['csv','svg','png'])button(name).disabled=true;
      message('Run HIA-04, HIA-07 or HIA-10 with explicit observation mappings before applying scientific colors.');
    });
    function accepted(report,sourceRevision) {
      return state.graph && report?.source_digest===state.graph.binding_contract?.source_digest &&
        sourceRevision===state.graph.view.source_revision;
    }
    for(const [eventName,kind] of [['mlwb:hia-network-report','network'],
                                   ['mlwb:hia-level-report','level'],
                                   ['mlwb:hia-atomic-report','atomic']]) {
      root.addEventListener(eventName,event=>{
        const {report,source_revision}=event.detail||{};
        if(!accepted(report,source_revision))return;
        state[kind]=report;$('basis').value=kind;
        chooseAvailable();paint();if(kind==='network')redrawComparison();
        message(`${kind} statistics applied to ${rows.length} mapped nodes; source graph and edges are unchanged.`);
      });
    }
    for(const [eventName,kind] of [['mlwb:hia-network-invalidated','network'],
                                   ['mlwb:hia-level-invalidated','level'],
                                   ['mlwb:hia-atomic-invalidated','atomic']]) {
      root.addEventListener(eventName,()=>{
        state[kind]=null;chooseAvailable();paint();if(kind==='network')redrawComparison();
      });
    }
    root.addEventListener('mlwb:graph-viewport',schedule);
    root.addEventListener('mlwb:graph-select',schedule);
    chooseAvailable();
  }
}
