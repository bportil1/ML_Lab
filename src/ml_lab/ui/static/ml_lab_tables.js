/* ML-UW-26: pyPIQUE-style table controls, reused for every ML_Lab data grid.
   The source-table server pager and matrix's selectable cells retain their own
   behavior; all share one CSS grid presentation. Never mutate source data. */
(() => {
  const LIMITS = ['25', '50', '100', '250', 'all'];
  const state = new WeakMap();
  const cellText = (cell) => (cell?.innerText || cell?.textContent || '').trim();
  const rowsOf = (table) => Array.from(table.tBodies[0]?.rows || []);
  const csvEscape = (value) => {
    // Spreadsheet applications interpret leading formula characters as code.
    const safe = /^[\s]*[=+@-]/.test(value) ? "'" + value : value;
    return '"' + safe.replace(/"/g, '""') + '"';
  };
  const numericKey = (v) => {
    const stripped = v.replace(/[%, $]/g, '');
    return stripped && Number.isFinite(Number(stripped)) ? Number(stripped) : null;
  };
  const sortCompare = (a, b) => {
    const na = numericKey(a), nb = numericKey(b);
    if (na !== null && nb !== null) return na - nb;
    return a.localeCompare(b, undefined, {numeric: true, sensitivity: 'base'});
  };
  function download(name, type, content) {
    const url = URL.createObjectURL(new Blob([content], {type}));
    const anchor = document.createElement('a');
    anchor.href = url; anchor.download = name;
    document.body.append(anchor); anchor.click(); anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }
  function render(table) {
    const s = state.get(table);
    if (!s) return;
    const all = rowsOf(table);
    // Re-read cell content on every user action: editable cells and generated
    // rows should never require a second table engine.
    const matching = all.filter(row => {
      const values = Array.from(row.cells).map(cellText);
      return (!s.search || values.some(v => v.toLowerCase().includes(s.search))) &&
        s.filters.every((filter, index) => !filter ||
          (s.filterModes[index] === 'exact'
            ? (values[index] || '').toLowerCase() === filter
            : (values[index] || '').toLowerCase().includes(filter)));
    });
    if (s.column !== null) {
      matching.sort((a, b) => {
        const d = sortCompare(cellText(a.cells[s.column]), cellText(b.cells[s.column]));
        return (s.direction === 'desc' ? -d : d) || s.order.get(a) - s.order.get(b);
      });
    } else matching.sort((a,b) => s.order.get(a) - s.order.get(b));
    const size = s.pageSize === 'all' ? Math.max(1, matching.length) : Number(s.pageSize);
    const pages = Math.max(1, Math.ceil(matching.length / size));
    s.page = Math.min(s.page, pages);
    const start = (s.page - 1) * size;
    const visible = matching.slice(start, start + size);
    const visibleSet = new Set(visible);
    // Do not detach/reinsert rows just to filter. For sorted tables, reorder
    // matching rows once (no observer is attached to existing row changes).
    if (s.column !== null) {
      const fragment = document.createDocumentFragment();
      matching.forEach(r => fragment.append(r));
      all.filter(r => !visibleSet.has(r) && !matching.includes(r)).forEach(r => fragment.append(r));
      table.tBodies[0].append(fragment);
    }
    all.forEach(row => {row.hidden = !visibleSet.has(row);});
    s.count.textContent = matching.length ?
      `Showing ${start + 1}–${start + visible.length} of ${matching.length} rows` : 'No matching rows';
    s.pageLabel.textContent = `Page ${s.page} of ${pages}`;
    s.prev.disabled = s.page <= 1;
    s.next.disabled = s.page >= pages;
    s.headings.forEach((h, i) => h.setAttribute('aria-sort', s.column === i
      ? (s.direction === 'asc' ? 'ascending' : 'descending') : 'none'));
    s.visible = visible;
  }
  function enhance(table) {
    if (table.dataset.mlwbGridReady || !table.tHead?.rows.length || !table.tBodies.length) return;
    table.classList.add('mlwb-grid');
    // A pairwise matrix is a scientific visualization, not a data-frame. The
    // server-paged source viewer already owns filtering/pagination controls.
    if (table.classList.contains('source-table') || table.classList.contains('mlwb-dependency-matrix') ||
        table.dataset.mlwbGridPassive === 'true') {
      table.dataset.mlwbGridReady = 'passive'; return;
    }
    table.dataset.mlwbGridReady = 'true';
    const originalRows = rowsOf(table);
    const headingCells = Array.from(table.tHead.rows[0].cells);
    if (!headingCells.length || originalRows.some(r => r.cells.length !== headingCells.length)) {
      table.dataset.mlwbGridReady = 'passive'; return;
    }
    const s = {search: '', filters: headingCells.map(() => ''), filterModes: headingCells.map(() => 'contains'),
      column: null, direction: 'asc', page: 1, pageSize: table.dataset.pageSize || '50',
      order: new Map(originalRows.map((r,i) => [r,i])), headings: headingCells};
    const wrap = table.closest('.table-wrap') || table.parentNode;
    const bar = document.createElement('div'); bar.className = 'mlwb-grid-toolbar';
    const search = document.createElement('input'); search.type='search';
    search.placeholder='Search all columns';search.setAttribute('aria-label','Search all table columns');
    const size = document.createElement('select');size.setAttribute('aria-label','Rows per page');
    LIMITS.forEach(value => {const op=document.createElement('option');op.value=value;
      op.textContent=value==='all'?'All rows':value+' rows';size.append(op);});
    size.value=LIMITS.includes(s.pageSize)?s.pageSize:'50';s.pageSize=size.value;
    const clear = document.createElement('button'); clear.type='button';clear.textContent='Clear filters / sort';
    const copy = document.createElement('button');copy.type='button';copy.textContent='Copy visible';
    const exportButton=document.createElement('button');exportButton.type='button';
    exportButton.textContent='Export visible CSV';
    const count=document.createElement('span');count.className='mlwb-grid-count';
    bar.append(search, size, clear, copy, exportButton, count);
    wrap.before(bar);
    const filterRow = document.createElement('tr');filterRow.className='mlwb-grid-filter-row';
    headingCells.forEach((heading, index) => {
      heading.dataset.mlwbSort = 'true'; heading.tabIndex=0; heading.setAttribute('role','button');
      const updateSort=()=>{if(s.column===index && s.direction==='asc')s.direction='desc';
        else if(s.column===index && s.direction==='desc')s.column=null;
        else{s.column=index;s.direction='asc';}s.page=1;render(table);};
      heading.addEventListener('click', e => {if(e.target.closest('button,a,input,select'))return;updateSort();});
      heading.addEventListener('keydown', e => {if(e.key==='Enter'||e.key===' '){e.preventDefault();updateSort();}});
      const fieldCell=document.createElement('th');
      const unique = new Map();
      originalRows.forEach(row => {const v=cellText(row.cells[index]);unique.set(v,(unique.get(v)||0)+1);});
      const choices=Array.from(unique).filter(([v])=>v.length<=80);
      if(choices.length>1 && choices.length<=24 && choices.length===unique.size) {
        const picker=document.createElement('select');picker.setAttribute('aria-label',`Filter ${cellText(heading)}`);
        const allOption=document.createElement('option');allOption.value='';allOption.textContent='All';picker.append(allOption);
        choices.forEach(([value,n])=>{const option=document.createElement('option');option.value=value;option.textContent=`${value} (${n})`;picker.append(option);});
        s.filterModes[index]='exact';
        picker.addEventListener('change',()=>{s.filters[index]=picker.value.toLowerCase();s.page=1;render(table);});
        fieldCell.append(picker);
      } else {
        const field=document.createElement('input');field.type='search';field.placeholder='Filter';
        field.setAttribute('aria-label',`Filter ${cellText(heading)}`);
        field.addEventListener('input',()=>{s.filters[index]=field.value.toLowerCase();s.page=1;render(table);});
        fieldCell.append(field);
      }
      filterRow.append(fieldCell);
    });
    table.tHead.append(filterRow);
    const pagination=document.createElement('div');pagination.className='mlwb-grid-pagination';
    const prev=document.createElement('button');prev.type='button';prev.textContent='Previous';
    const pageLabel=document.createElement('span');
    const next=document.createElement('button');next.type='button';next.textContent='Next';
    pagination.append(prev,pageLabel,next);wrap.after(pagination);
    Object.assign(s,{count,prev,next,pageLabel});state.set(table,s);
    search.addEventListener('input',()=>{s.search=search.value.toLowerCase();s.page=1;render(table);});
    size.addEventListener('change',()=>{s.pageSize=size.value;s.page=1;render(table);});
    clear.addEventListener('click',()=>{s.search='';s.filters.fill('');s.column=null;s.page=1;
      search.value='';filterRow.querySelectorAll('select,input').forEach(f=>{f.value='';});render(table);});
    prev.addEventListener('click',()=>{s.page--;render(table);});
    next.addEventListener('click',()=>{s.page++;render(table);});
    function visibleValues() {
      return [headingCells.map(cellText), ...s.visible.map(r => Array.from(r.cells).map(cellText))];
    }
    copy.addEventListener('click',async()=>{
      const value=visibleValues().map(row=>row.join('\t')).join('\n');
      try {await navigator.clipboard.writeText(value);} catch(_) {
        const area=document.createElement('textarea');area.value=value;document.body.append(area);
        area.select();document.execCommand('copy');area.remove();
      }
    });
    exportButton.addEventListener('click',()=>{
      const content='\ufeff'+visibleValues().map(row=>row.map(csvEscape).join(',')).join('\r\n')+'\r\n';
      download('ml-lab-visible-table.csv','text/csv;charset=utf-8',content);
    });
    render(table);
  }
  const scan=()=>document.querySelectorAll('table').forEach(enhance);
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',scan);
  else scan();
  // Some XML and experiment viewers create tables after the initial load.
  // Register new grids only; never rerun the initializer on existing tables.
  const observer = new MutationObserver(mutations=>{
    if(mutations.some(m=>Array.from(m.addedNodes).some(n=>
      n.nodeType===1 && (n.matches?.('table') || n.querySelector?.('table')))))scan();
  });
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',()=>observer.observe(document.body,{childList:true,subtree:true}));
  else observer.observe(document.body,{childList:true,subtree:true});
})();
