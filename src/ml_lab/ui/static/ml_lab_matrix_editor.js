(() => {
  const root = document.getElementById('matrix-draft');
  if (!root) return;
  const $ = (id) => document.getElementById(id);
  const head = $('matrix-head'), body = $('matrix-body'), note = $('matrix-message');
  const sizeSelect = $('matrix-page-size'), countLabel = $('matrix-pending');
  const undoButton = $('matrix-undo'), redoButton = $('matrix-redo'), saveButton = $('matrix-save');
  const prevButton = $('matrix-prev'), nextButton = $('matrix-next'), copyButton = $('matrix-copy');
  const resultBox = $('matrix-result');
  const dtypes = JSON.parse(root.dataset.dtypes || '[]');
  const sourceHash = root.dataset.sourceHash;
  const pending = new Map(), original = new Map(), history = [], redo = [];
  const state = {page: 1, pageCount: 1, first: 0, columns: [], rows: [], anchor: null, active: null, end: null, editing: false, loading: false, saved: false};
  let requestNumber = 0, dragging = false;
  const key = (row, col) => `${row}:${col}`;
  const message = (value) => { note.textContent = value; };
  const range = () => {
    if (!state.anchor || !state.end) return null;
    return {r1: Math.min(state.anchor.row, state.end.row), r2: Math.max(state.anchor.row, state.end.row),
      c1: Math.min(state.anchor.col, state.end.col), c2: Math.max(state.anchor.col, state.end.col)};
  };
  const cellValue = (row, col) => pending.has(key(row, col)) ? pending.get(key(row, col)) : original.get(key(row, col));
  const visibleCell = (row, col) => body.querySelector(`td[data-row="${row}"][data-col="${col}"]`);
  const withinPage = (row) => row >= state.first && row < state.first + state.rows.length;

  function updateButtons() {
    countLabel.textContent = state.saved ? `${pending.size} edits saved to a derived dataset` : (pending.size ? `${pending.size} unsaved cell edit${pending.size === 1 ? '' : 's'}` : 'No draft edits');
    undoButton.disabled = state.saved || !history.length;
    redoButton.disabled = state.saved || !redo.length;
    saveButton.disabled = state.saved || !pending.size || state.loading;
  }
  function renderValues() {
    body.querySelectorAll('td[data-row]').forEach((cell) => {
      const row = Number(cell.dataset.row), col = Number(cell.dataset.col);
      if (cell.querySelector('input')) return;
      const value = cellValue(row, col) ?? '';
      cell.textContent = value;
      cell.title = value;
      cell.classList.toggle('matrix-changed', pending.has(key(row, col)));
      cell.classList.toggle('matrix-missing', value === '');
    });
    updateButtons();
  }
  function refreshSelection() {
    const selected = range();
    body.querySelectorAll('td[data-row]').forEach(cell => {
      const row = Number(cell.dataset.row), col = Number(cell.dataset.col);
      cell.classList.toggle('matrix-selected', !!selected && row >= selected.r1 && row <= selected.r2 && col >= selected.c1 && col <= selected.c2);
      cell.classList.toggle('matrix-active', !!state.active && row === state.active.row && col === state.active.col);
      cell.setAttribute('aria-selected', cell.classList.contains('matrix-selected') ? 'true' : 'false');
    });
    if (selected) message(`Selected rows ${selected.r1}–${selected.r2}, columns ${selected.c1 + 1}–${selected.c2 + 1} · ${pending.size} unsaved`);
  }
  function choose(row, col, extend = false, focus = true) {
    if (!withinPage(row) || col < 0 || col >= state.columns.length) return;
    if (!extend || !state.anchor) state.anchor = {row, col};
    state.active = {row, col}; state.end = {row, col};
    refreshSelection();
    if (focus) visibleCell(row, col)?.focus({preventScroll: true});
  }
  function validate(value, col) {
    if (value === '') return null;
    const type = dtypes[col] || '';
    if (/^(u?int)/.test(type) && !/^[+-]?\d+$/.test(value)) return 'This column requires an integer. Escape cancels the edit.';
    if (type.startsWith('float') && !/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$|^[+-]?(?:nan|inf|infinity)$/i.test(value)) {
      return 'This column requires a numeric value. Escape cancels the edit.';
    }
    return null;
  }
  function applyChanges(edits) {
    if (state.saved) return;
    const group = [];
    for (const [row, col, value] of edits) {
      const id = key(row, col), before = cellValue(row, col);
      if (before === value) continue;
      group.push({row, col, before, after: value});
      if (value === original.get(id)) pending.delete(id); else pending.set(id, value);
    }
    if (group.length) { history.push(group); redo.length = 0; }
    renderValues(); refreshSelection();
  }
  function replay(source, destination, reverse) {
    if (state.saved) return;
    const group = source.pop(); if (!group) return;
    group.forEach(edit => {
      const id = key(edit.row, edit.col), next = reverse ? edit.before : edit.after;
      if (next === original.get(id)) pending.delete(id); else pending.set(id, next);
    });
    destination.push(group); renderValues(); refreshSelection();
  }
  function startEdit(initial) {
    if (!state.active || state.editing || state.saved) return;
    const {row, col} = state.active, cell = visibleCell(row, col);
    if (!cell) return;
    state.editing = true;
    const before = cellValue(row, col) ?? '';
    cell.textContent = '';
    const input = document.createElement('input');
    input.type = 'text'; input.value = initial === undefined ? before : initial;
    input.setAttribute('aria-label', `Edit row ${row}, ${state.columns[col]}`);
    cell.append(input); input.focus();
    if (initial === undefined) input.select();
    const cancel = () => { state.editing = false; input.remove(); renderValues(); cell.focus(); };
    const commit = () => {
      const error = validate(input.value, col);
      if (error) { message(error); input.setAttribute('aria-invalid', 'true'); return; }
      const edited = input.value; state.editing = false; input.remove(); applyChanges([[row, col, edited]]); cell.focus();
    };
    input.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); cancel(); }
      else if (event.key === 'Enter' || event.key === 'Tab') {
        event.preventDefault(); event.stopPropagation();
        const direction = event.key === 'Tab' ? (event.shiftKey ? -1 : 1) : 0;
        commit();
        if (!state.editing && direction) choose(row, col + direction);
        else if (!state.editing && event.key === 'Enter') choose(row + 1, col);
      } else if (event.key === 'ArrowUp' || event.key === 'ArrowDown') event.stopPropagation();
    });
    input.addEventListener('click', event => event.stopPropagation());
    input.addEventListener('blur', () => {
      if (!state.editing) return;
      if (validate(input.value, col)) { message(validate(input.value, col)); input.focus(); }
      else commit();
    });
  }
  // Strict TSV parser: quoted tabs/newlines are values, not cell delimiters.
  function parseTSV(text) {
    const rows = []; let row = [], value = '', quoted = false, atStart = true;
    for (let i = 0; i < text.length; i++) {
      const char = text[i];
      if (quoted) {
        if (char === '"' && text[i + 1] === '"') { value += '"'; i++; }
        else if (char === '"') quoted = false;
        else value += char;
      } else if (char === '"' && atStart) { quoted = true; atStart = false; }
      else if (char === '\t') { row.push(value); value = ''; atStart = true; }
      else if (char === '\r' || char === '\n') {
        if (char === '\r' && text[i + 1] === '\n') i++;
        row.push(value); rows.push(row); row = []; value = ''; atStart = true;
      } else { value += char; atStart = false; }
    }
    if (quoted) throw Error('Unclosed quoted value in pasted table');
    if (row.length || value !== '' || !rows.length) { row.push(value); rows.push(row); }
    if (!rows[0].length || rows.some(r => r.length !== rows[0].length)) throw Error('Pasted rectangle has inconsistent row widths');
    return rows;
  }
  function formatTSV(values) {
    return values.map(row => row.map(v => {
      const raw = v ?? '';
      return /[\t\r\n"]/.test(raw) ? `"${raw.replace(/"/g, '""')}"` : raw;
    }).join('\t')).join('\r\n');
  }
  function selectedTSV() {
    const box = range(); if (!box) return '';
    const values = [];
    for (let row = box.r1; row <= box.r2; row++) {
      if (!withinPage(row)) throw Error('Copy is limited to the current loaded page');
      const cells = []; for (let col = box.c1; col <= box.c2; col++) cells.push(cellValue(row, col));
      values.push(cells);
    }
    return formatTSV(values);
  }
  function pasteTSV(text) {
    if (!state.active || state.editing || state.saved) return;
    try {
      const values = parseTSV(text), anchor = range();
      const startRow = anchor.r1, startCol = anchor.c1;
      if (!withinPage(startRow + values.length - 1) || startCol + values[0].length > state.columns.length) {
        throw Error('Paste exceeds the loaded page or matrix columns; use smaller rectangular ranges.');
      }
      const edits = [];
      values.forEach((row, r) => row.forEach((value, c) => {
        const problem = validate(value, startCol + c);
        if (problem) throw Error(`Row ${startRow + r}, column ${startCol + c + 1}: ${problem}`);
        edits.push([startRow + r, startCol + c, value]);
      }));
      if (pending.size + edits.length > 5000) throw Error('Drafts are limited to 5000 changed cells');
      applyChanges(edits);
      state.anchor = {row: startRow, col: startCol};
      choose(startRow + values.length - 1, startCol + values[0].length - 1, true);
    } catch (error) { message(error.message); }
  }
  function renderTable() {
    head.replaceChildren(); body.replaceChildren();
    const headerRow = document.createElement('tr');
    const corner = document.createElement('th'); corner.className = 'matrix-row-index'; corner.textContent = '#';
    headerRow.append(corner);
    state.columns.forEach(name => { const th = document.createElement('th'); th.textContent = name; headerRow.append(th); });
    head.append(headerRow);
    state.rows.forEach((row, index) => {
      const absoluteRow = state.first + index, tr = document.createElement('tr');
      const indexCell = document.createElement('th'); indexCell.className = 'matrix-row-index'; indexCell.textContent = String(absoluteRow);
      tr.append(indexCell);
      row.forEach((value, col) => {
        const td = document.createElement('td'); td.tabIndex = 0;
        td.dataset.row = String(absoluteRow); td.dataset.col = String(col);
        td.setAttribute('role', 'gridcell');
        td.setAttribute('aria-label', `Row ${absoluteRow}, ${state.columns[col]}`);
        original.set(key(absoluteRow, col), value === null ? '' : String(value));
        td.addEventListener('pointerdown', event => {
          if (event.button !== 0 || state.editing) return;
          dragging = true; choose(absoluteRow, col, event.shiftKey);
        });
        td.addEventListener('pointerenter', () => { if (dragging && !state.editing) choose(absoluteRow, col, true, false); });
        td.addEventListener('dblclick', () => { choose(absoluteRow, col); startEdit(); });
        tr.append(td);
      });
      body.append(tr);
    });
    renderValues();
    if (state.rows.length) choose(state.first, 0, false, false);
  }
  async function load() {
    const current = ++requestNumber;
    state.loading = true; updateButtons(); message('Loading matrix…');
    const params = new URLSearchParams({page: String(state.page), page_size: sizeSelect.value});
    try {
      const response = await fetch(`${root.dataset.rowsUrl}?${params}`, {headers: {'Accept': 'application/json'}});
      const payload = await response.json();
      if (!response.ok) throw Error(payload.error || 'Matrix page could not be loaded');
      if (current !== requestNumber) return;
      if (payload.skipped_malformed_row_count) throw Error('Malformed rows detected; editing is unavailable');
      state.page = payload.page; state.pageCount = payload.page_count;
      state.first = payload.first_row; state.columns = payload.columns;
      state.rows = payload.rows;
      renderTable();
      $('matrix-page-label').textContent = `Page ${state.page} of ${state.pageCount} · ${payload.valid_source_row_count} rows`;
      prevButton.disabled = state.page <= 1; nextButton.disabled = state.page >= state.pageCount;
      message(`Ready · ${state.rows.length} rows loaded (unfiltered source order)`);
    } catch (error) { if (current === requestNumber) message(error.message); }
    finally { if (current === requestNumber) { state.loading = false; updateButtons(); } }
  }
  document.addEventListener('pointerup', () => { dragging = false; });
  root.addEventListener('keydown', event => {
    if (state.editing || !state.active || event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
    if (event.ctrlKey || event.metaKey) {
      if (event.key.toLowerCase() === 'z') { event.preventDefault(); replay(history, redo, true); }
      if (event.key.toLowerCase() === 'y' || (event.shiftKey && event.key.toLowerCase() === 'z')) { event.preventDefault(); replay(redo, history, false); }
      return;
    }
    const {row, col} = state.active;
    const moves = {ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1], Tab: [0, event.shiftKey ? -1 : 1]};
    if (moves[event.key]) {
      event.preventDefault(); choose(row + moves[event.key][0], col + moves[event.key][1], event.shiftKey && event.key !== 'Tab');
    } else if (event.key === 'Enter' || event.key === 'F2') { event.preventDefault(); startEdit(); }
    else if (event.key.length === 1 && !event.altKey && event.target.matches('td[data-row]')) {
      event.preventDefault(); startEdit(event.key);
    }
  });
  root.addEventListener('copy', event => {
    if (state.editing || !state.active) return;
    event.clipboardData.setData('text/plain', selectedTSV()); event.preventDefault(); message('Selected range copied');
  });
  root.addEventListener('paste', event => {
    if (state.editing || !state.active) return;
    event.preventDefault(); pasteTSV(event.clipboardData.getData('text/plain'));
  });
  undoButton.addEventListener('click', () => replay(history, redo, true));
  redoButton.addEventListener('click', () => replay(redo, history, false));
  copyButton.addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(selectedTSV()); message('Selected range copied'); }
    catch (_) { message('Clipboard permission unavailable; select a cell and use Ctrl/Cmd+C'); }
  });
  prevButton.addEventListener('click', () => { if (state.page > 1) { state.page--; load(); } });
  nextButton.addEventListener('click', () => { if (state.page < state.pageCount) { state.page++; load(); } });
  sizeSelect.addEventListener('change', () => { state.page = 1; load(); });
  saveButton.addEventListener('click', async () => {
    if (!pending.size || state.editing || state.saved) return;
    saveButton.disabled = true; message('Saving a new derived dataset…');
    const edits = Array.from(pending, ([coordinate, value]) => {
      const [row, column] = coordinate.split(':').map(Number);
      return {row, column, value};
    });
    try {
      const response = await fetch(root.dataset.saveUrl, {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({source_sha256: sourceHash, edits}),
      });
      const payload = await response.json();
      if (!response.ok) throw Error(payload.error || 'Could not save matrix draft');
      resultBox.replaceChildren();
      const summary = document.createElement('p'); summary.textContent = `Saved ${payload.changed_cells} edited cells to a new derived dataset.`;
      const link = document.createElement('a'); link.href = payload.source_url; link.textContent = 'Open derived dataset';
      const provenance = document.createElement('a'); provenance.href = payload.provenance_url; provenance.textContent = 'View provenance';
      resultBox.append(summary, link, document.createTextNode(' · '), provenance);
      if (payload.experiment_url) {
        const experiment = document.createElement('a');
        experiment.href = payload.experiment_url;
        experiment.textContent = 'Use in experiment';
        resultBox.append(document.createTextNode(' · '), experiment);
      }
      resultBox.hidden = false;
      state.saved = true; history.length = 0; redo.length = 0; renderValues();
      message('Derived dataset saved. The original source remains unchanged.');
      // Editing the same draft again must start from a new explicit source view.
      saveButton.disabled = true;
    } catch (error) { message(error.message); updateButtons(); }
  });
  load();
})();
