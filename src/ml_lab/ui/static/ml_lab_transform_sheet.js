(() => {
  const form = document.querySelector('.transform-form');
  const sheet = document.querySelector('[data-transform-sheet]');
  const dialog = document.querySelector('[data-transform-column-actions]');
  if (!form || !sheet || !dialog) return;

  const selected = new Set();
  const headers = Array.from(sheet.querySelectorAll('[data-sheet-column]'));
  const selectedCount = sheet.querySelector('[data-sheet-selection-count]');
  const clearButton = sheet.querySelector('[data-sheet-clear-selection]');
  const openButton = sheet.querySelector('[data-sheet-open-actions]');
  const actionTitle = dialog.querySelector('[data-sheet-action-title]');
  const actionMeta = dialog.querySelector('[data-sheet-action-meta]');
  const selectedColumns = dialog.querySelector('[data-sheet-selected-columns]');
  const singleActions = dialog.querySelector('[data-sheet-single-actions]');
  const status = dialog.querySelector('[data-sheet-action-status]');
  const recipeContext = sheet.querySelector('[data-sheet-recipe-context]');
  const recipeLinks = sheet.querySelector('[data-sheet-recipe-links]');
  const recipeEmpty = sheet.querySelector('[data-sheet-recipe-empty]');
  const metadata = new Map(
    Array.from(document.querySelectorAll('[data-column-item]')).map((item) => [
      item.dataset.columnName,
      {
        type: item.dataset.type || 'unknown',
        missing: item.dataset.missing === '1',
        meta: item.querySelector('.column-browser-meta')?.textContent?.trim() || '',
      },
    ])
  );

  const control = (name) => form.elements.namedItem(name);
  const splitCsv = (raw) => String(raw || '').split(',').map((value) => value.trim()).filter(Boolean);
  const unique = (values) => Array.from(new Set(values));
  const escapeHtml = (value) => String(value).replace(/[&<>'"]/g, (char) => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));

  function dispatch(field) {
    if (!field) return;
    field.dispatchEvent(new Event('input', { bubbles: true }));
    field.dispatchEvent(new Event('change', { bubbles: true }));
  }

  function setValue(name, value) {
    const field = control(name);
    if (!field) return;
    field.value = value;
    dispatch(field);
  }

  function setChecked(name, value) {
    const field = control(name);
    if (!field) return;
    field.checked = Boolean(value);
    dispatch(field);
  }

  function addColumns(name, columns) {
    const field = control(name);
    if (!field) return;
    field.value = unique([...splitCsv(field.value), ...columns]).join(', ');
    dispatch(field);
  }

  function parseMapping(raw) {
    const map = new Map();
    String(raw || '').split(/\r?\n/).forEach((line) => {
      const trimmed = line.trim();
      if (!trimmed || !trimmed.includes('=')) return;
      const index = trimmed.indexOf('=');
      const key = trimmed.slice(0, index).trim();
      const value = trimmed.slice(index + 1).trim();
      if (key) map.set(key, value);
    });
    return map;
  }

  function setMappings(name, pairs) {
    const field = control(name);
    if (!field) return;
    const map = parseMapping(field.value);
    pairs.forEach(([key, value]) => map.set(key, value));
    field.value = Array.from(map.entries()).map(([key, value]) => `${key}=${value}`).join('\n');
    dispatch(field);
  }

  function configuredRecipeStepsFor(columns) {
    const selectedColumns = new Set(columns);
    return Array.from(document.querySelectorAll('[data-recipe-step]')).filter((item) => {
      const affected = splitCsv(item.dataset.recipeColumns || '');
      return affected.some((column) => selectedColumns.has(column));
    });
  }

  function refreshRecipeContext() {
    if (!recipeContext || !recipeLinks || !recipeEmpty) return;
    const columns = currentColumns();
    recipeContext.hidden = columns.length === 0;
    recipeLinks.innerHTML = '';
    const steps = configuredRecipeStepsFor(columns);
    recipeEmpty.hidden = steps.length > 0;
    steps.forEach((step) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'button mini secondary sheet-recipe-link';
      button.textContent = step.querySelector('strong')?.textContent || step.dataset.recipeStep;
      button.title = 'Jump to this recipe step';
      button.addEventListener('click', () => document.dispatchEvent(new CustomEvent('ml-lab:sheet-recipe-jump', { detail: { key: step.dataset.recipeStep } })));
      recipeLinks.appendChild(button);
    });
  }

  function emitSelection() {
    document.dispatchEvent(new CustomEvent('ml-lab:sheet-selection', { detail: { columns: currentColumns() } }));
  }

  function refreshSelection() {
    headers.forEach((header) => {
      header.classList.toggle('is-selected', selected.has(header.dataset.sheetColumn));
      header.querySelector('[data-sheet-column-select]')?.setAttribute('aria-pressed', selected.has(header.dataset.sheetColumn) ? 'true' : 'false');
    });
    const count = selected.size;
    selectedCount.textContent = `${count} column${count === 1 ? '' : 's'} selected`;
    clearButton.disabled = count === 0;
    openButton.disabled = count === 0;
    refreshRecipeContext();
    emitSelection();
  }

  function selectOnly(name) {
    selected.clear();
    selected.add(name);
    refreshSelection();
  }

  function toggle(name, additive = true) {
    if (!additive) selected.clear();
    if (selected.has(name)) selected.delete(name); else selected.add(name);
    refreshSelection();
  }

  function currentColumns() {
    return Array.from(selected);
  }

  function describeSelection() {
    const columns = currentColumns();
    actionTitle.textContent = columns.length === 1 ? columns[0] : `${columns.length} selected columns`;
    if (columns.length === 1) {
      const info = metadata.get(columns[0]);
      actionMeta.textContent = info ? `${info.type}${info.meta ? ` · ${info.meta}` : ''}` : '';
    } else {
      actionMeta.textContent = 'Actions below apply to every selected column unless marked single-column.';
    }
    selectedColumns.innerHTML = columns.map((column) => `<span class="sheet-selected-chip">${escapeHtml(column)}</span>`).join('');
    singleActions.hidden = columns.length !== 1;
    status.textContent = '';
  }

  function openActions() {
    if (!selected.size) return;
    describeSelection();
    dialog.showModal();
  }

  function report(message) {
    status.textContent = message;
    refreshBadges();
  }

  function applyAction(action) {
    const columns = currentColumns();
    if (!columns.length) return;
    switch (action) {
      case 'scale':
        addColumns('scale_columns', columns);
        setValue('scale_method', dialog.querySelector('[data-sheet-scale-method]').value);
        report(`Scale step updated for ${columns.length} column(s).`);
        break;
      case 'type': {
        const type = dialog.querySelector('[data-sheet-type]').value;
        setMappings('type_overrides', columns.map((column) => [column, type]));
        report(`Type override set to ${type} for ${columns.length} column(s).`);
        break;
      }
      case 'drop-missing':
        addColumns('drop_missing_columns', columns);
        report(`Missing-row rule updated for ${columns.length} column(s).`);
        break;
      case 'clean':
        addColumns('clean_columns', columns);
        setValue('clean_case', dialog.querySelector('[data-sheet-clean-case]').value);
        setChecked('collapse_whitespace', dialog.querySelector('[data-sheet-collapse-whitespace]').checked);
        report(`String-cleaning step updated for ${columns.length} column(s).`);
        break;
      case 'encode':
        addColumns('encode_columns', columns);
        report(`One-hot encoding step updated for ${columns.length} column(s).`);
        break;
      case 'outliers':
        addColumns('outlier_columns', columns);
        setValue('outlier_method', dialog.querySelector('[data-sheet-outlier-method]').value);
        setValue('outlier_iqr_multiplier', dialog.querySelector('[data-sheet-outlier-multiplier]').value || '1.5');
        report(`Outlier step updated for ${columns.length} column(s).`);
        break;
      case 'rename': {
        if (columns.length !== 1) return;
        const target = dialog.querySelector('[data-sheet-rename-target]').value.trim();
        if (!target) {
          report('Enter a new column name first.');
          dialog.querySelector('[data-sheet-rename-target]').focus();
          return;
        }
        setMappings('rename_columns', [[columns[0], target]]);
        report(`Rename step updated: ${columns[0]} → ${target}.`);
        break;
      }
      case 'filter': {
        if (columns.length !== 1) return;
        const operator = dialog.querySelector('[data-sheet-filter-operator]').value;
        setValue('filter_column', columns[0]);
        setValue('filter_operator', operator);
        setValue('filter_value', ['is_missing', 'not_missing'].includes(operator) ? '' : dialog.querySelector('[data-sheet-filter-value]').value);
        report(`Filter step configured for ${columns[0]}.`);
        break;
      }
      case 'drop':
        addColumns('drop_columns', columns);
        report(`${columns.length} column(s) added to the Drop columns recipe step.`);
        break;
      default:
        break;
    }
  }

  function badge(label, title) {
    const span = document.createElement('span');
    span.className = 'sheet-transform-badge';
    span.textContent = label;
    span.title = title;
    return span;
  }

  function refreshBadges() {
    const drop = new Set(splitCsv(control('drop_columns')?.value));
    const missing = new Set(splitCsv(control('drop_missing_columns')?.value));
    const clean = new Set(splitCsv(control('clean_columns')?.value));
    const encode = new Set(splitCsv(control('encode_columns')?.value));
    const scale = new Set(splitCsv(control('scale_columns')?.value));
    const outliers = new Set(splitCsv(control('outlier_columns')?.value));
    const types = parseMapping(control('type_overrides')?.value);
    const renames = parseMapping(control('rename_columns')?.value);
    const filterColumn = (control('filter_column')?.value || '').trim();

    headers.forEach((header) => {
      const column = header.dataset.sheetColumn;
      const badges = header.querySelector('[data-sheet-column-badges]');
      badges.innerHTML = '';
      if (renames.has(column)) badges.appendChild(badge('RENAME', `Rename to ${renames.get(column)}`));
      if (types.has(column)) badges.appendChild(badge('TYPE', `Convert to ${types.get(column)}`));
      if (missing.has(column)) badges.appendChild(badge('MISSING', 'Used by missing-row rule'));
      if (clean.has(column)) badges.appendChild(badge('CLEAN', 'String cleaning configured'));
      if (encode.has(column)) badges.appendChild(badge('ENCODE', 'One-hot encoding configured'));
      if (scale.has(column)) badges.appendChild(badge('SCALE', `${control('scale_method')?.value || 'standard'} scaling`));
      if (outliers.has(column)) badges.appendChild(badge('OUTLIER', `${control('outlier_method')?.value || 'clip_iqr'} configured`));
      if (filterColumn === column) badges.appendChild(badge('FILTER', 'Row filter configured'));
      if (drop.has(column)) badges.appendChild(badge('DROP', 'Column will be dropped'));
      header.classList.toggle('has-transformations', badges.childElementCount > 0);
    });
    refreshRecipeContext();
    emitSelection();
  }

  sheet.querySelectorAll('[data-sheet-column-select]').forEach((button) => {
    button.addEventListener('click', (event) => {
      toggle(button.dataset.sheetColumnSelect, event.ctrlKey || event.metaKey || event.shiftKey || selected.size > 0);
    });
  });

  sheet.querySelectorAll('[data-sheet-column-menu]').forEach((button) => {
    button.addEventListener('click', () => {
      const column = button.dataset.sheetColumnMenu;
      if (!selected.has(column)) selectOnly(column);
      openActions();
    });
  });

  clearButton.addEventListener('click', () => {
    selected.clear();
    refreshSelection();
  });
  openButton.addEventListener('click', openActions);
  dialog.querySelector('[data-sheet-close-actions]').addEventListener('click', () => dialog.close());
  dialog.querySelectorAll('[data-sheet-apply-action]').forEach((button) => {
    button.addEventListener('click', () => applyAction(button.dataset.sheetApplyAction));
  });

  document.addEventListener('ml-lab:recipe-focus', (event) => {
    const columns = new Set(event.detail?.columns || []);
    headers.forEach((header) => header.classList.toggle('is-recipe-focused', columns.has(header.dataset.sheetColumn)));
    if (columns.size) {
      const first = headers.find((header) => columns.has(header.dataset.sheetColumn));
      first?.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
    }
  });

  form.addEventListener('input', refreshBadges);
  form.addEventListener('change', refreshBadges);
  refreshSelection();
  refreshBadges();
})();
