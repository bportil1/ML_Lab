(() => {
  const form = document.querySelector('.transform-form');
  const list = document.querySelector('[data-recipe-list]');
  const empty = document.querySelector('[data-recipe-empty]');
  const orderInput = document.querySelector('[data-recipe-order]');
  const disabledInput = document.querySelector('[data-recipe-disabled]');
  if (!form || !list || !orderInput || !disabledInput) return;

  const specs = [
    { key: 'select_columns', label: 'Select columns', primary: 'select_columns', fields: ['select_columns'] },
    { key: 'drop_columns', label: 'Drop columns', primary: 'drop_columns', fields: ['drop_columns'] },
    { key: 'drop_quality_columns', label: 'Quality-based removal', primary: 'drop_all_missing', fields: ['drop_all_missing', 'drop_constant', 'protected_quality_columns'], active: () => checked('drop_all_missing') || checked('drop_constant') },
    { key: 'rename_columns', label: 'Rename columns', primary: 'rename_columns', fields: ['rename_columns'] },
    { key: 'coerce_types', label: 'Convert types', primary: 'type_overrides', fields: ['type_overrides', 'coerce_errors'] },
    { key: 'sentinel_to_missing', label: 'Sentinels → missing', primary: 'sentinel_values', fields: ['sentinel_values', 'sentinel_columns'] },
    { key: 'fill_missing', label: 'Fill missing values', primary: 'fill_missing', fields: ['fill_missing'] },
    { key: 'drop_missing_rows', label: 'Drop missing rows', primary: 'drop_missing_columns', fields: ['drop_missing_columns', 'drop_missing_how'] },
    { key: 'clean_strings', label: 'Clean strings', primary: 'clean_columns', fields: ['clean_columns', 'clean_case', 'collapse_whitespace'] },
    { key: 'drop_duplicates', label: 'Drop duplicates', primary: 'drop_duplicates', fields: ['drop_duplicates', 'duplicate_columns', 'duplicate_keep'], active: () => checked('drop_duplicates') },
    { key: 'filter_rows', label: 'Filter rows', primary: 'filter_column', fields: ['filter_column', 'filter_operator', 'filter_value'] },
    { key: 'derive', label: 'Derive regex column', primary: 'derive_source', fields: ['derive_source', 'derive_target', 'derive_pattern', 'derive_group'], active: () => value('derive_source') || value('derive_target') || value('derive_pattern') },
    { key: 'encode_categorical', label: 'One-hot encode', primary: 'encode_columns', fields: ['encode_columns', 'encode_drop_first'] },
    { key: 'scale', label: 'Scale', primary: 'scale_columns', fields: ['scale_columns', 'scale_method'] },
    { key: 'outliers', label: 'Outlier handling', primary: 'outlier_columns', fields: ['outlier_columns', 'outlier_method', 'outlier_iqr_multiplier'] },
  ];
  const canonical = specs.map((spec) => spec.key);
  const byKey = new Map(specs.map((spec) => [spec.key, spec]));
  const control = (name) => form.elements.namedItem(name);
  const value = (name) => (control(name)?.value || '').trim();
  const checked = (name) => Boolean(control(name)?.checked);
  const parseList = (raw) => raw.split(',').map((x) => x.trim()).filter(Boolean);
  let order = parseList(orderInput.value).filter((key, i, arr) => byKey.has(key) && arr.indexOf(key) === i);
  canonical.forEach((key) => { if (!order.includes(key)) order.push(key); });
  let disabled = new Set(parseList(disabledInput.value).filter((key) => byKey.has(key)));

  function isActive(spec) {
    if (spec.active) return Boolean(spec.active());
    return Boolean(value(spec.primary));
  }

  function countCsv(raw) { return parseList(raw).length; }
  function lineCount(raw) { return raw.split(/\r?\n/).map((x) => x.trim()).filter(Boolean).length; }
  function summary(spec) {
    switch (spec.key) {
      case 'select_columns': return `${countCsv(value('select_columns'))} column(s)`;
      case 'drop_columns': return `${countCsv(value('drop_columns'))} column(s)`;
      case 'drop_quality_columns': return [checked('drop_all_missing') ? 'all-missing' : '', checked('drop_constant') ? 'constant' : ''].filter(Boolean).join(' + ');
      case 'rename_columns': return `${lineCount(value('rename_columns'))} rename(s)`;
      case 'coerce_types': return `${lineCount(value('type_overrides'))} override(s) · ${value('coerce_errors') || 'coerce'}`;
      case 'sentinel_to_missing': return `${countCsv(value('sentinel_values'))} sentinel(s)${value('sentinel_columns') ? ` · ${countCsv(value('sentinel_columns'))} column(s)` : ' · all columns'}`;
      case 'fill_missing': return `${lineCount(value('fill_missing'))} fill rule(s)`;
      case 'drop_missing_rows': return `${countCsv(value('drop_missing_columns'))} key column(s) · ${value('drop_missing_how') || 'any'}`;
      case 'clean_strings': return `${countCsv(value('clean_columns'))} column(s) · ${value('clean_case') || 'preserve'}`;
      case 'drop_duplicates': return `${value('duplicate_columns') ? `${countCsv(value('duplicate_columns'))} key column(s)` : 'whole row'} · keep ${value('duplicate_keep') || 'first'}`;
      case 'filter_rows': return `${value('filter_column')} · ${value('filter_operator') || 'eq'}${['is_missing','not_missing'].includes(value('filter_operator')) ? '' : ` · ${value('filter_value')}`}`;
      case 'derive': return `${value('derive_source') || 'source'} → ${value('derive_target') || 'target'}`;
      case 'encode_categorical': return `${countCsv(value('encode_columns'))} column(s) · one-hot`;
      case 'scale': return `${countCsv(value('scale_columns'))} column(s) · ${value('scale_method') || 'standard'}`;
      case 'outliers': return `${countCsv(value('outlier_columns'))} column(s) · ${value('outlier_method') || 'clip_iqr'}`;
      default: return '';
    }
  }

  function affectedColumns(spec) {
    const csv = (name) => new Set(parseList(value(name)));
    const mappingKeys = (name) => new Set(value(name).split(/\r?\n/).map((line) => line.trim()).filter((line) => line.includes('=')).map((line) => line.slice(0, line.indexOf('=')).trim()).filter(Boolean));
    switch (spec.key) {
      case 'select_columns': return [...csv('select_columns')];
      case 'drop_columns': return [...csv('drop_columns')];
      case 'drop_quality_columns': return [...csv('protected_quality_columns')];
      case 'rename_columns': return [...mappingKeys('rename_columns')];
      case 'coerce_types': return [...mappingKeys('type_overrides')];
      case 'sentinel_to_missing': return [...csv('sentinel_columns')];
      case 'fill_missing': return [...mappingKeys('fill_missing')];
      case 'drop_missing_rows': return [...csv('drop_missing_columns')];
      case 'clean_strings': return [...csv('clean_columns')];
      case 'drop_duplicates': return [...csv('duplicate_columns')];
      case 'filter_rows': return value('filter_column') ? [value('filter_column')] : [];
      case 'derive': return value('derive_source') ? [value('derive_source')] : [];
      case 'encode_categorical': return [...csv('encode_columns')];
      case 'scale': return [...csv('scale_columns')];
      case 'outliers': return [...csv('outlier_columns')];
      default: return [];
    }
  }

  function emitRecipeFocus(key, columns) {
    document.dispatchEvent(new CustomEvent('ml-lab:recipe-focus', { detail: { key, columns } }));
  }

  function emitRecipeChanged() {
    document.dispatchEvent(new CustomEvent('ml-lab:recipe-changed'));
  }

  function syncHidden() {
    orderInput.value = order.join(',');
    disabledInput.value = order.filter((key) => disabled.has(key)).join(',');
  }

  function edit(spec) {
    const target = control(spec.primary);
    if (!target) return;
    target.scrollIntoView({ behavior: 'smooth', block: 'center' });
    setTimeout(() => target.focus?.(), 250);
  }

  function clearField(name) {
    const field = control(name);
    if (!field) return;
    if (field.type === 'checkbox' || field.type === 'radio') field.checked = false;
    else if (field.tagName === 'SELECT') field.selectedIndex = 0;
    else field.value = '';
    field.dispatchEvent(new Event('change', { bubbles: true }));
  }

  function remove(spec) {
    spec.fields.forEach(clearField);
    disabled.delete(spec.key);
    render();
    emitRecipeChanged();
  }

  function move(key, direction) {
    const activeKeys = order.filter((candidate) => isActive(byKey.get(candidate)));
    const activeIndex = activeKeys.indexOf(key);
    const other = activeKeys[activeIndex + direction];
    if (!other) return;
    const a = order.indexOf(key); const b = order.indexOf(other);
    [order[a], order[b]] = [order[b], order[a]];
    render();
    emitRecipeChanged();
  }

  function button(text, title, handler, disabledState = false) {
    const el = document.createElement('button');
    el.type = 'button'; el.className = 'button mini secondary'; el.textContent = text; el.title = title; el.disabled = disabledState;
    el.addEventListener('click', handler); return el;
  }

  function render() {
    syncHidden();
    list.innerHTML = '';
    const activeKeys = order.filter((key) => isActive(byKey.get(key)));
    empty.hidden = activeKeys.length > 0;
    activeKeys.forEach((key, index) => {
      const spec = byKey.get(key);
      const item = document.createElement('li');
      item.className = `recipe-step${disabled.has(key) ? ' is-disabled' : ''}`;
      item.dataset.recipeStep = key;
      item.dataset.recipeColumns = affectedColumns(spec).join(',');
      const main = document.createElement('div'); main.className = 'recipe-step-main';
      const heading = document.createElement('div'); heading.className = 'recipe-step-heading';
      const number = document.createElement('span'); number.className = 'recipe-step-number'; number.textContent = String(index + 1);
      const text = document.createElement('div');
      const strong = document.createElement('strong'); strong.textContent = spec.label;
      const detail = document.createElement('div'); detail.className = 'tiny muted'; detail.textContent = summary(spec);
      text.append(strong, detail); heading.append(number, text); main.append(heading);
      const actions = document.createElement('div'); actions.className = 'recipe-step-actions';
      actions.append(
        button('Edit', `Edit ${spec.label}`, () => edit(spec)),
        button(disabled.has(key) ? 'Enable' : 'Disable', `${disabled.has(key) ? 'Enable' : 'Disable'} ${spec.label}`, () => { disabled.has(key) ? disabled.delete(key) : disabled.add(key); render(); emitRecipeChanged(); }),
        button('↑', 'Move earlier', () => move(key, -1), index === 0),
        button('↓', 'Move later', () => move(key, 1), index === activeKeys.length - 1),
        button('Delete', `Delete ${spec.label}`, () => remove(spec))
      );
      item.append(main, actions); list.append(item);
      main.tabIndex = 0;
      main.title = 'Highlight affected dataset columns';
      const focusColumns = () => emitRecipeFocus(key, affectedColumns(spec));
      main.addEventListener('click', focusColumns);
      main.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); focusColumns(); } });
    });
    syncHidden();
  }

  document.addEventListener('ml-lab:sheet-selection', (event) => {
    const selected = new Set(event.detail?.columns || []);
    list.querySelectorAll('[data-recipe-step]').forEach((item) => {
      const columns = parseList(item.dataset.recipeColumns || '');
      item.classList.toggle('is-sheet-related', selected.size > 0 && columns.some((column) => selected.has(column)));
    });
  });

  document.addEventListener('ml-lab:sheet-recipe-jump', (event) => {
    const item = list.querySelector(`[data-recipe-step="${event.detail?.key || ''}"]`);
    if (!item) return;
    list.querySelectorAll('.is-focused').forEach((node) => node.classList.remove('is-focused'));
    item.classList.add('is-focused');
    item.scrollIntoView({ behavior: 'smooth', block: 'center' });
    setTimeout(() => item.classList.remove('is-focused'), 1800);
  });

  form.addEventListener('input', (event) => { if (event.target !== orderInput && event.target !== disabledInput) render(); });
  form.addEventListener('change', (event) => { if (event.target !== orderInput && event.target !== disabledInput) render(); });
  render();
})();
