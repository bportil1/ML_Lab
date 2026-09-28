(() => {
  const browserButtons = Array.from(document.querySelectorAll('.column-browser-button'));
  const dialog = document.querySelector('[data-column-browser]');
  if (!dialog) {
    browserButtons.forEach((button) => {
      button.disabled = true;
      button.title = 'Load / refresh columns first';
    });
    return;
  }

  const items = Array.from(dialog.querySelectorAll('[data-column-item]'));
  const checks = Array.from(dialog.querySelectorAll('[data-column-checkbox]'));
  const search = dialog.querySelector('[data-column-search]');
  const purpose = dialog.querySelector('[data-column-browser-purpose]');
  const matchCount = dialog.querySelector('[data-column-match-count]');
  const selectedCount = dialog.querySelector('[data-column-selected-count]');
  const applyButton = dialog.querySelector('[data-apply-columns]');
  const patternInput = dialog.querySelector('[data-column-pattern]');
  const patternMode = dialog.querySelector('[data-column-pattern-mode]');
  const patternStatus = dialog.querySelector('[data-pattern-status]');
  const groupName = dialog.querySelector('[data-group-name]');
  const groupSelect = dialog.querySelector('[data-group-select]');
  const loadGroupButton = dialog.querySelector('[data-load-group]');
  const deleteGroupButton = dialog.querySelector('[data-delete-group]');
  const savedGroups = new Map();
  let activeFilter = 'all';
  let target = null;
  let single = false;

  const splitNames = (raw) => raw.split(',').map((value) => value.trim()).filter(Boolean);

  function matchesFilter(item) {
    if (activeFilter === 'all') return true;
    if (activeFilter === 'numeric') return ['integer', 'float'].includes(item.dataset.type);
    if (activeFilter === 'text') return item.dataset.type === 'text';
    if (activeFilter === 'categorical') return item.dataset.type === 'categorical';
    if (activeFilter === 'boolean') return item.dataset.type === 'boolean';
    if (activeFilter === 'datetime') return item.dataset.type === 'datetime';
    if (activeFilter === 'missing') return item.dataset.missing === '1';
    if (activeFilter === 'constant') return item.dataset.constant === '1';
    if (activeFilter === 'high-cardinality') return item.dataset.highCardinality === '1';
    if (activeFilter === 'selected') return item.querySelector('[data-column-checkbox]').checked;
    return true;
  }

  function refresh() {
    const needle = search.value.trim().toLowerCase();
    let shown = 0;
    items.forEach((item) => {
      const visible = (!needle || item.dataset.name.includes(needle)) && matchesFilter(item);
      item.hidden = !visible;
      if (visible) shown += 1;
    });
    const selected = checks.filter((checkbox) => checkbox.checked).length;
    matchCount.textContent = `${shown} of ${items.length} columns shown`;
    selectedCount.textContent = `${selected} selected`;
  }

  function resetFilters() {
    search.value = '';
    activeFilter = 'all';
    dialog.querySelectorAll('[data-column-filter]').forEach((filterButton) => {
      filterButton.classList.toggle('active', filterButton.dataset.columnFilter === 'all');
    });
  }

  function setSelection(names) {
    const selected = new Set(names);
    checks.forEach((checkbox) => { checkbox.checked = selected.has(checkbox.value); });
    if (single) {
      let found = false;
      checks.forEach((checkbox) => {
        if (!checkbox.checked) return;
        if (found) checkbox.checked = false;
        found = true;
      });
    }
    refresh();
  }

  function openFor(button) {
    target = document.querySelector(`[name="${CSS.escape(button.dataset.columnTarget)}"]`);
    if (!target) return;
    single = button.dataset.columnSingle === 'true';
    setSelection(splitNames(target.value));
    resetFilters();
    patternInput.value = '';
    patternMode.value = 'wildcard';
    patternStatus.textContent = 'Patterns match full column names.';
    purpose.textContent = single
      ? `Choose one column for ${button.dataset.columnTarget.replaceAll('_', ' ')}.`
      : `Choose columns for ${button.dataset.columnTarget.replaceAll('_', ' ')}.`;
    refresh();
    dialog.showModal();
    search.focus();
  }

  function wildcardToRegExp(pattern) {
    const escaped = pattern.replace(/[.+^${}()|[\]\\]/g, '\\$&');
    return new RegExp(`^${escaped.replaceAll('*', '.*').replaceAll('?', '.')}$`, 'i');
  }

  function compilePattern() {
    const raw = patternInput.value.trim();
    if (!raw) {
      patternStatus.textContent = 'Enter a wildcard or regular expression.';
      return null;
    }
    try {
      const expression = patternMode.value === 'regex' ? new RegExp(raw, 'i') : wildcardToRegExp(raw);
      patternStatus.textContent = patternMode.value === 'regex' ? 'Valid regular expression.' : 'Valid wildcard pattern.';
      return expression;
    } catch (error) {
      patternStatus.textContent = `Invalid pattern: ${error.message}`;
      return null;
    }
  }

  function applyPattern(selectMatches) {
    const expression = compilePattern();
    if (!expression) return;
    const matching = items.filter((item) => expression.test(item.dataset.columnName));
    if (single && selectMatches) {
      checks.forEach((checkbox) => { checkbox.checked = false; });
      const first = matching[0]?.querySelector('[data-column-checkbox]');
      if (first) first.checked = true;
    } else {
      matching.forEach((item) => {
        item.querySelector('[data-column-checkbox]').checked = selectMatches;
      });
    }
    patternStatus.textContent = `${matching.length} column${matching.length === 1 ? '' : 's'} matched.`;
    refresh();
  }

  function refreshGroupSelect(selectedName = '') {
    const current = selectedName || groupSelect.value;
    groupSelect.innerHTML = '<option value="">Choose a saved selection…</option>';
    Array.from(savedGroups.keys()).sort((a, b) => a.localeCompare(b)).forEach((name) => {
      const option = document.createElement('option');
      option.value = name;
      option.textContent = `${name} (${savedGroups.get(name).length})`;
      groupSelect.appendChild(option);
    });
    if (savedGroups.has(current)) groupSelect.value = current;
    loadGroupButton.disabled = !groupSelect.value;
    deleteGroupButton.disabled = !groupSelect.value;
  }

  browserButtons.forEach((button) => {
    button.disabled = false;
    button.addEventListener('click', () => openFor(button));
  });

  search.addEventListener('input', refresh);
  dialog.querySelectorAll('[data-column-filter]').forEach((button) => {
    button.addEventListener('click', () => {
      activeFilter = button.dataset.columnFilter;
      dialog.querySelectorAll('[data-column-filter]').forEach((candidate) => candidate.classList.toggle('active', candidate === button));
      refresh();
    });
  });

  checks.forEach((checkbox) => {
    checkbox.addEventListener('change', () => {
      if (single && checkbox.checked) {
        checks.forEach((other) => { if (other !== checkbox) other.checked = false; });
      }
      refresh();
    });
  });

  dialog.querySelector('[data-select-filtered]').addEventListener('click', () => {
    const visible = items.filter((item) => !item.hidden);
    if (single) {
      checks.forEach((checkbox) => { checkbox.checked = false; });
      const first = visible[0]?.querySelector('[data-column-checkbox]');
      if (first) first.checked = true;
    } else {
      visible.forEach((item) => { item.querySelector('[data-column-checkbox]').checked = true; });
    }
    refresh();
  });

  dialog.querySelector('[data-deselect-filtered]').addEventListener('click', () => {
    items.filter((item) => !item.hidden).forEach((item) => {
      item.querySelector('[data-column-checkbox]').checked = false;
    });
    refresh();
  });

  dialog.querySelector('[data-invert-columns]').addEventListener('click', () => {
    if (single) return;
    checks.forEach((checkbox) => { checkbox.checked = !checkbox.checked; });
    refresh();
  });

  dialog.querySelector('[data-clear-columns]').addEventListener('click', () => {
    checks.forEach((checkbox) => { checkbox.checked = false; });
    refresh();
  });

  dialog.querySelector('[data-select-pattern]').addEventListener('click', () => applyPattern(true));
  dialog.querySelector('[data-exclude-pattern]').addEventListener('click', () => applyPattern(false));
  patternInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      applyPattern(true);
    }
  });

  dialog.querySelector('[data-save-group]').addEventListener('click', () => {
    const name = groupName.value.trim();
    if (!name) {
      groupName.focus();
      return;
    }
    const values = checks.filter((checkbox) => checkbox.checked).map((checkbox) => checkbox.value);
    savedGroups.set(name, values);
    refreshGroupSelect(name);
    groupName.value = '';
  });

  groupSelect.addEventListener('change', () => {
    loadGroupButton.disabled = !groupSelect.value;
    deleteGroupButton.disabled = !groupSelect.value;
  });

  loadGroupButton.addEventListener('click', () => {
    const values = savedGroups.get(groupSelect.value);
    if (values) setSelection(values);
  });

  deleteGroupButton.addEventListener('click', () => {
    if (!groupSelect.value) return;
    savedGroups.delete(groupSelect.value);
    refreshGroupSelect();
  });

  applyButton.addEventListener('click', () => {
    if (!target) return;
    const values = checks.filter((checkbox) => checkbox.checked).map((checkbox) => checkbox.value);
    target.value = single ? (values[0] || '') : values.join(', ');
    target.dispatchEvent(new Event('change', { bubbles: true }));
    dialog.close();
  });
})();
