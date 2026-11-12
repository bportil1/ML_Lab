/* ML-UW-12: read-only context refresh on revisit; never auto-submit or change a selection. */
(() => {
  const node = document.querySelector('[data-mlwb-context-url]');
  if (!node || typeof window.fetch !== 'function') return;
  const url = node.getAttribute('data-mlwb-context-url');
  if (!url) return;
  let latestRequest = 0;

  function markStaleForms(context) {
    const generation = context && context.generation;
    const active = context && context.status === 'ready' ? context.active_dataset : null;
    const forms = document.querySelectorAll('form input[name="expected_generation"]');
    forms.forEach(input => {
      const form = input.form;
      if (!form) return;
      const original = input.value;
      const source = form.querySelector('input[name="source_sha256"]');
      // Dataset selection intentionally previews a different source; compare
      // only the generation, not the candidate's fingerprint.
      const selectingDataset = !!form.querySelector('input[name="source_dataset_id"]');
      const stale = String(generation) !== String(original) ||
        (!selectingDataset && source && (!active || source.value !== active.sha256));
      let warning = form.querySelector('[data-mlwb-stale-warning]');
      if (stale && !warning) {
        warning = document.createElement('div');
        warning.className = 'mlwb-context-warning';
        warning.setAttribute('role', 'alert');
        warning.setAttribute('data-mlwb-stale-warning', '');
        const message = document.createElement('span');
        message.textContent = 'The workbench dataset or revision changed since this form was opened. ' +
          'Reload this view before submitting. Unsaved edits will be lost.';
        const link = document.createElement('a');
        link.href = window.location.href;
        link.textContent = ' Reload current view';
        warning.appendChild(message);
        warning.appendChild(link);
        form.insertBefore(warning, form.firstChild);
      }
      if (warning) warning.hidden = !stale;
    });
  }

  function refreshContext() {
    if (document.visibilityState === 'hidden') return;
    const seq = ++latestRequest;
    window.fetch(url, {credentials: 'same-origin', cache: 'no-store',
      headers: {'Accept': 'application/json'}})
      .then(response => {
        if (!response.ok) throw new Error('context unavailable');
        return response.json();
      })
      .then(context => {
        if (seq !== latestRequest) return;
        const status = context && context.status;
        node.dataset.status = status || 'unknown';
        if (status === 'ready' && context.active_dataset) {
          const dataset = context.active_dataset;
          const source = typeof dataset.path === 'string' ? dataset.path : '';
          const name = source.split(/[\\/]/).filter(Boolean).pop();
          node.textContent = 'Active dataset: ' + (name || 'selected') + ' · rev ' + context.experiment_revision;
        } else if (status === 'stale') {
          node.textContent = 'Active dataset unavailable — source changed; reselect in Data';
        } else {
          node.textContent = 'No active workbench dataset';
        }
        if (typeof context.generation === 'number') markStaleForms(context);
      })
      .catch(() => {
        if (seq !== latestRequest) return;
        node.dataset.status = 'unavailable';
        node.textContent = 'Dataset context unavailable — open Data or reload';
        // On network failure, never infer that a form is stale or mutate it.
      });
  }

  refreshContext();
  window.addEventListener('pageshow', refreshContext);
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') refreshContext();
  });
})();
