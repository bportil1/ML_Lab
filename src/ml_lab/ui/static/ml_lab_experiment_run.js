(() => {
  'use strict';
  const root = document.querySelector('.training-run-panel[data-status-url]');
  if (!root) return;
  const status = document.getElementById('training-run-state');
  const stage = document.getElementById('training-run-stage');
  const step = document.getElementById('training-run-step');
  const error = document.getElementById('training-run-error');
  const artifacts = document.getElementById('training-run-artifacts');
  const results = document.getElementById('training-run-results');
  const done = new Set(['succeeded', 'failed', 'cancelled']);
  const poll = async () => {
    try {
      const response = await fetch(root.dataset.statusUrl, { cache: 'no-store' });
      if (!response.ok) throw new Error(`Run status unavailable (${response.status})`);
      const item = await response.json();
      status.textContent = item.run.status;
      stage.textContent = item.stage + (item.cancellation_requested ? ' · cancellation requested' : '');
      step.textContent = `${item.step} / ${item.steps}`;
      error.textContent = item.run.error || '';
      error.classList.toggle('hidden', !item.run.error);
      results.classList.toggle('hidden', item.run.status !== 'succeeded');
      artifacts.replaceChildren();
      (item.artifacts || []).forEach(a => {
        const li = document.createElement('li');
        const kind = document.createElement('code');
        kind.textContent = a.kind;
        li.append(kind, document.createTextNode(` · ${a.path}`));
        artifacts.appendChild(li);
      });
      if (done.has(item.run.status)) {
        // Regenerate the short-lived signed dataset-selection URL after the
        // asynchronous generator finishes; never fabricate a file link.
        if (item.run.status === 'succeeded' &&
            item.run.environment.executor === 'ml_lab.experiments.synthetic@1') {
          window.location.reload();
        }
        return;
      }
    } catch (err) {
      stage.textContent = String(err.message || err);
    }
    window.setTimeout(poll, 1500);
  };
  if (!done.has(status.textContent.trim())) window.setTimeout(poll, 700);
})();
