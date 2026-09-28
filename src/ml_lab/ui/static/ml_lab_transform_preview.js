(() => {
  const form = document.querySelector('.transform-form');
  if (!form) return;

  const validation = form.querySelector('[data-transform-validation]');
  const applyButton = form.querySelector('[data-transform-apply]');
  const state = form.querySelector('[data-validation-state]');
  let previewValid = validation?.dataset.previewValid === '1';

  const renderValidation = () => {
    if (!applyButton || !state) return;
    applyButton.disabled = !previewValid;
    if (previewValid) {
      state.innerHTML = '<strong>Preview validated against the current submitted recipe.</strong><span class="tiny muted">Apply is enabled until the recipe or source controls change.</span>';
    } else {
      state.innerHTML = '<strong>Preview required before Apply.</strong><span class="tiny muted">Recipe/source controls changed. Run Preview transformation again before writing a derived dataset.</span>';
    }
  };

  form.addEventListener('input', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    if (target.matches('[name="overwrite"]')) return;
    previewValid = false;
    renderValidation();
  });
  form.addEventListener('change', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    if (target.matches('[name="overwrite"]')) return;
    previewValid = false;
    renderValidation();
  });
  document.addEventListener('ml-lab:recipe-changed', () => {
    previewValid = false;
    renderValidation();
  });
  renderValidation();

  const panel = document.querySelector('[data-transform-preview-panel]');
  if (!panel) return;
  const tabs = Array.from(panel.querySelectorAll('[data-preview-tab]'));
  const views = Array.from(panel.querySelectorAll('[data-preview-view]'));
  tabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      const name = tab.dataset.previewTab;
      tabs.forEach((candidate) => candidate.classList.toggle('is-active', candidate === tab));
      views.forEach((view) => { view.hidden = view.dataset.previewView !== name; });
    });
  });
})();
