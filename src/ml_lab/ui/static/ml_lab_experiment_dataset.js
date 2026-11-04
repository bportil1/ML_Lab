(() => {
  // Conventional spreadsheet behavior: a column has one selected role.
  const form = document.querySelector('[data-experiment-dataset-form]');
  if (!form) return;
  form.addEventListener('change', (event) => {
    const item = event.target;
    if (!(item instanceof HTMLInputElement) || !item.checked ||
        !['features', 'targets'].includes(item.name)) return;
    const other = item.name === 'features' ? 'targets' : 'features';
    for (const candidate of form.querySelectorAll(`input[name="${other}"]`)) {
      if (candidate.value === item.value) candidate.checked = false;
    }
  });
})();
