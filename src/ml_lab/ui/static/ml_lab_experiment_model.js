/* Navigation enhancement only. All model controls work with ordinary HTML GET/POST. */
(() => {
  const picker = document.querySelector('.model-picker');
  if (!picker) return;
  for (const input of picker.querySelectorAll('[data-model-picker]')) {
    input.addEventListener('change', () => {
      // A new family must not carry a method from the previous family.
      if (input.name === 'family') picker.querySelector('[name="method"]').disabled = true;
      picker.requestSubmit();
    });
  }
})();
