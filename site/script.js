// The diagram is illustrative: no footage is uploaded or processed by this page.
const demo = document.querySelector('[data-demo]');
if (demo) {
  const controls = demo.querySelector('.demo-switch');
  const frames = demo.querySelector('#filmstrip');
  const status = demo.querySelector('#demo-status');
  const buttons = [...demo.querySelectorAll('[data-view]')];
  controls.hidden = false;
  for (const button of buttons) {
    button.addEventListener('click', () => {
      const refined = button.dataset.view === 'refined';
      demo.dataset.state = button.dataset.view;
      for (const item of buttons) item.setAttribute('aria-pressed', String(item === button));
      frames.setAttribute('aria-label', refined
        ? 'Illustrated landscape across three generations. After: the horizon and color align across both joins.'
        : 'Illustrated landscape across three generations. Before: the horizon and color shift at each join.');
      status.textContent = refined
        ? 'Framing aligned. Color matched. The feeling of one take.'
        : 'A changed crop. A different tone. A break in the spell.';
    });
  }
}
