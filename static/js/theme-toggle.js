(function () {
  const KEY = 'theme';
  const root = document.documentElement; // <html>
  const btn = document.getElementById('toggle-theme');

  function applyTheme(theme) {
    root.setAttribute('data-theme', theme);
    localStorage.setItem(KEY, theme);
    if (btn) btn.textContent = theme === 'dark' ? '🌞' : '🌙';
  }

  const stored = localStorage.getItem(KEY);
  const initial =
    stored || (window.matchMedia &&
               window.matchMedia('(prefers-color-scheme: dark)').matches
               ? 'dark' : 'light');

  applyTheme(initial);

  if (btn) {
    btn.addEventListener('click', () => {
      const next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      applyTheme(next);
    });
  }
})();
