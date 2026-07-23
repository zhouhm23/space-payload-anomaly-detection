/* PHM admin custom-page shared JS utilities.
   - On boot, injects window.THEME.colors into :root CSS variables (single source
     of truth, shared across front-end and back-end).
   - fetchJSON: unified POST wrapper carrying CSRF + JSON.
   - toast / drawer / spinner: basic UI components.
   All functions live under the window.PHM namespace to avoid polluting globals. */
(function () {
  'use strict';

  // ── Theme injection into :root CSS variables ───────────────────
  // The admin follows SimpleUI's light style (white cards); it does NOT inject
  // the front-end dark background/text colors (that would clash with the white
  // SimpleUI base and look jarring).
  // Only "accent" colors (red/yellow/green/blue/cyan) are injected — they drive
  // badge/status colors, are consistent across front-end and admin, and stay
  // legible on light cards.
  function injectTheme() {
    var theme = (window.THEME && window.THEME.colors) || {};
    var root = document.documentElement;
    var accentMap = {
      blue: '--phm-blue',
      green: '--phm-green',
      yellow: '--phm-yellow',
      red: '--phm-red',
      cyan: '--phm-cyan'
    };
    Object.keys(accentMap).forEach(function (k) {
      if (theme[k]) root.style.setProperty(accentMap[k], theme[k]);
    });
  }

  // ── Toast ─────────────────────────────────────────────────────
  function ensureToastContainer() {
    var c = document.getElementById('phm-toast-container');
    if (!c) {
      c = document.createElement('div');
      c.id = 'phm-toast-container';
      document.body.appendChild(c);
    }
    return c;
  }

  function toast(message, type, duration) {
    type = type || 'info';
    duration = duration || 3000;
    var c = ensureToastContainer();
    var el = document.createElement('div');
    el.className = 'phm-toast phm-toast-' + type;
    el.textContent = message;
    c.appendChild(el);
    setTimeout(function () {
      el.style.opacity = '0';
      el.style.transition = 'opacity 0.2s';
      setTimeout(function () { if (el.parentNode) el.parentNode.removeChild(el); }, 200);
    }, duration);
  }

  // ── fetch JSON (with CSRF) ─────────────────────────────────────
  function getCookie(name) {
    var value = '; ' + document.cookie;
    var parts = value.split('; ' + name + '=');
    if (parts.length === 2) return parts.pop().split(';').shift();
    return null;
  }

  function fetchJSON(url, options) {
    options = options || {};
    options.headers = options.headers || {};
    options.headers['X-CSRFToken'] = window.CSRF_TOKEN || getCookie('csrftoken');
    if (options.body && typeof options.body === 'object' && !(options.body instanceof FormData)) {
      options.headers['Content-Type'] = 'application/json';
      options.body = JSON.stringify(options.body);
    }
    return fetch(url, options).then(function (resp) {
      var ct = resp.headers.get('content-type') || '';
      if (ct.indexOf('application/json') >= 0) return resp.json();
      return resp.text().then(function (t) { return { _text: t, _status: resp.status }; });
    }).then(function (data) {
      if (data && data._status && data._status >= 400) {
        throw new Error(data._text || ('HTTP ' + data._status));
      }
      return data;
    });
  }

  // ── Drawer (slide-in from the right) ──────────────────────────
  function openDrawer(title, bodyHTML) {
    closeDrawer();
    var mask = document.createElement('div');
    mask.className = 'phm-drawer-mask';
    var drawer = document.createElement('div');
    drawer.className = 'phm-drawer';
    drawer.innerHTML =
      '<div class="phm-drawer-header">' +
        '<h3 class="phm-drawer-title"></h3>' +
        '<button class="phm-drawer-close" aria-label="关闭">&times;</button>' +
      '</div>' +
      '<div class="phm-drawer-body"></div>';
    drawer.querySelector('.phm-drawer-title').textContent = title || '';
    drawer.querySelector('.phm-drawer-body').innerHTML = bodyHTML || '';
    mask.appendChild(drawer);
    document.body.appendChild(mask);
    // Trigger the animation
    requestAnimationFrame(function () {
      mask.classList.add('active');
      drawer.classList.add('active');
    });
    mask.addEventListener('click', closeDrawer);
    drawer.querySelector('.phm-drawer-close').addEventListener('click', closeDrawer);
    // Exposed so the caller can fill it
    PHM._drawerEl = drawer;
    return drawer;
  }

  function closeDrawer() {
    var mask = document.querySelector('.phm-drawer-mask');
    if (mask) {
      var drawer = mask.querySelector('.phm-drawer');
      if (drawer) drawer.classList.remove('active');
      mask.classList.remove('active');
      setTimeout(function () { if (mask.parentNode) mask.parentNode.removeChild(mask); }, 250);
    }
    PHM._drawerEl = null;
  }

  // ── Close the drawer on ESC ───────────────────────────────────
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') closeDrawer();
  });

  // ── Utility functions ─────────────────────────────────────────
  function fmtTimestamp(ts) {
    if (!ts) return '—';
    var d = new Date(ts * 1000);
    if (isNaN(d.getTime())) return String(ts);
    var pad = function (n) { return n < 10 ? '0' + n : n; };
    return d.getUTCFullYear() + '-' + pad(d.getUTCMonth() + 1) + '-' + pad(d.getUTCDate()) +
           ' ' + pad(d.getUTCHours()) + ':' + pad(d.getUTCMinutes()) + ':' + pad(d.getUTCSeconds()) + ' UTC';
  }

  function escapeHTML(s) {
    if (s == null) return '';
    return String(s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  // ── Public exports ────────────────────────────────────────────
  window.PHM = {
    injectTheme: injectTheme,
    toast: toast,
    fetchJSON: fetchJSON,
    openDrawer: openDrawer,
    closeDrawer: closeDrawer,
    fmtTimestamp: fmtTimestamp,
    escapeHTML: escapeHTML,
    _drawerEl: null
  };

  // Inject the theme on boot
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', injectTheme);
  } else {
    injectTheme();
  }
})();
