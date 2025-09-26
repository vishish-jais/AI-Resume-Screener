(function(){
  // Find the current <script> element
  const scripts = document.getElementsByTagName('script');
  const self = scripts[scripts.length - 1];
  const ds = self.dataset || {};

  // Config from data attributes
  const apiBase = (ds.apiBase || '').replace(/\/$/, '');
  const token = ds.token || '';

  // Expose to widget.js
  window.ATS_WIDGET_CONFIG = {
    apiBase: apiBase,
    token: token
  };

  // Avoid duplicate injection
  if (document.getElementById('ats-widget-styles')) return;

  // Inject styles (hosted by the same server as widget)
  // Expect the user to reference this script via your ATS domain, e.g. https://your-domain/static/js/ats-embed.js
  const baseUrl = (function(){
    const a = document.createElement('a');
    a.href = self.src;
    // strip trailing /static/js/ats-embed.js
    return a.href.replace(/\/static\/js\/ats-embed\.js(?:\?.*)?$/, '');
  })();

  const link = document.createElement('link');
  link.id = 'ats-widget-styles';
  link.rel = 'stylesheet';
  link.href = baseUrl + '/static/css/styles.css';
  document.head.appendChild(link);

  // Inject widget.js
  const s = document.createElement('script');
  s.src = baseUrl + '/static/js/widget.js';
  s.defer = true;
  document.head.appendChild(s);
})();
