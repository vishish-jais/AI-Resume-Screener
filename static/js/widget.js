(function(){
  // Allow embedding: read optional config set by ats-embed.js
  const CFG = (window.ATS_WIDGET_CONFIG || {});
  const API_BASE = (CFG.apiBase || '').replace(/\/$/, ''); // no trailing slash
  const WIDGET_TOKEN = CFG.token || null;

  // Create toggle button
  const toggle = document.createElement('button');
  toggle.className = 'ats-widget-toggle';
  toggle.title = 'Resume Summarizer';
  toggle.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" fill="currentColor" viewBox="0 0 16 16"><path d="M8 0a8 8 0 1 0 4.906 14.32L16 16l-1.68-3.094A8 8 0 0 0 8 0zM3 7h10v2H3V7z"/></svg>';
  document.body.appendChild(toggle);

  // Create panel
  const panel = document.createElement('div');
  panel.className = 'ats-widget-panel';
  panel.innerHTML = `
    <div class="ats-widget-header">
      <div><strong>Resume Summarizer</strong></div>
      <button class="btn btn-sm btn-light" id="ats-close">×</button>
    </div>
    <div class="ats-widget-body" id="ats-body">
      <div class="ats-message bot">Hi! Paste resume text or upload a file (PDF/DOCX) and I'll summarize it.</div>
    </div>
    <div class="ats-widget-footer">
      <form id="ats-form">
        <div class="mb-2">
          <textarea class="form-control" id="ats-text" rows="3" placeholder="Paste resume text..."></textarea>
        </div>
        <div class="d-flex gap-2 align-items-center flex-wrap">
          <select id="ats-style" class="form-select form-select-sm" style="width:auto">
            <option value="pointwise" selected>Pointwise</option>
            <option value="crisp">Crisp</option>
            <option value="detailed">Detailed</option>
          </select>
          <input type="file" id="ats-file" class="form-control form-control-sm" accept=".pdf,.doc,.docx,.txt,.png,.jpg,.jpeg,.bmp,.tiff,.webp" multiple />
          <button class="btn btn-primary btn-sm" id="ats-send">Summarize</button>
          <button class="btn btn-secondary btn-sm" id="ats-scan-all" type="button">Scan All</button>
        </div>
      </form>
    </div>`;
  document.body.appendChild(panel);

  const closeBtn = panel.querySelector('#ats-close');
  const body = panel.querySelector('#ats-body');
  const form = panel.querySelector('#ats-form');
  const textEl = panel.querySelector('#ats-text');
  const fileEl = panel.querySelector('#ats-file');
  const styleEl = panel.querySelector('#ats-style');
  const sendBtn = panel.querySelector('#ats-send');
  const scanAllBtn = panel.querySelector('#ats-scan-all');

  function setOpen(open){ panel.style.display = open ? 'flex' : 'none'; }

  toggle.addEventListener('click', ()=> setOpen(panel.style.display !== 'flex'));
  closeBtn.addEventListener('click', ()=> setOpen(false));

  function addMessage(content, who, opts){
    const div = document.createElement('div');
    div.className = 'ats-message ' + (who || 'bot');
    const isHtml = opts && opts.html === true;
    if(isHtml){
      div.innerHTML = content;
    } else {
      // Preserve newlines for plain text
      div.innerHTML = (content || '').replace(/\n/g, '<br>');
    }
    body.appendChild(div);
    body.scrollTop = body.scrollHeight;
  }

  form.addEventListener('submit', async (e)=>{
    e.preventDefault();
    const text = textEl.value.trim();
    const file = fileEl.files[0];
    if(!text && !file){
      addMessage('Please provide resume text or choose a file to upload.', 'bot');
      return;
    }

    if(text){ addMessage(text, 'me'); }
    if(file){ addMessage('Uploaded file: ' + file.name, 'me'); }

    sendBtn.disabled = true; sendBtn.textContent = 'Summarizing...';
    try {
      const formData = new FormData();
      if(text) formData.append('text', text);
      if(file) formData.append('file', file);
      formData.append('style', (styleEl.value || 'pointwise').toLowerCase());
      const headers = {};
      if(WIDGET_TOKEN){ headers['X-Widget-Token'] = WIDGET_TOKEN; }
      const url = (API_BASE ? `${API_BASE}/api/summarize` : '/api/summarize');
      const res = await fetch(url, { method: 'POST', body: formData, headers });
      const data = await res.json();
      if(!data.ok) throw new Error(data.error || 'Failed');
      if(data.summary_html){
        addMessage(data.summary_html, 'bot', { html: true });
      } else {
        addMessage(data.summary, 'bot');
      }
      textEl.value = '';
      fileEl.value = '';
    } catch(err){
      addMessage('Error: ' + err.message, 'bot');
    } finally {
      sendBtn.disabled = false; sendBtn.textContent = 'Summarize';
    }
  });

  // Batch scan: summarize multiple selected resumes in one go
  scanAllBtn.addEventListener('click', async ()=>{
    const files = Array.from(fileEl.files || []);
    if(files.length === 0){
      addMessage('Please select one or more resume files first.', 'bot');
      return;
    }
    scanAllBtn.disabled = true; scanAllBtn.textContent = 'Scanning...';
    try {
      const formData = new FormData();
      files.forEach(f => formData.append('files', f));
      formData.append('style', (styleEl.value || 'pointwise').toLowerCase());
      const headers = {};
      if(WIDGET_TOKEN){ headers['X-Widget-Token'] = WIDGET_TOKEN; }
      const url = (API_BASE ? `${API_BASE}/api/summarize_batch` : '/api/summarize_batch');
      const res = await fetch(url, { method: 'POST', body: formData, headers });
      const data = await res.json();
      if(!data.ok) throw new Error(data.error || 'Failed');
      // Render each file result
      (data.results || []).forEach(r => {
        if(!r.ok){
          addMessage(`${r.filename}: Error - ${r.error}`, 'bot');
          return;
        }
        const header = `<div><strong>${r.filename}</strong></div>`;
        if(r.summary_html){
          addMessage(header + (r.summary_html || ''), 'bot', { html: true });
        } else {
          addMessage(`${r.filename}\n${r.summary}`, 'bot');
        }
      });
      // Clear selection after processing
      fileEl.value = '';
    } catch(err){
      addMessage('Batch error: ' + err.message, 'bot');
    } finally {
      scanAllBtn.disabled = false; scanAllBtn.textContent = 'Scan All';
    }
  });
})();
