/**
 * InsightPilot Frontend Application
 * Real backend integration with FastAPI, LangGraph, and PostgreSQL pgvector.
 * Real API integration, real streaming responses, real error states.
 */

// ── Application State ──────────────────────────────────────────────────────
const state = {
  apiKey: localStorage.getItem('insightpilot_api_key') || '',
  tenantId: localStorage.getItem('insightpilot_tenant_id') || 'default',
  apiBase: localStorage.getItem('insightpilot_api_base') || window.location.origin,
  sessionId: 'sess-' + Math.random().toString(36).substring(2, 10),
  documents: [],
  totalTokens: 0,
  activeCitation: null,
  isProcessing: false,
  isUploading: false
};

// ── DOM References ─────────────────────────────────────────────────────────
const dom = {
  docList: document.getElementById('documents-container'),
  docCountBadge: document.getElementById('doc-count-badge'),
  chatStream: document.getElementById('chat-messages-stream'),
  queryInput: document.getElementById('chat-query-input'),
  chatForm: document.getElementById('chat-input-form'),
  sendBtn: document.getElementById('btn-send-message'),
  sessionLabel: document.getElementById('session-display-id'),
  tokenCounter: document.getElementById('token-usage-counter'),
  tenantLabel: document.getElementById('active-tenant-label'),
  systemStatus: document.getElementById('system-status-indicator'),
  systemStatusText: document.getElementById('system-status-text'),
  inspectorContent: document.getElementById('inspector-content-area'),
  emptyCitationState: document.getElementById('citation-empty-state'),
  dropzone: document.getElementById('file-dropzone'),
  fileInput: document.getElementById('file-picker-input'),
  btnUpload: document.getElementById('btn-submit-upload'),
  docCompanyInput: document.getElementById('doc-company-input'),
  docYearInput: document.getElementById('doc-year-input'),
  btnClearChat: document.getElementById('btn-clear-chat'),
  configModal: document.getElementById('config-modal'),
  btnConfig: document.getElementById('btn-api-config'),
  btnCloseModal: document.getElementById('btn-close-modal'),
  btnCancelModal: document.getElementById('btn-cancel-modal'),
  btnSaveModal: document.getElementById('btn-save-modal'),
  cfgApiKey: document.getElementById('cfg-api-key'),
  cfgTenantId: document.getElementById('cfg-tenant-id'),
  cfgApiBase: document.getElementById('cfg-api-base'),
  quickPromptChips: document.getElementById('quick-prompt-chips')
};

// ── Initialization ─────────────────────────────────────────────────────────
async function init() {
  dom.sessionLabel.textContent = state.sessionId;
  dom.tenantLabel.textContent = state.tenantId;
  dom.cfgApiKey.value = state.apiKey;
  dom.cfgTenantId.value = state.tenantId;
  dom.cfgApiBase.value = state.apiBase;

  setupEventListeners();
  await ensureAuth();
  await checkBackendHealth();
  await loadDocuments();
}

// ── Authentication Management ──────────────────────────────────────────────
async function ensureAuth() {
  // If no API key is in localStorage, attempt to fetch a local development session key
  if (!state.apiKey) {
    try {
      const res = await fetch(`${state.apiBase}/v1/auth/dev-key`);
      if (res.ok) {
        const data = await res.json();
        if (data.api_key) {
          state.apiKey = data.api_key;
          state.tenantId = data.tenant_id || state.tenantId;
          localStorage.setItem('insightpilot_api_key', state.apiKey);
          localStorage.setItem('insightpilot_tenant_id', state.tenantId);
          dom.cfgApiKey.value = state.apiKey;
          dom.tenantLabel.textContent = state.tenantId;
        }
      }
    } catch (_) {
      // In production or isolated mode, dev-key endpoint is disabled
    }
  }
}

// ── Backend Health Check ───────────────────────────────────────────────────
async function checkBackendHealth() {
  try {
    const res = await fetch(`${state.apiBase}/health`, { signal: AbortSignal.timeout(4000) });
    if (res.ok) {
      dom.systemStatusText.textContent = 'Agent Online';
      dom.systemStatus.style.background = 'rgba(16, 185, 129, 0.12)';
      dom.systemStatus.style.color = 'var(--text-emerald)';
      dom.systemStatus.style.borderColor = 'rgba(16, 185, 129, 0.25)';
      return true;
    }
  } catch (_) {
    // Handled below
  }

  dom.systemStatusText.textContent = 'Backend Offline';
  dom.systemStatus.style.background = 'rgba(239, 68, 68, 0.12)';
  dom.systemStatus.style.color = 'var(--text-rose)';
  dom.systemStatus.style.borderColor = 'rgba(239, 68, 68, 0.25)';
  return false;
}

// ── Document Library (GET /v1/documents & DELETE /v1/documents) ───────────
async function loadDocuments() {
  if (!state.apiKey) {
    renderDocumentList();
    return;
  }

  try {
    const res = await fetch(`${state.apiBase}/v1/documents`, {
      headers: {
        'Authorization': `Bearer ${state.apiKey}`
      },
      signal: AbortSignal.timeout(6000)
    });

    if (res.ok) {
      const data = await res.json();
      state.documents = Array.isArray(data) ? data : [];
      renderDocumentList();
    } else if (res.status === 401) {
      state.documents = [];
      renderDocumentList();
      appendSystemError('Authentication failed (401). Please configure a valid Bearer API Key.');
    } else {
      state.documents = [];
      renderDocumentList();
      appendSystemError(`Failed to load document library: HTTP ${res.status}`);
    }
  } catch (err) {
    state.documents = [];
    renderDocumentList();
    console.error('Error fetching documents from backend:', err);
  }
}

function renderDocumentList() {
  dom.docCountBadge.textContent = `${state.documents.length} Docs`;
  dom.docList.innerHTML = '';

  if (state.documents.length === 0) {
    dom.docList.innerHTML = `
      <div style="padding: 1.5rem 0.5rem; text-align: center; color: var(--text-dim); font-size: 0.8rem; line-height: 1.5;">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="margin: 0 auto 0.5rem; display: block; opacity: 0.4;">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
          <polyline points="14 2 14 8 20 8"></polyline>
        </svg>
        No indexed documents found for tenant <strong>${escapeHtml(state.tenantId)}</strong>.<br>
        Upload a PDF, TXT, DOCX, or MD report to begin.
      </div>
    `;
    return;
  }

  state.documents.forEach((doc, idx) => {
    const card = document.createElement('div');
    card.className = `doc-card ${idx === 0 ? 'active' : ''}`;
    const docName = doc.document_name || doc.source || 'Unnamed Document';
    const company = doc.company ? `<span class="tag-badge tag-cyan">${escapeHtml(doc.company)}</span>` : '';
    const year = doc.financial_year ? `<span class="tag-badge tag-emerald">${escapeHtml(doc.financial_year)}</span>` : '';
    const chunks = doc.chunk_count ? `<span class="tag-badge">${doc.chunk_count} chunks</span>` : '';
    const pages = doc.page_count ? `<span class="tag-badge">${doc.page_count} pgs</span>` : '';

    card.innerHTML = `
      <div class="doc-header">
        <span class="doc-name" title="${escapeHtml(docName)}">${escapeHtml(docName)}</span>
        <button class="btn-icon" style="padding: 2px 4px; font-size: 0.7rem;" title="Delete document" onclick="deleteDocument('${escapeHtml(doc.document_id)}')">✕</button>
      </div>
      <div class="doc-tags">
        ${company}
        ${year}
        ${chunks}
        ${pages}
      </div>
    `;
    dom.docList.appendChild(card);
  });
}

window.deleteDocument = async function(docId) {
  if (!confirm('Are you sure you want to delete this document from the knowledge base?')) {
    return;
  }

  try {
    const res = await fetch(`${state.apiBase}/v1/documents/${docId}`, {
      method: 'DELETE',
      headers: {
        'Authorization': `Bearer ${state.apiKey}`
      }
    });

    if (res.status === 204 || res.ok) {
      appendSystemNotice(`Document ${docId} deleted successfully.`);
      await loadDocuments();
    } else {
      const err = await res.json().catch(() => ({}));
      appendSystemError(`Failed to delete document: ${err.message || 'HTTP ' + res.status}`);
    }
  } catch (err) {
    appendSystemError(`Error deleting document: ${err.message}`);
  }
};

// ── Document Ingestion (POST /v1/documents/ingest) ──────────────────────────
async function handleFiles(file) {
  if (!file) return;

  const allowedExts = ['.pdf', '.txt', '.docx', '.md'];
  const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
  if (!allowedExts.includes(ext)) {
    alert(`File type "${ext}" not supported. Allowed formats: PDF, TXT, DOCX, MD.`);
    return;
  }

  if (file.size > 20 * 1024 * 1024) {
    alert('File size exceeds the 20MB limit.');
    return;
  }

  if (!state.apiKey) {
    alert('Missing API key. Please open Config to enter your Bearer API Key.');
    dom.configModal.classList.add('open');
    return;
  }

  state.isUploading = true;
  dom.btnUpload.disabled = true;
  dom.btnUpload.innerHTML = `<span class="upload-loading-spinner"></span> Ingesting...`;

  const company = dom.docCompanyInput.value.trim();
  const year = dom.docYearInput.value.trim();

  const url = new URL(`${state.apiBase}/v1/documents/ingest`);
  url.searchParams.set('document_name', file.name);
  if (company) url.searchParams.set('company', company);
  if (year) url.searchParams.set('financial_year', year);

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch(url.toString(), {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${state.apiKey}`
      },
      body: formData
    });

    const data = await res.json();

    if (res.status === 201) {
      appendSystemNotice(
        `Successfully ingested "${data.source || data.document_name}" (ID: ${data.document_id}, ${data.chunks_indexed} chunks indexed, status: ${data.status}).`
      );
      dom.docCompanyInput.value = '';
      dom.docYearInput.value = '';
      dom.fileInput.value = '';
      await loadDocuments();
    } else {
      const msg = data.detail?.message || data.message || (typeof data.detail === 'string' ? data.detail : 'Upload failed');
      appendSystemError(`Ingestion failed (${res.status}): ${msg}`);
    }
  } catch (err) {
    console.error('Ingestion request error:', err);
    appendSystemError(`Ingestion connection error: ${err.message || 'Could not connect to backend server'}`);
  } finally {
    state.isUploading = false;
    dom.btnUpload.disabled = false;
    dom.btnUpload.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M12 5v14M5 12h14"/>
      </svg>
      Ingest & Index
    `;
  }
}

// ── Agent Query Execution (POST /v1/agent/run) ──────────────────────────────
async function handleSubmitQuery() {
  const query = dom.queryInput.value.trim();
  if (!query || state.isProcessing) return;

  dom.queryInput.value = '';
  state.isProcessing = true;
  dom.sendBtn.disabled = true;
  dom.sendBtn.style.opacity = '0.6';

  // 1. Render User message
  appendUserMessage(query);

  // 2. Render Thinking state
  const thinkingCard = appendThinkingMessage();
  dom.chatStream.scrollTop = dom.chatStream.scrollHeight;

  // 3. Check auth
  if (!state.apiKey) {
    thinkingCard.remove();
    appendAssistantError('Missing Bearer API Key. Click Config in the navigation bar to set your API Key.');
    state.isProcessing = false;
    dom.sendBtn.disabled = false;
    dom.sendBtn.style.opacity = '1';
    return;
  }

  // 4. Send query to POST /v1/agent/run
  try {
    const res = await fetch(`${state.apiBase}/v1/agent/run`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${state.apiKey}`
      },
      body: JSON.stringify({
        session_id: state.sessionId,
        message: query
      })
    });

    thinkingCard.remove();

    if (res.ok) {
      const data = await res.json();
      appendAssistantResponse({
        answer: data.response || 'No response returned from agent.',
        route: data.route,
        confidence: data.confidence,
        citations: data.citations || [],
        tools: data.tools || [],
        tokens: data.usage || { input_tokens: 0, output_tokens: 0 }
      });
    } else {
      let errMsg = `Backend returned HTTP ${res.status}`;
      try {
        const errJson = await res.json();
        errMsg = errJson.message || (typeof errJson.detail === 'string' ? errJson.detail : (errJson.detail?.message || JSON.stringify(errJson.detail)));
      } catch (_) {}
      appendAssistantError(errMsg);
    }
  } catch (netErr) {
    thinkingCard.remove();
    console.error('Agent execution network error:', netErr);
    appendAssistantError(`Connection error: Could not reach backend at ${state.apiBase}. Verify that the server is running.`);
  } finally {
    state.isProcessing = false;
    dom.sendBtn.disabled = false;
    dom.sendBtn.style.opacity = '1';
    dom.chatStream.scrollTop = dom.chatStream.scrollHeight;
  }
}

// ── Message Renderers ──────────────────────────────────────────────────────
function appendUserMessage(text) {
  const row = document.createElement('div');
  row.className = 'message-row user';
  row.innerHTML = `<div class="user-bubble">${escapeHtml(text)}</div>`;
  dom.chatStream.appendChild(row);
}

function appendThinkingMessage() {
  const row = document.createElement('div');
  row.className = 'message-row assistant';
  row.innerHTML = `
    <div class="assistant-card" style="opacity: 0.85;">
      <div class="reasoning-trail">
        <span class="reasoning-pill pill-guard-pass">Input Guard: Scanning...</span>
        <span class="reasoning-pill pill-route-doc">LangGraph: Routing...</span>
      </div>
      <div style="display: flex; align-items: center; gap: 0.5rem; color: var(--text-muted); font-size: 0.85rem;">
        <span class="status-dot"></span> Generating evidence-grounded answer...
      </div>
    </div>
  `;
  dom.chatStream.appendChild(row);
  return row;
}

function appendAssistantResponse(data) {
  const row = document.createElement('div');
  row.className = 'message-row assistant';

  // Update token counter with real numbers
  if (data.tokens) {
    const turnTokens = (data.tokens.input_tokens || 0) + (data.tokens.output_tokens || 0);
    state.totalTokens += turnTokens;
    dom.tokenCounter.textContent = `${state.totalTokens.toLocaleString()} tokens`;
  }

  // Format route pill if returned by backend
  let routePill = '';
  if (data.route === 'document_search') {
    routePill = `<span class="reasoning-pill pill-route-doc">Route: Document Search</span>`;
  } else if (data.route === 'calculator') {
    routePill = `<span class="reasoning-pill pill-route-calc">Route: Calculator</span>`;
  } else if (data.route === 'web_search') {
    routePill = `<span class="reasoning-pill pill-route-web">Route: Web Search</span>`;
  } else if (data.route === 'direct') {
    routePill = `<span class="reasoning-pill pill-confidence">Route: Direct</span>`;
  } else if (data.route) {
    routePill = `<span class="reasoning-pill pill-confidence">Route: ${escapeHtml(data.route)}</span>`;
  }

  // Tool trace pills
  let toolPills = '';
  if (Array.isArray(data.tools) && data.tools.length > 0) {
    toolPills = data.tools
      .map(t => `<span class="reasoning-pill pill-guard-pass">Tool: ${escapeHtml(t.tool_name)} (${escapeHtml(t.status)})</span>`)
      .join(' ');
  }

  // Confidence pill
  let confPill = '';
  if (data.confidence) {
    confPill = `<span class="reasoning-pill pill-confidence">Confidence: ${escapeHtml(data.confidence)}</span>`;
  }

  // Format citations in text: replace [1], [2] with clickable links
  let formattedAnswer = escapeHtml(data.answer);
  if (Array.isArray(data.citations) && data.citations.length > 0) {
    data.citations.forEach((c, idx) => {
      const citeNumber = idx + 1;
      const jsonPayload = encodeURIComponent(JSON.stringify(c));
      const pageText = c.page ? ` (p.${c.page})` : '';
      const linkTag = `<a href="#citation-${citeNumber}" class="citation-link" data-evidence="${jsonPayload}" title="View source citation from ${escapeHtml(c.source)}${pageText}">[${citeNumber}]</a>`;
      formattedAnswer = formattedAnswer.replaceAll(`[${citeNumber}]`, linkTag);
    });
  }

  // Markdown bold, paragraphs, and line breaks
  formattedAnswer = formattedAnswer
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br>');

  row.innerHTML = `
    <div class="assistant-card">
      <div class="reasoning-trail">
        ${routePill}
        ${toolPills}
        ${confPill}
      </div>

      <div class="answer-content">
        <p>${formattedAnswer}</p>
      </div>
    </div>
  `;

  dom.chatStream.appendChild(row);

  // Auto-display first citation in right inspector if present
  if (Array.isArray(data.citations) && data.citations.length > 0) {
    showCitationInspector(data.citations[0]);
  } else {
    clearCitationInspector();
  }
}

function appendAssistantError(errorMessage) {
  const row = document.createElement('div');
  row.className = 'message-row assistant';
  row.innerHTML = `
    <div class="assistant-card error-state">
      <div class="reasoning-trail">
        <span class="reasoning-pill pill-error">Error: Execution Failed</span>
      </div>
      <div class="answer-content" style="color: var(--text-rose);">
        <p><strong>Error:</strong> ${escapeHtml(errorMessage)}</p>
      </div>
    </div>
  `;
  dom.chatStream.appendChild(row);
  clearCitationInspector();
}

function appendSystemNotice(text) {
  const row = document.createElement('div');
  row.className = 'message-row';
  row.style.alignItems = 'center';
  row.innerHTML = `
    <div style="font-size: 0.75rem; color: var(--text-emerald); background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.2); padding: 0.4rem 0.85rem; border-radius: var(--radius-full); font-family: var(--font-mono);">
      ✓ ${escapeHtml(text)}
    </div>
  `;
  dom.chatStream.appendChild(row);
}

function appendSystemError(text) {
  const row = document.createElement('div');
  row.className = 'message-row';
  row.style.alignItems = 'center';
  row.innerHTML = `
    <div class="system-notice-error">
      ✕ ${escapeHtml(text)}
    </div>
  `;
  dom.chatStream.appendChild(row);
}

// ── Citation Inspector Drawer ──────────────────────────────────────────────
function showCitationInspector(citation) {
  state.activeCitation = citation;
  const pageStr = citation.page ? `Page: <strong>${citation.page}</strong>` : 'Page: <strong>N/A</strong>';
  const verifiedStr = citation.verified ? '<span style="color: var(--text-emerald);">✓ Verified Grounding</span>' : '<span style="color: var(--text-dim);">Unverified</span>';
  const snippet = citation.snippet || '';

  dom.inspectorContent.innerHTML = `
    <div class="evidence-card">
      <div class="evidence-header">
        <div class="evidence-source">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
          </svg>
          ${escapeHtml(citation.source)}
        </div>
      </div>

      <div style="font-size: 0.72rem; color: var(--text-dim); display: flex; gap: 0.75rem; margin-top: 0.25rem;">
        <span>${pageStr}</span>
        <span>Tenant: <strong>${escapeHtml(state.tenantId)}</strong></span>
        <span>${verifiedStr}</span>
      </div>

      <div class="evidence-text">
        ${highlightFinancialNumbers(escapeHtml(snippet))}
      </div>

      <div style="display: flex; justify-content: flex-end; margin-top: 0.25rem;">
        <button class="btn btn-secondary btn-sm" id="btn-copy-excerpt">
          Copy Excerpt
        </button>
      </div>
    </div>
  `;

  const copyBtn = document.getElementById('btn-copy-excerpt');
  if (copyBtn) {
    copyBtn.addEventListener('click', () => {
      navigator.clipboard.writeText(snippet);
      alert('Source excerpt copied to clipboard.');
    });
  }
}

function clearCitationInspector() {
  dom.inspectorContent.innerHTML = '';
  if (dom.emptyCitationState) {
    dom.inspectorContent.appendChild(dom.emptyCitationState);
  }
}

function highlightFinancialNumbers(text) {
  return text.replace(/(\$[\d,.]+\s*(?:billion|million|B|M)?|\b\d+(?:\.\d+)?%|\b\d+\s*bps\b)/gi, '<mark>$1</mark>');
}

// ── Settings Modal ─────────────────────────────────────────────────────────
function saveConfig() {
  state.apiKey = dom.cfgApiKey.value.trim();
  state.tenantId = dom.cfgTenantId.value.trim() || 'default';
  state.apiBase = dom.cfgApiBase.value.trim() || window.location.origin;

  localStorage.setItem('insightpilot_api_key', state.apiKey);
  localStorage.setItem('insightpilot_tenant_id', state.tenantId);
  localStorage.setItem('insightpilot_api_base', state.apiBase);

  dom.tenantLabel.textContent = state.tenantId;
  dom.configModal.classList.remove('open');
  checkBackendHealth();
  loadDocuments();
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

// ── Event Listeners ────────────────────────────────────────────────────────
function setupEventListeners() {
  // Chat form submit
  dom.chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    handleSubmitQuery();
  });

  // Quick prompt chips
  if (dom.quickPromptChips) {
    dom.quickPromptChips.addEventListener('click', (e) => {
      const chip = e.target.closest('.prompt-chip');
      if (chip && chip.dataset.prompt) {
        dom.queryInput.value = chip.dataset.prompt;
        handleSubmitQuery();
      }
    });
  }

  // Clear chat / New session
  dom.btnClearChat.addEventListener('click', () => {
    state.sessionId = 'sess-' + Math.random().toString(36).substring(2, 10);
    dom.sessionLabel.textContent = state.sessionId;
    state.totalTokens = 0;
    dom.tokenCounter.textContent = '0 tokens';
    const rows = dom.chatStream.querySelectorAll('.message-row');
    rows.forEach(r => r.remove());
    clearCitationInspector();
  });

  // Dropzone drag-and-drop
  dom.dropzone.addEventListener('click', () => dom.fileInput.click());
  dom.dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dom.dropzone.classList.add('dragover');
  });
  dom.dropzone.addEventListener('dragleave', () => dom.dropzone.classList.remove('dragover'));
  dom.dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dom.dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
      handleFiles(e.dataTransfer.files[0]);
    }
  });

  dom.fileInput.addEventListener('change', () => {
    if (dom.fileInput.files.length) {
      handleFiles(dom.fileInput.files[0]);
    }
  });

  dom.btnUpload.addEventListener('click', () => {
    if (dom.fileInput.files.length) {
      handleFiles(dom.fileInput.files[0]);
    } else {
      alert('Please select or drop a financial document file first.');
    }
  });

  // Settings modal
  dom.btnConfig.addEventListener('click', () => dom.configModal.classList.add('open'));
  dom.btnCloseModal.addEventListener('click', () => dom.configModal.classList.remove('open'));
  dom.btnCancelModal.addEventListener('click', () => dom.configModal.classList.remove('open'));
  dom.btnSaveModal.addEventListener('click', saveConfig);

  // Citation inspector click delegation in chat stream
  dom.chatStream.addEventListener('click', (e) => {
    const citation = e.target.closest('.citation-link');
    if (citation) {
      e.preventDefault();
      const citeData = citation.dataset.evidence;
      if (citeData) {
        try {
          const parsed = JSON.parse(decodeURIComponent(citeData));
          showCitationInspector(parsed);
        } catch (err) {
          console.error('Error parsing citation payload:', err);
        }
      }
    }
  });
}

// Boot application
document.addEventListener('DOMContentLoaded', init);
