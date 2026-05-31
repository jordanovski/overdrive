const hubState = {
  results: [],
};

const hubUi = {
  hubRoot: document.getElementById('hub-root'),
  searchForm: document.getElementById('hub-search-form'),
  searchButton: document.getElementById('search-models'),
  resultCount: document.getElementById('result-count'),
  results: document.getElementById('hub-results'),
  searchEvents: document.getElementById('search-events'),
  downloadLog: document.getElementById('download-log'),
};

function logSearch(message) {
  hubUi.searchEvents.textContent = `${new Date().toLocaleTimeString()}  ${message}\n${hubUi.searchEvents.textContent}`.trim();
}

function logDownload(message) {
  hubUi.downloadLog.textContent = `${new Date().toLocaleTimeString()}  ${message}\n${hubUi.downloadLog.textContent}`.trim();
}

async function hubJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  const raw = await response.text();
  let payload = {};
  if (raw.trim()) {
    try {
      payload = JSON.parse(raw);
    } catch {
      payload = { detail: raw.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim() };
    }
  }
  if (!response.ok) {
    throw new Error(payload.detail || 'Hub request failed.');
  }
  return payload;
}

function searchPayload() {
  const formData = new FormData(hubUi.searchForm);
  const payload = {
    query: String(formData.get('query') || '').trim(),
    quantization: String(formData.get('quantization') || '').trim() || null,
    author: String(formData.get('author') || '').trim() || null,
    pipeline_tag: String(formData.get('pipeline_tag') || '').trim() || null,
    library: String(formData.get('library') || '').trim() || null,
    min_downloads: Number(formData.get('min_downloads') || 0),
    limit: Number(formData.get('limit') || 25),
    dgx_ready_only: Boolean(formData.get('dgx_ready_only')),
    sort: 'downloads',
  };
  if (!Number.isFinite(payload.min_downloads) || payload.min_downloads < 0) {
    payload.min_downloads = 0;
  }
  if (!Number.isFinite(payload.limit) || payload.limit < 1) {
    payload.limit = 25;
  }
  return payload;
}

function renderTags(tags) {
  if (!tags || !tags.length) {
    return '<span class="meta-line">No tags</span>';
  }
  return tags.slice(0, 10).map((tag) => `<span class="tag-chip">${tag}</span>`).join('');
}

function renderResults() {
  hubUi.hubRoot.textContent = document.body.dataset.hubRoot;
  hubUi.resultCount.textContent = `${hubState.results.length} found`;
  if (!hubState.results.length) {
    hubUi.results.innerHTML = '<div class="status-card empty-state">No models matched the filters.</div>';
    return;
  }

  hubUi.results.innerHTML = hubState.results.map((item) => `
    <article class="model-card hub-result-card" data-model-id="${item.id}">
      <div class="hub-result-header">
        <strong>${item.id}</strong>
        <button class="success-button" type="button" data-download-model="${item.id}">Download</button>
      </div>
      <div class="meta-line">downloads=${item.downloads} • likes=${item.likes} • task=${item.pipeline_tag || 'n/a'} • lib=${item.library_name || 'n/a'}</div>
      <div class="tag-row">${renderTags(item.dgx_tags)}</div>
      <div class="tag-row">${renderTags(item.tags)}</div>
    </article>
  `).join('');

  for (const button of hubUi.results.querySelectorAll('[data-download-model]')) {
    button.addEventListener('click', async () => {
      const modelId = button.dataset.downloadModel;
      if (!modelId) {
        return;
      }
      button.disabled = true;
      try {
        const payload = await hubJson('/api/hub/download', {
          method: 'POST',
          body: JSON.stringify({ model_id: modelId }),
        });
        logDownload(`Downloaded ${payload.model_id} to ${payload.local_dir}`);
      } catch (error) {
        logDownload(`Download failed for ${modelId}: ${error.message}`);
      } finally {
        button.disabled = false;
      }
    });
  }
}

async function runSearch() {
  hubUi.searchButton.disabled = true;
  try {
    const payload = await hubJson('/api/hub/search', {
      method: 'POST',
      body: JSON.stringify(searchPayload()),
    });
    hubState.results = payload.models;
    renderResults();
    logSearch(`Search complete with ${payload.count} result(s).`);
  } catch (error) {
    logSearch(error.message);
  } finally {
    hubUi.searchButton.disabled = false;
  }
}

hubUi.searchButton.addEventListener('click', runSearch);
hubUi.searchForm.addEventListener('submit', (event) => {
  event.preventDefault();
  runSearch();
});

hubUi.hubRoot.textContent = document.body.dataset.hubRoot;
renderResults();
