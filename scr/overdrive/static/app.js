const state = {
  models: [],
  containers: [],
  stats: [],
  selectedModelId: null,
};

const ui = {
  hubRoot: document.getElementById('hub-root'),
  modelCount: document.getElementById('model-count'),
  modelList: document.getElementById('model-list'),
  selectedTitle: document.getElementById('selected-title'),
  selectedSubtitle: document.getElementById('selected-subtitle'),
  selectedSummary: document.getElementById('selected-summary'),
  recommendationGrid: document.getElementById('recommendation-grid'),
  containers: document.getElementById('containers'),
  stats: document.getElementById('stats'),
  logs: document.getElementById('logs'),
  logsCaption: document.getElementById('logs-caption'),
  commandPreview: document.getElementById('command-preview'),
  commandCaption: document.getElementById('command-caption'),
  eventLog: document.getElementById('event-log'),
  form: document.getElementById('settings-form'),
  refreshModels: document.getElementById('refresh-models'),
  planAction: document.getElementById('plan-action'),
  launchAction: document.getElementById('launch-action'),
  saveProfileAction: document.getElementById('save-profile-action'),
  stopAction: document.getElementById('stop-action'),
  cleanupAction: document.getElementById('cleanup-action'),
};

function selectedModel() {
  return state.models.find((model) => model.model_id === state.selectedModelId) || null;
}

function appendLog(message) {
  ui.eventLog.textContent = `${new Date().toLocaleTimeString()}  ${message}\n${ui.eventLog.textContent}`.trim();
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || 'Request failed.');
  }
  return payload;
}

function currentSettings() {
  const formData = new FormData(ui.form);
  const payload = {};
  for (const [key, rawValue] of formData.entries()) {
    if (rawValue === '') {
      payload[key] = null;
      continue;
    }
    if (key === 'kv_cache_dtype') {
      payload[key] = rawValue;
      continue;
    }
    payload[key] = Number(rawValue);
  }
  return payload;
}

function setFormValues(model) {
  const settings = model.recommendations;
  document.getElementById('port').value = settings.preferred_port ?? '';
  document.getElementById('max-model-len').value = settings.max_model_len ?? '';
  document.getElementById('tensor-parallel').value = settings.tensor_parallel_size ?? 1;
  document.getElementById('kv-cache-dtype').value = settings.kv_cache_dtype ?? '';
  document.getElementById('gpu-budget').value = settings.gpu_memory_budget_gb ?? '';
}

function formatCommandPreview(preview) {
  if (!preview) {
    return 'No command preview available.';
  }
  const lines = [
    `Image: ${preview.image}`,
    `Ports: host ${preview.host_port} -> container ${preview.container_port}`,
    '',
    'Model Mapping:',
    `Source on host: ${preview.model_source_path || 'n/a'}`,
    `Mounted in container: ${preview.model_container_path || '/models/current'}`,
    `vLLM --model arg: ${preview.model_container_path || '/models/current'}`,
    'Why this is expected: the selected host path is bind-mounted to the container path above.',
    '',
    'Inner vLLM Command:',
    preview.shell,
  ];
  if (preview.docker_shell) {
    lines.push('', 'Full Docker Launch (equivalent):', preview.docker_shell);
  }
  return lines.join('\n');
}

async function refreshCommandPreview() {
  const model = selectedModel();
  if (!model) {
    ui.commandCaption.textContent = 'Resolved from the current form values';
    ui.commandPreview.textContent = 'Select a model to preview launch commands and how model paths are mapped into the container.';
    return;
  }
  try {
    const payload = await fetchJson(`/api/models/${encodeURIComponent(model.model_id)}/plan`, {
      method: 'POST',
      body: JSON.stringify(currentSettings()),
    });
    ui.commandCaption.textContent = payload.display;
    ui.commandPreview.textContent = formatCommandPreview(payload.command_preview);
  } catch (error) {
    ui.commandCaption.textContent = 'Unable to compute preview';
    ui.commandPreview.textContent = error.message;
  }
}

function renderModels() {
  ui.hubRoot.textContent = document.body.dataset.hubRoot;
  ui.modelCount.textContent = `${state.models.length} loaded`;
  if (!state.models.length) {
    ui.modelList.innerHTML = '<div class="status-card empty-state">No models discovered.</div>';
    return;
  }
  ui.modelList.innerHTML = state.models.map((model) => `
    <button class="model-card ${model.model_id === state.selectedModelId ? 'active' : ''}" data-model-id="${model.model_id}" type="button">
      <strong>${model.display_name}</strong>
      <div class="meta-line">${model.architecture}</div>
      <div class="meta-line">${model.snapshot_path}</div>
    </button>
  `).join('');
  for (const button of ui.modelList.querySelectorAll('[data-model-id]')) {
    button.addEventListener('click', () => {
      state.selectedModelId = button.dataset.modelId;
      const model = selectedModel();
      if (model) {
        setFormValues(model);
      }
      renderModels();
      renderSelected();
      refreshCommandPreview();
      refreshLogs();
    });
  }
}

function renderSelected() {
  const model = selectedModel();
  if (!model) {
    ui.selectedTitle.textContent = 'Select a model';
    ui.selectedSubtitle.textContent = '';
    ui.selectedSummary.textContent = 'Choose a discovered model to edit launch settings.';
    ui.recommendationGrid.innerHTML = '';
    return;
  }
  ui.selectedTitle.textContent = model.model_id;
  ui.selectedSubtitle.textContent = `${model.architecture} • ${model.dtype_display}`;
  ui.selectedSummary.textContent = model.hardware_summary;
  ui.recommendationGrid.innerHTML = [
    ['Recommended Port', model.recommendations.preferred_port],
    ['Max Model Length', model.recommendations.max_model_len],
    ['Tensor Parallel', model.recommendations.tensor_parallel_size],
    ['KV Cache', model.recommendations.kv_cache_dtype || 'auto'],
    ['GPU Budget', `${model.recommendations.gpu_memory_budget_gb} GiB`],
  ].map(([label, value]) => `
    <div class="recommendation-card">
      <strong>${label}</strong>
      <div class="meta-line">${value}</div>
    </div>
  `).join('');
}

function renderRuntime() {
  if (!state.containers.length) {
    ui.containers.innerHTML = 'No active Overdrive containers.';
  } else {
    ui.containers.innerHTML = state.containers.map((item) => `
      <div class="status-card">
        <strong>${item.model_id || item.name}</strong>
        <div class="meta-line">${item.status} • port ${item.host_port ?? 'n/a'}</div>
        <div class="meta-line">${item.image}</div>
      </div>
    `).join('');
  }

  if (!state.stats.length) {
    ui.stats.innerHTML = 'No live stats available.';
  } else {
    ui.stats.innerHTML = state.stats.map((item) => `
      <div class="status-card">
        <strong>${item.name}</strong>
        <div class="meta-line">cpu ${item.cpu_percent ?? 'n/a'}%</div>
        <div class="meta-line">mem ${item.memory_usage_gb ?? 'n/a'}/${item.memory_limit_gb ?? 'n/a'} GiB</div>
        <div class="meta-line">net ${item.network_rx_mb ?? 'n/a'}/${item.network_tx_mb ?? 'n/a'} MB</div>
      </div>
    `).join('');
  }
}

async function refreshModels(preserveSelection = true) {
  const models = await fetchJson('/api/models');
  state.models = models;
  if (!preserveSelection || !state.selectedModelId || !models.some((item) => item.model_id === state.selectedModelId)) {
    state.selectedModelId = models[0]?.model_id || null;
    if (models[0]) {
      setFormValues(models[0]);
    }
  }
  renderModels();
  renderSelected();
  await refreshCommandPreview();
}

async function refreshRuntime() {
  const payload = await fetchJson('/api/runtime');
  state.containers = payload.containers;
  state.stats = payload.stats;
  renderRuntime();
}

async function refreshLogs() {
  const model = selectedModel();
  if (!model) {
    ui.logsCaption.textContent = 'No running container selected.';
    ui.logs.textContent = 'No logs yet.';
    return;
  }
  const payload = await fetchJson(`/api/logs/${encodeURIComponent(model.model_id)}`);
  if (!payload.container_name) {
    ui.logsCaption.textContent = 'Selected model is not running.';
    ui.logs.textContent = 'No logs yet.';
    return;
  }
  ui.logsCaption.textContent = payload.container_name;
  ui.logs.textContent = payload.lines.length ? payload.lines.join('\n') : 'No logs yet.';
}

async function performAction(path, successMessage) {
  const model = selectedModel();
  if (!model) {
    appendLog('Select a model first.');
    return null;
  }
  try {
    const payload = await fetchJson(path.replace('{modelId}', encodeURIComponent(model.model_id)), {
      method: 'POST',
      body: JSON.stringify(currentSettings()),
    });
    appendLog(successMessage(payload));
    return payload;
  } catch (error) {
    appendLog(error.message);
    return null;
  }
}

ui.refreshModels.addEventListener('click', async () => {
  await refreshModels(false);
  appendLog('Model list refreshed.');
});

ui.planAction.addEventListener('click', async () => {
  const payload = await performAction('/api/models/{modelId}/plan', (payload) => payload.display);
  if (payload) {
    ui.commandCaption.textContent = payload.display;
    ui.commandPreview.textContent = formatCommandPreview(payload.command_preview);
  }
});

ui.launchAction.addEventListener('click', async () => {
  const payload = await performAction(
    '/api/models/{modelId}/launch',
    (result) => `${result.status}: ${result.container_name} on port ${result.host_port}`,
  );
  if (payload) {
    await refreshRuntime();
    await refreshLogs();
  }
});

ui.saveProfileAction.addEventListener('click', async () => {
  const payload = await performAction(
    '/api/models/{modelId}/profile',
    (result) => `Saved profile for ${result.model_id} to ${result.path}`,
  );
  if (payload) {
    await refreshModels(true);
  }
});

ui.stopAction.addEventListener('click', async () => {
  const model = selectedModel();
  if (!model) {
    appendLog('Select a model first.');
    return;
  }
  try {
    const payload = await fetchJson(`/api/models/${encodeURIComponent(model.model_id)}/stop`, { method: 'POST' });
    appendLog(`Stopped ${payload.model_id}: ${payload.stopped}`);
    await refreshRuntime();
    await refreshLogs();
  } catch (error) {
    appendLog(error.message);
  }
});

ui.cleanupAction.addEventListener('click', async () => {
  try {
    const payload = await fetchJson('/api/cleanup', { method: 'POST' });
    appendLog(`Stopped managed containers: ${payload.stopped_count}`);
    await refreshRuntime();
    await refreshLogs();
  } catch (error) {
    appendLog(error.message);
  }
});

for (const element of ui.form.querySelectorAll('input')) {
  element.addEventListener('input', () => {
    refreshCommandPreview();
  });
}

async function bootstrap() {
  await refreshModels(false);
  await refreshRuntime();
  await refreshLogs();
  appendLog('Web console ready.');
  window.setInterval(async () => {
    await refreshRuntime();
    await refreshLogs();
  }, 2000);
}

bootstrap().catch((error) => {
  appendLog(error.message);
});