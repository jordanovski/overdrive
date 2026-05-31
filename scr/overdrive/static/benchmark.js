const benchmarkState = {
  models: [],
  jobs: [],
  selectedModels: new Set(),
  selectedJobId: null,
  selectedLogModelId: null,
  diagnostics: null,
};

const benchmarkUi = {
  hubRoot: document.getElementById('hub-root'),
  modelCount: document.getElementById('benchmark-model-count'),
  modelList: document.getElementById('benchmark-model-list'),
  refreshModels: document.getElementById('refresh-benchmark-models'),
  selectAll: document.getElementById('select-all-models'),
  clearAll: document.getElementById('clear-selected-models'),
  startBenchmark: document.getElementById('start-benchmark'),
  events: document.getElementById('benchmark-events'),
  status: document.getElementById('benchmark-status'),
  caption: document.getElementById('benchmark-job-caption'),
  chart: document.getElementById('benchmark-chart'),
  history: document.getElementById('benchmark-history'),
  overview: document.getElementById('benchmark-overview'),
  progressBar: document.getElementById('benchmark-progress-bar'),
  progressText: document.getElementById('benchmark-progress-text'),
  selectionSummary: document.getElementById('benchmark-selection-summary'),
  form: document.getElementById('benchmark-form'),
  diagnostics: document.getElementById('discovery-diagnostics'),
};

const FINISHED_RUN_STATES = new Set(['completed', 'failed']);

function selectedBenchmarkJob() {
  return benchmarkState.jobs.find((job) => job.job_id === benchmarkState.selectedJobId) || null;
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function statusTone(status) {
  if (status === 'completed') return 'success';
  if (status === 'failed') return 'danger';
  if (status === 'running' || status === 'launching' || status === 'waiting_for_vllm' || status === 'generating_predictions' || status === 'evaluating') return 'warning';
  return 'neutral';
}

function benchmarkProgress(job) {
  if (!job || !job.model_runs.length) {
    return { finished: 0, total: 0, completed: 0, failed: 0, percent: 0 };
  }
  const total = job.model_runs.length;
  const completed = job.model_runs.filter((run) => run.status === 'completed').length;
  const failed = job.model_runs.filter((run) => run.status === 'failed').length;
  const finished = completed + failed;
  const percent = total ? Math.round((finished / total) * 100) : 0;
  return { finished, total, completed, failed, percent };
}

function formatDateTime(value) {
  if (!value) {
    return 'n/a';
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

function logBenchmark(message) {
  benchmarkUi.events.textContent = `${new Date().toLocaleTimeString()}  ${message}\n${benchmarkUi.events.textContent}`.trim();
}

async function benchmarkJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || 'Benchmark request failed.');
  }
  return payload;
}

function renderBenchmarkModels() {
  benchmarkUi.hubRoot.textContent = document.body.dataset.hubRoot;
  benchmarkUi.modelCount.textContent = `${benchmarkState.models.length} loaded`;
  benchmarkUi.selectionSummary.textContent = benchmarkState.selectedModels.size
    ? `${benchmarkState.selectedModels.size} model(s) selected for the next run.`
    : 'Select one or more local models to create a run.';
  if (!benchmarkState.models.length) {
    benchmarkUi.modelList.innerHTML = '<div class="status-card empty-state">No discovered models available.</div>';
    return;
  }
  benchmarkUi.modelList.innerHTML = benchmarkState.models.map((model) => `
    <label class="model-card checkbox-card">
      <input type="checkbox" value="${model.model_id}" ${benchmarkState.selectedModels.has(model.model_id) ? 'checked' : ''}>
      <span>
        <strong>${model.display_name}</strong>
        <span class="meta-line">${model.architecture}</span>
        <span class="meta-line">${model.hardware_summary}</span>
      </span>
    </label>
  `).join('');
  for (const input of benchmarkUi.modelList.querySelectorAll('input[type="checkbox"]')) {
    input.addEventListener('change', () => {
      if (input.checked) {
        benchmarkState.selectedModels.add(input.value);
      } else {
        benchmarkState.selectedModels.delete(input.value);
      }
      benchmarkUi.selectionSummary.textContent = benchmarkState.selectedModels.size
        ? `${benchmarkState.selectedModels.size} model(s) selected for the next run.`
        : 'Select one or more local models to create a run.';
    });
  }
}

function benchmarkFormPayload() {
  const formData = new FormData(benchmarkUi.form);
  return {
    model_ids: Array.from(benchmarkState.selectedModels),
    dataset_name: String(formData.get('dataset_name') || '').trim(),
    split: String(formData.get('split') || '').trim(),
    instance_limit: Number(formData.get('instance_limit') || 0) || null,
    max_eval_workers: Number(formData.get('max_eval_workers') || 0) || 1,
    reuse_cached_results: Boolean(formData.get('reuse_cached_results')),
  };
}

function renderBenchmarkJob(job) {
  if (!job) {
    benchmarkUi.caption.textContent = 'No active benchmark job.';
    benchmarkUi.overview.textContent = 'Start a run to see live progress, failures, and final results.';
    benchmarkUi.status.textContent = 'Create or select a run to inspect model-by-model progress.';
    benchmarkUi.events.textContent = 'Waiting for a benchmark run.';
    benchmarkUi.chart.textContent = 'No results yet.';
    benchmarkUi.progressBar.style.width = '0%';
    benchmarkUi.progressText.textContent = 'No benchmark in progress.';
    return;
  }

  const progress = benchmarkProgress(job);
  const currentRun = job.model_runs.find((run) => run.model_id === job.current_model_id) || null;
  benchmarkUi.caption.textContent = `${job.status} • ${job.config.dataset_name} • ${job.config.split}`;
  benchmarkUi.overview.innerHTML = `
    <div class="status-card benchmark-overview-card">
      <strong>${job.status === 'running' ? 'Benchmark in progress' : 'Benchmark summary'}</strong>
      <div class="meta-line">job ${job.job_id}</div>
      <div class="meta-line">created ${formatDateTime(job.created_at)}</div>
      <div class="meta-line">${currentRun ? `current model: ${currentRun.display_name || currentRun.model_id}` : 'no model currently running'}</div>
    </div>
    <div class="status-card benchmark-overview-card">
      <strong>${progress.finished}/${progress.total} model runs finished</strong>
      <div class="meta-line">completed ${progress.completed}</div>
      <div class="meta-line">failed ${progress.failed}</div>
      <div class="meta-line">selected ${job.config.model_ids.length}</div>
    </div>
    <div class="status-card benchmark-overview-card">
      <strong>${job.config.dataset_name}</strong>
      <div class="meta-line">split ${job.config.split}</div>
      <div class="meta-line">instance limit ${job.config.instance_limit ?? 'all'}</div>
      <div class="meta-line">reuse cache ${job.config.reuse_cached_results ? 'yes' : 'no'}</div>
    </div>
  `;
  benchmarkUi.progressBar.style.width = `${progress.percent}%`;
  benchmarkUi.progressText.textContent = progress.total
    ? `${progress.percent}% complete • ${progress.finished} of ${progress.total} model runs finished`
    : 'No model runs in this job.';
  benchmarkUi.status.innerHTML = job.model_runs.map((run) => `
    <div class="status-card benchmark-model-card">
      <div class="benchmark-model-header">
        <strong>${run.display_name || run.model_id}</strong>
        <span class="status-pill ${statusTone(run.status)}">${run.status.replaceAll('_', ' ')}</span>
      </div>
      <div class="benchmark-metrics-grid">
        <div class="benchmark-metric"><span class="meta-line">Port</span><strong>${run.host_port ?? 'n/a'}</strong></div>
        <div class="benchmark-metric"><span class="meta-line">Resolved</span><strong>${run.resolved_instances}/${run.submitted_instances}</strong></div>
        <div class="benchmark-metric"><span class="meta-line">Rate</span><strong>${run.resolution_rate ?? 0}%</strong></div>
        <div class="benchmark-metric"><span class="meta-line">Finished</span><strong>${formatDateTime(run.finished_at)}</strong></div>
      </div>
      ${run.launch_command ? `<div class="meta-line command-line">launch: ${run.launch_command}</div>` : ''}
      ${run.evaluation_command ? `<div class="meta-line command-line">eval: ${run.evaluation_command}</div>` : ''}
      ${run.predictions_path ? `<div class="meta-line">predictions: ${run.predictions_path}</div>` : ''}
      ${run.report_path ? `<div class="meta-line">report: ${run.report_path}</div>` : ''}
      ${run.evaluation_log_path ? `<div class="meta-line">eval log: ${run.evaluation_log_path}</div>` : ''}
      ${run.evaluation_log_path ? `<div class="action-row compact-actions"><button class="ghost-button" type="button" data-open-full-log="${run.model_id}">Show Full Eval Log</button></div>` : ''}
      ${run.error ? `<div class="meta-line benchmark-error">error: ${escapeHtml(run.error)}</div>` : ''}
      ${run.evaluation_log_excerpt ? `<pre class="event-log benchmark-inline-log">${escapeHtml(run.evaluation_log_excerpt)}</pre>` : ''}
    </div>
  `).join('');

  for (const button of benchmarkUi.status.querySelectorAll('[data-open-full-log]')) {
    button.addEventListener('click', async () => {
      benchmarkState.selectedLogModelId = button.dataset.openFullLog;
      await refreshSelectedBenchmarkLog();
    });
  }

  const finishedRuns = job.model_runs.filter((run) => run.status === 'completed' || run.status === 'failed');
  if (!finishedRuns.length) {
    benchmarkUi.chart.textContent = 'No results yet.';
  } else {
    benchmarkUi.chart.innerHTML = finishedRuns.map((run) => {
      const rate = Number(run.resolution_rate || 0);
      const failedClass = run.status === 'failed' ? 'failed' : '';
      return `
        <div class="chart-row">
          <div class="chart-label">
            <span>${run.display_name || run.model_id}</span>
            <span>${rate}%</span>
          </div>
          <div class="chart-track">
            <div class="chart-bar ${failedClass}" style="width: ${Math.max(0, Math.min(rate, 100))}%"></div>
          </div>
        </div>
      `;
    }).join('');
  }

  const logLines = [];
  if (job.events?.length) {
    logLines.push(...job.events);
  }
  for (const run of job.model_runs) {
    if (run.error) {
      logLines.push(`ERROR ${run.display_name || run.model_id}: ${run.error}`);
    }
    if (run.evaluation_log_excerpt) {
      logLines.push(`--- evaluation log tail for ${run.display_name || run.model_id} ---`);
      logLines.push(run.evaluation_log_excerpt);
    }
  }
  if (!benchmarkState.selectedLogModelId) {
    benchmarkUi.events.textContent = logLines.length ? logLines.join('\n\n') : 'No log output yet.';
  }
}

async function refreshSelectedBenchmarkLog() {
  if (!benchmarkState.selectedJobId || !benchmarkState.selectedLogModelId) {
    return;
  }
  try {
    const payload = await benchmarkJson(
      `/api/benchmarks/jobs/${encodeURIComponent(benchmarkState.selectedJobId)}/logs/${encodeURIComponent(benchmarkState.selectedLogModelId)}`,
    );
    const title = payload.display_name || payload.model_id;
    const pathLine = payload.evaluation_log_path ? `Path: ${payload.evaluation_log_path}\n\n` : '';
    benchmarkUi.events.textContent = `Full evaluation log for ${title}\n${pathLine}${payload.content || 'No evaluation log content available.'}`;
  } catch (error) {
    logBenchmark(error.message);
  }
}

function renderBenchmarkHistory() {
  if (!benchmarkState.jobs.length) {
    benchmarkUi.history.textContent = 'No benchmark runs yet.';
    return;
  }
  benchmarkUi.history.innerHTML = benchmarkState.jobs.map((job) => {
    const progress = benchmarkProgress(job);
    const activeClass = job.job_id === benchmarkState.selectedJobId ? 'active' : '';
    return `
      <button class="model-card benchmark-history-card ${activeClass}" type="button" data-job-id="${job.job_id}">
        <div class="benchmark-model-header">
          <strong>${job.config.dataset_name}</strong>
          <span class="status-pill ${statusTone(job.status)}">${job.status}</span>
        </div>
        <div class="meta-line">${job.config.model_ids.length} model(s) • ${job.config.split}</div>
        <div class="meta-line">${progress.finished}/${progress.total} finished • ${progress.completed} completed • ${progress.failed} failed</div>
        <div class="meta-line">created ${formatDateTime(job.created_at)}</div>
      </button>
    `;
  }).join('');

  for (const button of benchmarkUi.history.querySelectorAll('[data-job-id]')) {
    button.addEventListener('click', async () => {
      benchmarkState.selectedJobId = button.dataset.jobId;
      benchmarkState.selectedLogModelId = null;
      renderBenchmarkHistory();
      await refreshBenchmarkJob();
    });
  }
}

function renderDiscoveryDiagnostics() {
  const diagnostics = benchmarkState.diagnostics;
  if (!diagnostics) {
    benchmarkUi.diagnostics.textContent = 'Waiting for diagnostics.';
    return;
  }

  const missing = diagnostics.missing_config_directories || [];
  const discovered = diagnostics.discovered_model_ids || [];
  const lines = [
    `hub_root=${diagnostics.hub_root}`,
    `exists=${diagnostics.exists}`,
    `top_level_directory_count=${diagnostics.top_level_directory_count}`,
    `config_candidate_count=${diagnostics.candidate_count}`,
    `discovered_count=${diagnostics.discovered_count}`,
    `discovered_model_ids=${discovered.join(', ') || 'none'}`,
  ];

  if (missing.length) {
    lines.push(`missing_config_directories=${missing.join(', ')}`);
  }

  const topLevel = diagnostics.top_level || [];
  for (const entry of topLevel.slice(0, 20)) {
    lines.push(`- ${entry.name}: config_count=${entry.config_count}`);
    if (entry.sample_configs?.length) {
      lines.push(`  sample=${entry.sample_configs.join(', ')}`);
    }
  }

  benchmarkUi.diagnostics.textContent = lines.join('\n');
}

async function refreshBenchmarkModels() {
  benchmarkState.models = await benchmarkJson('/api/models');
  benchmarkState.diagnostics = await benchmarkJson('/api/models/diagnostics');
  renderBenchmarkModels();
  renderDiscoveryDiagnostics();
}

async function refreshBenchmarkJobList() {
  const jobs = await benchmarkJson('/api/benchmarks/jobs');
  benchmarkState.jobs = jobs;
  const activeJob = jobs.find((job) => job.status === 'running' || job.status === 'queued') || jobs[0] || null;
  if (!benchmarkState.selectedJobId || !jobs.some((job) => job.job_id === benchmarkState.selectedJobId)) {
    benchmarkState.selectedJobId = activeJob?.job_id || null;
  }
  renderBenchmarkHistory();
  if (benchmarkState.selectedJobId) {
    await refreshBenchmarkJob();
  } else {
    renderBenchmarkJob(null);
  }
}

async function refreshBenchmarkJob() {
  if (!benchmarkState.selectedJobId) {
    renderBenchmarkJob(null);
    return;
  }
  try {
    const job = await benchmarkJson(`/api/benchmarks/jobs/${encodeURIComponent(benchmarkState.selectedJobId)}`);
    benchmarkState.jobs = benchmarkState.jobs.map((entry) => entry.job_id === job.job_id ? job : entry);
    renderBenchmarkHistory();
    renderBenchmarkJob(job);
    if (benchmarkState.selectedLogModelId) {
      const stillExists = job.model_runs.some((run) => run.model_id === benchmarkState.selectedLogModelId && run.evaluation_log_path);
      if (stillExists) {
        await refreshSelectedBenchmarkLog();
      } else {
        benchmarkState.selectedLogModelId = null;
        renderBenchmarkJob(job);
      }
    }
  } catch (error) {
    logBenchmark(error.message);
  }
}

benchmarkUi.refreshModels.addEventListener('click', async () => {
  await refreshBenchmarkModels();
  logBenchmark('Benchmark model list refreshed.');
});

benchmarkUi.selectAll.addEventListener('click', () => {
  benchmarkState.selectedModels = new Set(benchmarkState.models.map((model) => model.model_id));
  renderBenchmarkModels();
});

benchmarkUi.clearAll.addEventListener('click', () => {
  benchmarkState.selectedModels.clear();
  renderBenchmarkModels();
});

benchmarkUi.startBenchmark.addEventListener('click', async () => {
  const payload = benchmarkFormPayload();
  if (!payload.model_ids.length) {
    logBenchmark('Select at least one model.');
    return;
  }
  try {
    benchmarkUi.startBenchmark.disabled = true;
    benchmarkUi.startBenchmark.textContent = 'Starting...';
    const job = await benchmarkJson('/api/benchmarks/jobs', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    benchmarkState.selectedJobId = job.job_id;
    benchmarkState.selectedLogModelId = null;
    benchmarkState.jobs = [job, ...benchmarkState.jobs.filter((entry) => entry.job_id !== job.job_id)];
    renderBenchmarkHistory();
    renderBenchmarkJob(job);
    logBenchmark(`Started benchmark job ${job.job_id}.`);
  } catch (error) {
    logBenchmark(error.message);
  } finally {
    benchmarkUi.startBenchmark.disabled = false;
    benchmarkUi.startBenchmark.textContent = 'Start Benchmark';
  }
});

async function bootstrapBenchmarkPage() {
  await refreshBenchmarkModels();
  await refreshBenchmarkJobList();
  window.setInterval(refreshBenchmarkJobList, 2000);
}

bootstrapBenchmarkPage().catch((error) => {
  logBenchmark(error.message);
});