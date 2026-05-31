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
  commands: document.getElementById('benchmark-commands'),
  modelCount: document.getElementById('benchmark-model-count'),
  modelList: document.getElementById('benchmark-model-list'),
  refreshModels: document.getElementById('refresh-benchmark-models'),
  selectAll: document.getElementById('select-all-models'),
  clearAll: document.getElementById('clear-selected-models'),
  startBenchmark: document.getElementById('start-benchmark'),
  events: document.getElementById('benchmark-events'),
  logCaption: document.getElementById('benchmark-log-caption'),
  resetLogView: document.getElementById('benchmark-reset-log-view'),
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

function renderRunDiagnosticsLog(job) {
  const lines = [];
  if (job.events?.length) {
    lines.push('=== Job Events ===');
    lines.push(...job.events);
  }

  const failedRuns = job.model_runs.filter((run) => run.error);
  if (failedRuns.length) {
    lines.push('');
    lines.push('=== Failure Summary ===');
    for (const run of failedRuns) {
      lines.push(`${run.display_name || run.model_id}: ${run.error}`);
    }
  }

  const evalLogsAvailable = job.model_runs.filter((run) => run.evaluation_log_path);
  if (evalLogsAvailable.length) {
    lines.push('');
    lines.push('=== Evaluation Logs Available ===');
    for (const run of evalLogsAvailable) {
      lines.push(`Use "Show Full Eval Log" on ${run.display_name || run.model_id}`);
    }
  }

  benchmarkUi.logCaption.textContent = 'Actionable run diagnostics and failure context';
  benchmarkUi.events.textContent = lines.length ? lines.join('\n') : 'No log output yet.';
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

function copyToClipboard(text, button) {
  navigator.clipboard.writeText(text).then(() => {
    const original = button.textContent;
    button.textContent = 'Copied!';
    setTimeout(() => { button.textContent = original; }, 1500);
  }).catch(() => {
    button.textContent = 'Failed';
    setTimeout(() => { button.textContent = 'Copy'; }, 1500);
  });
}

function renderBenchmarkCommands(job) {
  if (!job || !job.model_runs.length) {
    benchmarkUi.commands.className = 'commands-list empty-state';
    benchmarkUi.commands.textContent = 'Select a run to see its commands.';
    return;
  }
  benchmarkUi.commands.className = 'commands-list';

  const sections = job.model_runs.map((run) => {
    const steps = [];
    let missingStepOne = false;

    // Step 1: docker run
    if (run.docker_run_command) {
      steps.push({ label: 'Step 1 — Launch vLLM container', cmd: run.docker_run_command });
    } else if (run.container_name && run.host_port) {
      missingStepOne = true;
    }

    // Step 2: readiness probe
    const probeUrl = run.vllm_probe_url || (run.host_port ? `http://host.docker.internal:${run.host_port}/v1/models` : null);
    if (probeUrl) {
      steps.push({ label: 'Step 2 — Wait for vLLM readiness (poll until 200)', cmd: `curl -s ${probeUrl}` });
    }

    // Step 3: evaluation
    if (run.evaluation_command) {
      steps.push({ label: 'Step 3 — Run SWE-bench evaluation', cmd: run.evaluation_command });
    }

    if (!steps.length) {
      return `<div class="command-block"><div class="command-block-header"><strong>${escapeHtml(run.display_name || run.model_id)}</strong><span class="status-pill ${statusTone(run.status)}">${run.status.replaceAll('_', ' ')}</span></div><div class="meta-line muted">No commands recorded yet — commands are captured as the run progresses.</div></div>`;
    }

    const stepsHtml = steps.map((step, i) => `
      <div class="command-step">
        <div class="command-step-label">${escapeHtml(step.label)}</div>
        <div class="command-step-row">
          <pre class="command-pre">${escapeHtml(step.cmd)}</pre>
          <button class="ghost-button copy-btn" type="button" data-cmd="${i}-${encodeURIComponent(run.model_id)}">Copy</button>
        </div>
      </div>
    `).join('');

    const missingStepOneHtml = missingStepOne
      ? '<div class="meta-line muted">Step 1 launch command is unavailable for this historical run.</div>'
      : '';

    return `<div class="command-block" data-model-id="${escapeHtml(run.model_id)}">
      <div class="command-block-header">
        <strong>${escapeHtml(run.display_name || run.model_id)}</strong>
        <span class="status-pill ${statusTone(run.status)}">${run.status.replaceAll('_', ' ')}</span>
      </div>
      ${missingStepOneHtml}
      ${stepsHtml}
    </div>`;
  });

  benchmarkUi.commands.innerHTML = sections.join('');

  // Wire up copy buttons — store commands in a map keyed by data-cmd
  const cmdMap = new Map();
  job.model_runs.forEach((run) => {
    const probeUrl = run.vllm_probe_url || (run.host_port ? `http://host.docker.internal:${run.host_port}/v1/models` : null);
    const steps = [];
    if (run.docker_run_command) steps.push(run.docker_run_command);
    else steps.push(null);
    if (probeUrl) steps.push(`curl -s ${probeUrl}`);
    else steps.push(null);
    if (run.evaluation_command) steps.push(run.evaluation_command);
    else steps.push(null);
    steps.forEach((cmd, i) => {
      if (cmd) cmdMap.set(`${i}-${encodeURIComponent(run.model_id)}`, cmd);
    });
  });

  for (const btn of benchmarkUi.commands.querySelectorAll('.copy-btn')) {
    btn.addEventListener('click', () => {
      const cmd = cmdMap.get(btn.dataset.cmd);
      if (cmd) copyToClipboard(cmd, btn);
    });
  }
}

function renderBenchmarkJob(job) {
  if (!job) {
    benchmarkUi.caption.textContent = 'No active benchmark job.';
    benchmarkUi.overview.textContent = 'Start a run to see live progress, failures, and final results.';
    benchmarkUi.status.textContent = 'Create or select a run to inspect model-by-model progress.';
    benchmarkUi.events.textContent = 'Waiting for a benchmark run.';
    benchmarkUi.logCaption.textContent = 'Actionable run diagnostics and failure context';
    benchmarkUi.chart.textContent = 'No results yet.';
    benchmarkUi.progressBar.style.width = '0%';
    benchmarkUi.progressText.textContent = 'No benchmark in progress.';
    renderBenchmarkCommands(null);
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

  renderBenchmarkCommands(job);
  if (!benchmarkState.selectedLogModelId) {
    renderRunDiagnosticsLog(job);
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
    benchmarkUi.logCaption.textContent = `Full evaluation log • ${title}`;
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
    benchmarkUi.diagnostics.className = 'discovery-diagnostics empty-state';
    benchmarkUi.diagnostics.textContent = 'Waiting for diagnostics.';
    return;
  }
  benchmarkUi.diagnostics.className = 'discovery-diagnostics';

  const missing = diagnostics.missing_config_directories || [];
  const discovered = diagnostics.discovered_model_ids || [];
  const topLevel = diagnostics.top_level || [];

  const rootOk = diagnostics.exists;
  const stats = [
    { label: 'Hub root', value: diagnostics.hub_root, mono: true },
    { label: 'Root exists', value: rootOk ? 'yes' : 'NO — path not found', warn: !rootOk },
    { label: 'Top-level folders', value: diagnostics.top_level_directory_count ?? '—' },
    { label: 'Config files found', value: diagnostics.candidate_count ?? '—' },
    { label: 'Models discovered', value: diagnostics.discovered_count ?? '—' },
  ];

  const statsHtml = `<div class="diag-facts">${stats.map((s) => `
    <div class="diag-fact${s.warn ? ' diag-warn' : ''}">
      <span class="diag-fact-label">${escapeHtml(s.label)}</span>
      <span class="diag-fact-value${s.mono ? ' diag-mono' : ''}">${escapeHtml(String(s.value))}</span>
    </div>`).join('')}
  </div>`;

  const modelListHtml = discovered.length
    ? `
      <div class="diag-section">
        <div class="diag-section-title">Discovered models</div>
        <div class="diag-model-ids">${discovered.map((id) => `<span class="tag-chip">${escapeHtml(id)}</span>`).join('')}</div>
      </div>`
    : '';

  const missingHtml = missing.length
    ? `<div class="diag-warn-block"><strong>Folders with no config.json:</strong> ${missing.map((m) => `<code>${escapeHtml(m)}</code>`).join(', ')}</div>`
    : '';

  const folderCardsHtml = topLevel.slice(0, 20).map((entry) => {
    const ok = entry.config_count > 0 || entry.heuristic_model_like;
    const sample = entry.sample_configs?.length
      ? `<div class="diag-folder-sample">${escapeHtml(entry.sample_configs[0])}</div>`
      : '';
    return `
      <article class="diag-folder${ok ? '' : ' diag-folder-warn'}">
        <div class="diag-folder-header">
          <strong class="diag-mono">${escapeHtml(entry.name)}</strong>
          <span class="diag-folder-status">${entry.config_count > 0 ? `${entry.config_count} config${entry.config_count === 1 ? '' : 's'}` : (entry.heuristic_model_like ? 'auto-detected' : 'missing config')}</span>
        </div>
        <div class="diag-folder-meta">configs ${entry.config_count} • auto-detected ${entry.heuristic_model_like ? 'yes' : 'no'}</div>
        ${sample}
      </article>`;
  }).join('');

  const folderListHtml = topLevel.length ? `
    <div class="diag-section">
      <div class="diag-section-title">Folders scanned</div>
      <div class="diag-folder-list">${folderCardsHtml}</div>
    </div>` : '';

  benchmarkUi.diagnostics.innerHTML = statsHtml + modelListHtml + missingHtml + folderListHtml;
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

benchmarkUi.resetLogView.addEventListener('click', async () => {
  benchmarkState.selectedLogModelId = null;
  const job = selectedBenchmarkJob();
  if (job) {
    renderRunDiagnosticsLog(job);
  } else {
    benchmarkUi.logCaption.textContent = 'Actionable run diagnostics and failure context';
    benchmarkUi.events.textContent = 'Waiting for a benchmark run.';
  }
});

bootstrapBenchmarkPage().catch((error) => {
  logBenchmark(error.message);
});