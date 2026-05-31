const benchmarkState = {
  models: [],
  selectedModels: new Set(),
  currentJobId: null,
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
  form: document.getElementById('benchmark-form'),
  diagnostics: document.getElementById('discovery-diagnostics'),
};

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
  };
}

function renderBenchmarkJob(job) {
  if (!job) {
    benchmarkUi.caption.textContent = 'No active benchmark job.';
    benchmarkUi.status.textContent = 'Create a run to see model-by-model progress.';
    benchmarkUi.chart.textContent = 'No results yet.';
    return;
  }

  benchmarkUi.caption.textContent = `${job.status} • ${job.config.dataset_name} • ${job.config.split}`;
  benchmarkUi.status.innerHTML = job.model_runs.map((run) => `
    <div class="status-card">
      <strong>${run.display_name || run.model_id}</strong>
      <div class="meta-line">status: ${run.status}</div>
      <div class="meta-line">port: ${run.host_port ?? 'n/a'}</div>
      <div class="meta-line">resolved: ${run.resolved_instances}/${run.submitted_instances}</div>
      <div class="meta-line">rate: ${run.resolution_rate ?? 0}%</div>
      ${run.launch_command ? `<div class="meta-line command-line">launch: ${run.launch_command}</div>` : ''}
      ${run.evaluation_command ? `<div class="meta-line command-line">eval: ${run.evaluation_command}</div>` : ''}
      ${run.evaluation_log_path ? `<div class="meta-line">eval log: ${run.evaluation_log_path}</div>` : ''}
      ${run.error ? `<div class="meta-line">error: ${run.error}</div>` : ''}
    </div>
  `).join('');

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

  if (job.events?.length) {
    benchmarkUi.events.textContent = job.events.join('\n');
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
  if (!benchmarkState.currentJobId && jobs.length) {
    benchmarkState.currentJobId = jobs[0].job_id;
  }
  if (benchmarkState.currentJobId) {
    await refreshBenchmarkJob();
  } else {
    renderBenchmarkJob(null);
  }
}

async function refreshBenchmarkJob() {
  if (!benchmarkState.currentJobId) {
    renderBenchmarkJob(null);
    return;
  }
  try {
    const job = await benchmarkJson(`/api/benchmarks/jobs/${encodeURIComponent(benchmarkState.currentJobId)}`);
    renderBenchmarkJob(job);
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
    const job = await benchmarkJson('/api/benchmarks/jobs', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    benchmarkState.currentJobId = job.job_id;
    renderBenchmarkJob(job);
    logBenchmark(`Started benchmark job ${job.job_id}.`);
  } catch (error) {
    logBenchmark(error.message);
  }
});

async function bootstrapBenchmarkPage() {
  await refreshBenchmarkModels();
  await refreshBenchmarkJobList();
  window.setInterval(refreshBenchmarkJob, 2000);
}

bootstrapBenchmarkPage().catch((error) => {
  logBenchmark(error.message);
});