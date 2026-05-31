"""Sequential SWE-bench orchestration for local vLLM-backed models."""

from __future__ import annotations

import json
import os
import platform
import re
import shlex
import subprocess
import sys
import threading
import time
import uuid
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import requests

from overdrive.hardware import (
    GPUDevice,
    detect_gpus,
    recommended_gpu_budget_gb,
    recommended_kv_cache_dtype,
    recommended_max_model_len,
    recommended_tensor_parallel_size,
)
from overdrive.models import BenchmarkConfig, BenchmarkJob, BenchmarkModelRun, ModelMetadata
from overdrive.paths import benchmarks_root
from overdrive.state import EngineStateManager

VLLM_READY_TIMEOUT_SECONDS = 600
VLLM_READY_POLL_SECONDS = 2
SWE_SYSTEM_PROMPT = (
    "You are a software engineer solving a SWE-bench issue. "
    "Return only a valid unified git diff patch that can be applied with git apply. "
    "Do not include explanations, markdown fences, or extra prose."
)


def _runtime_host() -> str:
    return os.environ.get("OVERDRIVE_RUNTIME_HOST", "127.0.0.1")


def _runtime_base_url(host_port: int) -> str:
    return f"http://{_runtime_host()}:{host_port}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _slugify_model_id(model_id: str) -> str:
    return model_id.replace("/", "-").replace("_", "-").lower()


def _recommended_settings(
    manager: EngineStateManager,
    model: ModelMetadata,
    gpus: list[GPUDevice],
) -> dict[str, int | float | str | None]:
    return {
        "preferred_port": manager.runtime.reserve_port(model.profile.preferred_port),
        "max_model_len": recommended_max_model_len(model, gpus),
        "tensor_parallel_size": recommended_tensor_parallel_size(model, gpus),
        "kv_cache_dtype": recommended_kv_cache_dtype(model, gpus),
        "gpu_memory_budget_gb": recommended_gpu_budget_gb(model, gpus),
    }


def _stringify_field(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, dict)):
        return "\n".join(str(item) for item in value)
    return str(value)


def _build_prompt(instance: dict[str, object]) -> str:
    sections = [
        f"Repository: {instance.get('repo', 'unknown')}",
        f"Instance ID: {instance.get('instance_id', 'unknown')}",
        "Issue:",
        _stringify_field(instance.get("problem_statement")),
    ]
    hints = _stringify_field(instance.get("hints_text"))
    if hints.strip():
        sections.extend(["Hints:", hints])
    fail_to_pass = _stringify_field(instance.get("FAIL_TO_PASS"))
    if fail_to_pass.strip():
        sections.extend(["Tests to fix:", fail_to_pass])
    pass_to_pass = _stringify_field(instance.get("PASS_TO_PASS"))
    if pass_to_pass.strip():
        sections.extend(["Tests that must keep passing:", pass_to_pass])
    sections.extend(
        [
            "Requirements:",
            "- Return only a unified diff patch.",
            "- Start the patch with diff --git when possible.",
            "- Make the smallest change that resolves the issue.",
        ]
    )
    return "\n\n".join(section for section in sections if section)


def _extract_patch(output: str) -> str:
    fenced = re.search(r"```(?:diff|patch)?\s*(.*?)```", output, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
        if candidate:
            return candidate
    diff_index = output.find("diff --git")
    if diff_index >= 0:
        return output[diff_index:].strip()
    return output.strip()


class BenchmarkService:
    def __init__(
        self,
        manager: EngineStateManager,
        *,
        gpus: list[GPUDevice] | None = None,
        dataset_loader: Callable[..., object] | None = None,
        background_runner: Callable[[Callable[[], None]], None] | None = None,
    ) -> None:
        self.manager = manager
        self.gpus = gpus if gpus is not None else detect_gpus()
        self._dataset_loader = dataset_loader or self._default_dataset_loader
        self._background_runner = background_runner or self._start_background
        self._lock = threading.Lock()
        self._jobs: dict[str, BenchmarkJob] = self._load_jobs()

    @staticmethod
    def _default_dataset_loader(dataset_name: str, split: str) -> object:
        from datasets import load_dataset

        return load_dataset(dataset_name, split=split)

    def list_jobs(self) -> list[BenchmarkJob]:
        with self._lock:
            jobs = [self._hydrate_job_view(job.model_copy(deep=True)) for job in self._jobs.values()]
        return sorted(jobs, key=lambda job: job.created_at, reverse=True)

    def get_job(self, job_id: str) -> BenchmarkJob:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            job = self._jobs[job_id].model_copy(deep=True)
        return self._hydrate_job_view(job)

    def get_model_run_log(self, job_id: str, model_id: str) -> dict[str, object]:
        job = self.get_job(job_id)
        model_run = next((item for item in job.model_runs if item.model_id == model_id), None)
        if model_run is None:
            raise KeyError(model_id)
        log_path = model_run.evaluation_log_path
        full_log = self._read_full_log(log_path)
        return {
            "job_id": job_id,
            "model_id": model_id,
            "display_name": model_run.display_name,
            "status": model_run.status,
            "evaluation_log_path": str(log_path) if log_path else None,
            "content": full_log,
        }

    def create_job(self, config: BenchmarkConfig) -> BenchmarkJob:
        if not config.model_ids:
            raise ValueError("Select at least one model.")
        with self._lock:
            if any(job.status in {"queued", "running"} for job in self._jobs.values()):
                raise RuntimeError("A benchmark job is already running.")
            job = BenchmarkJob(
                job_id=uuid.uuid4().hex,
                config=config,
                model_runs=[BenchmarkModelRun(model_id=model_id) for model_id in config.model_ids],
                events=[f"Queued benchmark for {len(config.model_ids)} model(s)."],
            )
            self._jobs[job.job_id] = job
            self._persist_job(job)
        self._background_runner(lambda: self._run_job(job.job_id))
        return self.get_job(job.job_id)

    def _start_background(self, func: Callable[[], None]) -> None:
        thread = threading.Thread(target=func, daemon=True)
        thread.start()

    def _job_root(self, job_id: str) -> Path:
        root = benchmarks_root() / job_id
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _jobs_dir(self) -> Path:
        path = benchmarks_root() / "jobs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _job_snapshot_path(self, job_id: str) -> Path:
        return self._jobs_dir() / f"{job_id}.json"

    def _load_jobs(self) -> dict[str, BenchmarkJob]:
        jobs: dict[str, BenchmarkJob] = {}
        jobs_dir = self._jobs_dir()
        for path in sorted(jobs_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                job = BenchmarkJob.model_validate(payload)
            except (OSError, json.JSONDecodeError, ValueError):
                continue
            jobs[job.job_id] = job
        return jobs

    def _persist_job(self, job: BenchmarkJob) -> None:
        path = self._job_snapshot_path(job.job_id)
        path.write_text(json.dumps(job.model_dump(mode="json"), indent=2), encoding="utf-8")

    def _hydrate_job_view(self, job: BenchmarkJob) -> BenchmarkJob:
        for model_run in job.model_runs:
            model_run.evaluation_log_excerpt = self._read_log_excerpt(model_run.evaluation_log_path)
        return job

    @staticmethod
    def _read_log_excerpt(path: Path | None, *, max_lines: int = 40) -> str | None:
        if path is None or not path.exists():
            return None
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return None
        if not lines:
            return None
        return "\n".join(lines[-max_lines:])

    @staticmethod
    def _read_full_log(path: Path | None) -> str | None:
        if path is None or not path.exists():
            return None
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    def _cache_index_path(self) -> Path:
        cache_root = benchmarks_root() / "cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        return cache_root / "index.json"

    def _load_cache_index(self) -> dict[str, dict[str, object]]:
        path = self._cache_index_path()
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        normalized: dict[str, dict[str, object]] = {}
        for key, value in payload.items():
            if isinstance(key, str) and isinstance(value, dict):
                normalized[key] = value
        return normalized

    def _write_cache_index(self, cache_index: dict[str, dict[str, object]]) -> None:
        path = self._cache_index_path()
        path.write_text(json.dumps(cache_index, indent=2, sort_keys=True), encoding="utf-8")

    def _benchmark_cache_key(
        self,
        *,
        model_id: str,
        config: BenchmarkConfig,
        settings: dict[str, int | float | str | None],
        instance_ids: list[str],
    ) -> str:
        payload = {
            "model_id": model_id,
            "dataset_name": config.dataset_name,
            "split": config.split,
            "instance_limit": config.instance_limit,
            "timeout_seconds": config.timeout_seconds,
            "temperature": config.temperature,
            "max_response_tokens": config.max_response_tokens,
            "max_eval_workers": config.max_eval_workers,
            "settings": settings,
            "instance_ids": instance_ids,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return digest

    def _load_cached_result(
        self,
        *,
        model_id: str,
        config: BenchmarkConfig,
        settings: dict[str, int | float | str | None],
        instance_ids: list[str],
    ) -> dict[str, object] | None:
        cache_index = self._load_cache_index()
        cache_key = self._benchmark_cache_key(
            model_id=model_id,
            config=config,
            settings=settings,
            instance_ids=instance_ids,
        )
        entry = cache_index.get(cache_key)
        if not entry:
            return None
        report_path = Path(str(entry.get("report_path", "")))
        predictions_path = Path(str(entry.get("predictions_path", "")))
        if not report_path.exists() or not predictions_path.exists():
            return None
        return entry

    def _save_cached_result(
        self,
        *,
        model_id: str,
        config: BenchmarkConfig,
        settings: dict[str, int | float | str | None],
        instance_ids: list[str],
        report_path: Path,
        predictions_path: Path,
        evaluation_log_path: Path,
        evaluation_command: str,
        submitted_instances: int,
        completed_instances: int,
        resolved_instances: int,
        resolution_rate: float | None,
    ) -> None:
        cache_index = self._load_cache_index()
        cache_key = self._benchmark_cache_key(
            model_id=model_id,
            config=config,
            settings=settings,
            instance_ids=instance_ids,
        )
        cache_index[cache_key] = {
            "model_id": model_id,
            "dataset_name": config.dataset_name,
            "split": config.split,
            "instance_limit": config.instance_limit,
            "saved_at": _now().isoformat(),
            "report_path": str(report_path),
            "predictions_path": str(predictions_path),
            "evaluation_log_path": str(evaluation_log_path),
            "evaluation_command": evaluation_command,
            "submitted_instances": submitted_instances,
            "completed_instances": completed_instances,
            "resolved_instances": resolved_instances,
            "resolution_rate": resolution_rate,
        }
        self._write_cache_index(cache_index)

    def _mutate_job(self, job_id: str, mutation: Callable[[BenchmarkJob], None]) -> None:
        with self._lock:
            job = self._jobs[job_id]
            mutation(job)
            job.updated_at = _now()
            self._persist_job(job)

    def _append_event(self, job: BenchmarkJob, message: str) -> None:
        job.events.insert(0, message)

    def _run_job(self, job_id: str) -> None:
        try:
            self._mutate_job(
                job_id,
                lambda job: (
                    setattr(job, "status", "running"),
                    self._append_event(job, "Benchmark job started."),
                ),
            )
            job = self.get_job(job_id)
            instances = self._load_instances(job.config)
            if not instances:
                raise RuntimeError("No SWE-bench instances available for the selected dataset.")
            for model_run in job.model_runs:
                self._run_model(job_id, model_run.model_id, job.config, instances)
            self._mutate_job(
                job_id,
                lambda current: (
                    setattr(current, "status", "completed"),
                    setattr(current, "finished_at", _now()),
                    setattr(current, "current_model_id", None),
                    self._append_event(current, "Benchmark job finished."),
                ),
            )
        except Exception as exc:
            error_message = str(exc)
            self._mutate_job(
                job_id,
                lambda job: (
                    setattr(job, "status", "failed"),
                    setattr(job, "finished_at", _now()),
                    setattr(job, "error", error_message),
                    setattr(job, "current_model_id", None),
                    self._append_event(job, f"Benchmark job failed: {error_message}"),
                ),
            )

    def _run_model(
        self,
        job_id: str,
        model_id: str,
        config: BenchmarkConfig,
        instances: list[dict[str, object]],
    ) -> None:
        launch_result = None
        model = self.manager.get_model(model_id)
        settings = _recommended_settings(self.manager, model, self.gpus)
        instance_ids = [str(item["instance_id"]) for item in instances]
        if config.reuse_cached_results:
            cached = self._load_cached_result(
                model_id=model_id,
                config=config,
                settings=settings,
                instance_ids=instance_ids,
            )
            if cached:
                submitted_instances = int(cached.get("submitted_instances", len(instances)))
                completed_instances = int(cached.get("completed_instances", submitted_instances))
                resolved_instances = int(cached.get("resolved_instances", 0))
                resolution_rate = (
                    float(cached["resolution_rate"])
                    if cached.get("resolution_rate") is not None
                    else None
                )
                self._mutate_job(
                    job_id,
                    lambda job: self._mark_model(
                        job,
                        model_id,
                        status="completed",
                        display_name=model.display_name,
                        selected_settings=settings,
                        report_path=Path(str(cached["report_path"])),
                        predictions_path=Path(str(cached["predictions_path"])),
                        evaluation_log_path=Path(str(cached["evaluation_log_path"])),
                        evaluation_command=str(cached.get("evaluation_command") or ""),
                        submitted_instances=submitted_instances,
                        completed_instances=completed_instances,
                        resolved_instances=resolved_instances,
                        resolution_rate=resolution_rate,
                        started=True,
                        finished=True,
                        event=(
                            f"Reused cached SWE-bench result for {model_id}: "
                            f"{resolved_instances}/{submitted_instances} resolved."
                        ),
                    ),
                )
                return
        self._mutate_job(
            job_id,
            lambda job: self._mark_model(
                job,
                model_id,
                status="launching",
                display_name=model.display_name,
                selected_settings=settings,
                started=True,
                current_model_id=model_id,
                event=f"Launching {model_id} with recommended settings.",
            ),
        )
        try:
            launch_result = self.manager.launch_model(
                model_id,
                preferred_port=self._setting_int(settings, "preferred_port"),
                max_model_len=self._setting_int(settings, "max_model_len"),
                tensor_parallel_size=self._setting_int(settings, "tensor_parallel_size"),
                kv_cache_dtype=self._setting_str(settings, "kv_cache_dtype"),
                gpu_memory_budget_gb=self._setting_float(settings, "gpu_memory_budget_gb"),
                keep_alive=False,
            )
            launch_args = list(getattr(launch_result, "command", []) or [])
            launch_command = (
                shlex.join(["vllm", "serve", *launch_args]) if launch_args else None
            )
            self._mutate_job(
                job_id,
                lambda job: self._mark_model(
                    job,
                    model_id,
                    status="waiting_for_vllm",
                    launch_command=launch_command,
                    host_port=launch_result.host_port,
                    container_name=launch_result.container_name,
                    event=f"Waiting for vLLM readiness on port {launch_result.host_port}.",
                ),
            )
            if launch_command:
                self._mutate_job(
                    job_id,
                    lambda job: self._append_event(
                        job,
                        f"Launch command for {model_id}: {launch_command}",
                    ),
                )
            served_model = self._wait_for_vllm(launch_result.host_port)
            model_root = self._job_root(job_id) / _slugify_model_id(model_id)
            predictions_path = self._write_predictions(
                model=model,
                config=config,
                instances=instances,
                host_port=launch_result.host_port,
                served_model=served_model,
                output_dir=model_root,
            )
            self._mutate_job(
                job_id,
                lambda job: self._mark_model(
                    job,
                    model_id,
                    status="evaluating",
                    predictions_path=predictions_path,
                    submitted_instances=len(instances),
                    event=f"Evaluating predictions for {model_id} with SWE-bench.",
                ),
            )
            report_path, report, evaluation_command, evaluation_log_path = self._run_evaluation(
                job_id=job_id,
                model_id=model_id,
                config=config,
                predictions_path=predictions_path,
                instance_ids=instance_ids,
                output_dir=model_root,
            )
            completed_instances = int(report.get("completed_instances", 0))
            resolved_instances = int(report.get("resolved_instances", 0))
            submitted_instances = int(report.get("submitted_instances", len(instances)))
            resolution_rate = None
            if submitted_instances > 0:
                resolution_rate = round((resolved_instances / submitted_instances) * 100, 2)
            self._save_cached_result(
                model_id=model_id,
                config=config,
                settings=settings,
                instance_ids=instance_ids,
                report_path=report_path,
                predictions_path=predictions_path,
                evaluation_log_path=evaluation_log_path,
                evaluation_command=shlex.join(evaluation_command),
                submitted_instances=submitted_instances,
                completed_instances=completed_instances,
                resolved_instances=resolved_instances,
                resolution_rate=resolution_rate,
            )
            self._mutate_job(
                job_id,
                lambda job: self._mark_model(
                    job,
                    model_id,
                    status="completed",
                    report_path=report_path,
                    evaluation_command=shlex.join(evaluation_command),
                    evaluation_log_path=evaluation_log_path,
                    submitted_instances=submitted_instances,
                    completed_instances=completed_instances,
                    resolved_instances=resolved_instances,
                    resolution_rate=resolution_rate,
                    finished=True,
                    event=(
                        f"Completed {model_id}: {resolved_instances}/{submitted_instances} "
                        "resolved."
                    ),
                ),
            )
        except Exception as exc:
            error_message = str(exc)
            self._mutate_job(
                job_id,
                lambda job: self._mark_model(
                    job,
                    model_id,
                    status="failed",
                    error=error_message,
                    finished=True,
                    event=f"Failed {model_id}: {error_message}",
                ),
            )
        finally:
            if launch_result is not None:
                self.manager.runtime.stop_model(container_name=launch_result.container_name)
            self._mutate_job(
                job_id,
                lambda job: setattr(job, "current_model_id", None),
            )

    def _mark_model(
        self,
        job: BenchmarkJob,
        model_id: str,
        *,
        status: str,
        event: str | None = None,
        started: bool = False,
        finished: bool = False,
        current_model_id: str | None = None,
        **updates: object,
    ) -> None:
        model_run = next(item for item in job.model_runs if item.model_id == model_id)
        model_run.status = status  # type: ignore[assignment]
        for key, value in updates.items():
            setattr(model_run, key, value)
        if started and model_run.started_at is None:
            model_run.started_at = _now()
        if finished:
            model_run.finished_at = _now()
        if current_model_id is not None:
            job.current_model_id = current_model_id
        if event:
            self._append_event(job, event)

    def _load_instances(self, config: BenchmarkConfig) -> list[dict[str, object]]:
        dataset = self._dataset_loader(config.dataset_name, split=config.split)
        if config.instance_limit is not None:
            limit = max(config.instance_limit, 0)
            if hasattr(dataset, "select"):
                dataset = dataset.select(range(min(limit, len(dataset))))
            else:
                dataset = list(dataset)[:limit]
        return [dict(instance) for instance in dataset]

    def _wait_for_vllm(self, host_port: int) -> str:
        deadline = time.monotonic() + VLLM_READY_TIMEOUT_SECONDS
        base_url = _runtime_base_url(host_port)
        while time.monotonic() < deadline:
            try:
                response = requests.get(f"{base_url}/v1/models", timeout=10)
                if response.ok:
                    payload = response.json()
                    data = payload.get("data") or []
                    if data:
                        return str(data[0].get("id") or "")
            except requests.RequestException:
                pass
            time.sleep(VLLM_READY_POLL_SECONDS)
        raise RuntimeError(f"Timed out waiting for vLLM to become ready on port {host_port}.")

    def _write_predictions(
        self,
        *,
        model: ModelMetadata,
        config: BenchmarkConfig,
        instances: list[dict[str, object]],
        host_port: int,
        served_model: str,
        output_dir: Path,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = output_dir / "predictions.jsonl"
        with predictions_path.open("w", encoding="utf-8") as handle:
            for instance in instances:
                raw_output = self._generate_patch(
                    host_port=host_port,
                    served_model=served_model,
                    prompt=_build_prompt(instance),
                    config=config,
                )
                print(
                    json.dumps(
                        {
                            "instance_id": instance["instance_id"],
                            "model_name_or_path": model.model_id,
                            "model_patch": _extract_patch(raw_output),
                        }
                    ),
                    file=handle,
                )
        return predictions_path

    def _generate_patch(
        self,
        *,
        host_port: int,
        served_model: str,
        prompt: str,
        config: BenchmarkConfig,
    ) -> str:
        base_url = f"{_runtime_base_url(host_port)}/v1"
        chat_payload = {
            "model": served_model,
            "messages": [
                {"role": "system", "content": SWE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": config.temperature,
            "max_tokens": config.max_response_tokens,
        }
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                json=chat_payload,
                timeout=600,
            )
            response.raise_for_status()
            payload = response.json()
            return str(payload["choices"][0]["message"]["content"])
        except requests.RequestException:
            completion_payload = {
                "model": served_model,
                "prompt": f"{SWE_SYSTEM_PROMPT}\n\n{prompt}",
                "temperature": config.temperature,
                "max_tokens": config.max_response_tokens,
            }
            response = requests.post(
                f"{base_url}/completions",
                json=completion_payload,
                timeout=600,
            )
            response.raise_for_status()
            payload = response.json()
            return str(payload["choices"][0]["text"])

    def _run_evaluation(
        self,
        *,
        job_id: str,
        model_id: str,
        config: BenchmarkConfig,
        predictions_path: Path,
        instance_ids: list[str],
        output_dir: Path,
    ) -> tuple[Path, dict[str, object], list[str], Path]:
        report_dir = output_dir / "evaluation_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        run_id = f"benchmark-{job_id[:8]}-{_slugify_model_id(model_id)}"
        command = [
            sys.executable,
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            config.dataset_name,
            "--split",
            config.split,
            "--predictions_path",
            str(predictions_path),
            "--max_workers",
            str(config.max_eval_workers),
            "--timeout",
            str(config.timeout_seconds),
            "--clean",
            "True",
            "--run_id",
            run_id,
            "--report_dir",
            str(report_dir),
        ]
        if instance_ids:
            command.extend(["--instance_ids", *instance_ids])
        if platform.machine().lower() in {"arm64", "aarch64"}:
            command.extend(["--namespace", "none"])
        self._mutate_job(
            job_id,
            lambda job: self._mark_model(
                job,
                model_id,
                status="evaluating",
                evaluation_command=shlex.join(command),
                evaluation_log_path=output_dir / "evaluation.log",
            ),
        )
        self._mutate_job(
            job_id,
            lambda job: self._append_event(
                job,
                f"SWE-bench command for {model_id}: {shlex.join(command)}",
            ),
        )
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        log_path = output_dir / "evaluation.log"
        log_path.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(
                result.stderr.strip() or result.stdout.strip() or "SWE-bench run failed."
            )
        report_path = report_dir / f"{model_id.replace('/', '__')}.{run_id}.json"
        if not report_path.exists():
            raise RuntimeError(f"SWE-bench report not found at {report_path}.")
        return report_path, json.loads(report_path.read_text(encoding="utf-8")), command, log_path

    @staticmethod
    def _setting_int(settings: dict[str, object], key: str) -> int | None:
        value = settings.get(key)
        return int(value) if value is not None else None

    @staticmethod
    def _setting_float(settings: dict[str, object], key: str) -> float | None:
        value = settings.get(key)
        return float(value) if value is not None else None

    @staticmethod
    def _setting_str(settings: dict[str, object], key: str) -> str | None:
        value = settings.get(key)
        return str(value) if value is not None else None