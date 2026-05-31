from pathlib import Path
from types import SimpleNamespace

from overdrive.benchmarks import (
    BenchmarkService,
    _build_prompt,
    _extract_patch,
    _runtime_base_url,
)
from overdrive.models import BenchmarkConfig, ModelMetadata, ModelProfile


def _model(model_id: str, *, size: float = 35.0) -> ModelMetadata:
    return ModelMetadata(
        model_id=model_id,
        model_name=model_id.split("/")[-1],
        architecture="Qwen",
        model_type="qwen",
        parameter_size_billions=size,
        dtype="bfloat16",
        snapshot_path=Path(f"/models/{model_id.replace('/', '__') }"),
        config_path=Path(f"/models/{model_id.replace('/', '__')}/config.json"),
        config_data={},
        profile=ModelProfile(),
    )


def test_extract_patch_prefers_fenced_diff() -> None:
    output = "before\n```diff\ndiff --git a/foo.py b/foo.py\n--- a/foo.py\n+++ b/foo.py\n```\nafter"

    patch = _extract_patch(output)

    assert patch.startswith("diff --git a/foo.py b/foo.py")


def test_build_prompt_includes_issue_and_tests() -> None:
    prompt = _build_prompt(
        {
            "instance_id": "sympy__sympy-1",
            "repo": "sympy/sympy",
            "problem_statement": "Fix symbolic parsing.",
            "hints_text": "Look at sympify.",
            "FAIL_TO_PASS": ["test_fix"],
            "PASS_TO_PASS": ["test_keep"],
        }
    )

    assert "Fix symbolic parsing." in prompt
    assert "Look at sympify." in prompt
    assert "test_fix" in prompt
    assert "test_keep" in prompt


def test_runtime_base_url_uses_configured_runtime_host(monkeypatch) -> None:
    monkeypatch.setenv("OVERDRIVE_RUNTIME_HOST", "host.docker.internal")

    assert _runtime_base_url(8000) == "http://host.docker.internal:8000"


def test_benchmark_service_runs_selected_models_sequentially(monkeypatch, tmp_path: Path) -> None:
    model_a = _model("org/model-a")
    model_b = _model("org/model-b")
    launches: list[str] = []
    stops: list[str] = []

    manager = SimpleNamespace(
        get_model=lambda model_id: {model_a.model_id: model_a, model_b.model_id: model_b}[model_id],
        launch_model=lambda model_id, **kwargs: launches.append(model_id)
        or SimpleNamespace(
            container_name=f"overdrive-{model_id.replace('/', '-')}",
            host_port=8000 + len(launches),
        ),
        runtime=SimpleNamespace(
            stop_model=lambda container_name=None, model_id=None: stops.append(
                container_name or model_id
            )
        ),
    )
    service = BenchmarkService(
        manager,
        gpus=[],
        dataset_loader=lambda dataset_name, split: [
            {
                "instance_id": "sympy__sympy-1",
                "repo": "sympy/sympy",
                "problem_statement": "Fix parsing.",
            }
        ],
        background_runner=lambda func: func(),
    )
    monkeypatch.setattr(
        "overdrive.benchmarks._recommended_settings",
        lambda manager, model, gpus: {
            "preferred_port": 8000,
            "max_model_len": 4096,
            "tensor_parallel_size": 1,
            "kv_cache_dtype": "auto",
            "gpu_memory_budget_gb": 110.0,
        },
    )
    monkeypatch.setattr(service, "_wait_for_vllm", lambda host_port: "served-model")
    monkeypatch.setattr(
        service,
        "_write_predictions",
        lambda **kwargs: tmp_path / f"{kwargs['model'].model_name}.jsonl",
    )
    monkeypatch.setattr(
        service,
        "_run_evaluation",
        lambda **kwargs: (
            tmp_path / f"{kwargs['model_id'].replace('/', '__')}.json",
            {
                "submitted_instances": 1,
                "completed_instances": 1,
                "resolved_instances": 1,
            },
            ["python", "-m", "swebench.harness.run_evaluation"],
            tmp_path / "evaluation.log",
        ),
    )

    job = service.create_job(BenchmarkConfig(model_ids=[model_a.model_id, model_b.model_id]))
    completed_job = service.get_job(job.job_id)

    assert completed_job.status == "completed"
    assert launches == [model_a.model_id, model_b.model_id]
    assert stops == ["overdrive-org-model-a", "overdrive-org-model-b"]
    assert [item.status for item in completed_job.model_runs] == ["completed", "completed"]
    assert all(item.resolution_rate == 100.0 for item in completed_job.model_runs)


def test_benchmark_service_continues_after_model_failure(monkeypatch, tmp_path: Path) -> None:
    model_a = _model("org/model-a")
    model_b = _model("org/model-b")
    launches: list[str] = []

    def launch_model(model_id: str, **kwargs):
        launches.append(model_id)
        if model_id == model_a.model_id:
            raise RuntimeError("launch failed")
        return SimpleNamespace(container_name="overdrive-org-model-b", host_port=8001)

    manager = SimpleNamespace(
        get_model=lambda model_id: {model_a.model_id: model_a, model_b.model_id: model_b}[model_id],
        launch_model=launch_model,
        runtime=SimpleNamespace(stop_model=lambda container_name=None, model_id=None: True),
    )
    service = BenchmarkService(
        manager,
        gpus=[],
        dataset_loader=lambda dataset_name, split: [
            {"instance_id": "id-1", "problem_statement": "Fix it."}
        ],
        background_runner=lambda func: func(),
    )
    monkeypatch.setattr(
        "overdrive.benchmarks._recommended_settings",
        lambda manager, model, gpus: {
            "preferred_port": 8000,
            "max_model_len": 4096,
            "tensor_parallel_size": 1,
            "kv_cache_dtype": "auto",
            "gpu_memory_budget_gb": 110.0,
        },
    )
    monkeypatch.setattr(service, "_wait_for_vllm", lambda host_port: "served-model")
    monkeypatch.setattr(
        service,
        "_write_predictions",
        lambda **kwargs: tmp_path / "predictions.jsonl",
    )
    monkeypatch.setattr(
        service,
        "_run_evaluation",
        lambda **kwargs: (
            tmp_path / "report.json",
            {
                "submitted_instances": 1,
                "completed_instances": 1,
                "resolved_instances": 0,
            },
            ["python", "-m", "swebench.harness.run_evaluation"],
            tmp_path / "evaluation.log",
        ),
    )

    job = service.create_job(BenchmarkConfig(model_ids=[model_a.model_id, model_b.model_id]))
    completed_job = service.get_job(job.job_id)

    assert completed_job.status == "completed"
    assert launches == [model_a.model_id, model_b.model_id]
    assert completed_job.model_runs[0].status == "failed"
    assert completed_job.model_runs[1].status == "completed"


def test_generate_patch_uses_runtime_host_for_containerized_overdrive(monkeypatch) -> None:
    manager = SimpleNamespace()
    service = BenchmarkService(
        manager,
        gpus=[],
        dataset_loader=lambda dataset_name, split: [],
        background_runner=lambda func: func(),
    )
    captured: dict[str, str] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"choices": [{"message": {"content": "diff --git a/x b/x"}}]}

    def fake_post(url: str, json: dict[str, object], timeout: int):
        captured["url"] = url
        return FakeResponse()

    monkeypatch.setenv("OVERDRIVE_RUNTIME_HOST", "host.docker.internal")
    monkeypatch.setattr("overdrive.benchmarks.requests.post", fake_post)

    output = service._generate_patch(
        host_port=8000,
        served_model="served-model",
        prompt="Fix it",
        config=BenchmarkConfig(model_ids=["org/model-a"]),
    )

    assert output == "diff --git a/x b/x"
    assert captured["url"] == "http://host.docker.internal:8000/v1/chat/completions"


def test_benchmark_service_reuses_cached_results(monkeypatch, tmp_path: Path) -> None:
    model = _model("org/model-a")
    launches: list[str] = []

    manager = SimpleNamespace(
        get_model=lambda model_id: model,
        launch_model=lambda model_id, **kwargs: launches.append(model_id)
        or SimpleNamespace(
            container_name=f"overdrive-{model_id.replace('/', '-')}",
            host_port=8001,
            command=["--model", "/models/current", "--port", "8000"],
        ),
        runtime=SimpleNamespace(stop_model=lambda container_name=None, model_id=None: True),
    )
    service = BenchmarkService(
        manager,
        gpus=[],
        dataset_loader=lambda dataset_name, split: [
            {
                "instance_id": "sympy__sympy-1",
                "repo": "sympy/sympy",
                "problem_statement": "Fix parsing.",
            }
        ],
        background_runner=lambda func: func(),
    )

    monkeypatch.setattr("overdrive.benchmarks.benchmarks_root", lambda: tmp_path)
    monkeypatch.setattr(
        "overdrive.benchmarks._recommended_settings",
        lambda manager, model, gpus: {
            "preferred_port": 8000,
            "max_model_len": 4096,
            "tensor_parallel_size": 1,
            "kv_cache_dtype": "auto",
            "gpu_memory_budget_gb": 110.0,
        },
    )
    monkeypatch.setattr(service, "_wait_for_vllm", lambda host_port: "served-model")
    monkeypatch.setattr(
        service,
        "_write_predictions",
        lambda **kwargs: tmp_path / "predictions.jsonl",
    )
    monkeypatch.setattr(
        service,
        "_run_evaluation",
        lambda **kwargs: (
            tmp_path / "report.json",
            {
                "submitted_instances": 1,
                "completed_instances": 1,
                "resolved_instances": 1,
            },
            ["python", "-m", "swebench.harness.run_evaluation"],
            tmp_path / "evaluation.log",
        ),
    )
    (tmp_path / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "report.json").write_text("{}", encoding="utf-8")
    (tmp_path / "evaluation.log").write_text("ok", encoding="utf-8")

    first = service.create_job(BenchmarkConfig(model_ids=[model.model_id]))
    second = service.create_job(BenchmarkConfig(model_ids=[model.model_id]))

    first_job = service.get_job(first.job_id)
    second_job = service.get_job(second.job_id)

    assert first_job.status == "completed"
    assert second_job.status == "completed"
    assert launches == [model.model_id]
    assert second_job.model_runs[0].status == "completed"
    assert "Reused cached SWE-bench result" in "\n".join(second_job.events)


def test_benchmark_service_loads_persisted_jobs_with_log_excerpt(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("overdrive.benchmarks.benchmarks_root", lambda: tmp_path)

    log_path = tmp_path / "job-1" / "org-model-a" / "evaluation.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("line-1\nline-2\nline-3\n", encoding="utf-8")

    persisted_job = {
        "job_id": "job-1",
        "status": "failed",
        "config": {
            "model_ids": ["org/model-a"],
            "dataset_name": "princeton-nlp/SWE-bench_Lite",
            "split": "test",
            "instance_limit": 10,
            "max_eval_workers": 2,
            "timeout_seconds": 1800,
            "temperature": 0.0,
            "max_response_tokens": 1200,
            "reuse_cached_results": True,
        },
        "model_runs": [
            {
                "model_id": "org/model-a",
                "display_name": "model-a",
                "status": "failed",
                "evaluation_log_path": str(log_path),
                "error": "evaluation failed",
            }
        ],
        "events": ["Failed org/model-a: evaluation failed"],
    }
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    (jobs_dir / "job-1.json").write_text(__import__("json").dumps(persisted_job), encoding="utf-8")

    service = BenchmarkService(
        SimpleNamespace(),
        gpus=[],
        dataset_loader=lambda dataset_name, split: [],
        background_runner=lambda func: func(),
    )

    jobs = service.list_jobs()

    assert len(jobs) == 1
    assert jobs[0].job_id == "job-1"
    assert jobs[0].model_runs[0].evaluation_log_excerpt == "line-1\nline-2\nline-3"


def test_benchmark_service_returns_full_model_run_log(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("overdrive.benchmarks.benchmarks_root", lambda: tmp_path)

    log_path = tmp_path / "job-1" / "org-model-a" / "evaluation.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    persisted_job = {
        "job_id": "job-1",
        "status": "failed",
        "config": {
            "model_ids": ["org/model-a"],
            "dataset_name": "princeton-nlp/SWE-bench_Lite",
            "split": "test",
            "instance_limit": 10,
            "max_eval_workers": 2,
            "timeout_seconds": 1800,
            "temperature": 0.0,
            "max_response_tokens": 1200,
            "reuse_cached_results": True,
        },
        "model_runs": [
            {
                "model_id": "org/model-a",
                "display_name": "model-a",
                "status": "failed",
                "evaluation_log_path": str(log_path),
                "error": "evaluation failed",
            }
        ],
        "events": ["Failed org/model-a: evaluation failed"],
    }
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    (jobs_dir / "job-1.json").write_text(__import__("json").dumps(persisted_job), encoding="utf-8")

    service = BenchmarkService(
        SimpleNamespace(),
        gpus=[],
        dataset_loader=lambda dataset_name, split: [],
        background_runner=lambda func: func(),
    )

    payload = service.get_model_run_log("job-1", "org/model-a")

    assert payload["job_id"] == "job-1"
    assert payload["model_id"] == "org/model-a"
    assert payload["content"] == "alpha\nbeta\ngamma\n"