from pathlib import Path
from types import SimpleNamespace

import overdrive.web as web_module
from fastapi.testclient import TestClient
from overdrive.models import (
    BenchmarkConfig,
    BenchmarkJob,
    BenchmarkModelRun,
    ContainerRecord,
    ContainerStats,
    ModelMetadata,
    ModelProfile,
    PreflightReport,
)


def _model(*, size: float | None, dtype: str = "unknown", profile: ModelProfile | None = None):
    return ModelMetadata(
        model_id="qwen-35b-moe",
        model_name="qwen-35b-moe",
        architecture="Qwen",
        model_type="qwen",
        parameter_size_billions=size,
        dtype=dtype,
        snapshot_path=Path("/models/qwen-35b-moe"),
        config_path=Path("/models/qwen-35b-moe/config.json"),
        config_data={"text_config": {"dtype": "bfloat16"}},
        profile=profile or ModelProfile(),
    )


def _manager(model: ModelMetadata):
    saved: dict[str, object] = {}
    report = PreflightReport(
        model_id=model.model_id,
        requested_port=8000,
        estimated_model_memory_gb=42.0,
        active_reserved_memory_gb=10.0,
        total_reserved_memory_gb=52.0,
        gpu_memory_budget_gb=110.0,
        allowed=True,
    )
    runtime = SimpleNamespace(
        reserve_port=lambda preferred: preferred or 8000,
        build_command=lambda launch: [
            "--model",
            "/models/current",
            "--port",
            "8000",
            "--tensor-parallel-size",
            str(launch.tensor_parallel_size),
        ],
        list_managed_stats=lambda: [
            ContainerStats(
                container_id="abc123",
                name="overdrive-qwen",
                model_id=model.model_id,
                cpu_percent=25.0,
                memory_usage_gb=8.0,
                memory_limit_gb=16.0,
                memory_percent=50.0,
                network_rx_mb=12.0,
                network_tx_mb=4.0,
            )
        ],
        stream_logs=lambda container_name, tail=50: ["first line", "second line"],
    )

    def save_profile(model_id: str, profile: ModelProfile):
        saved["model_id"] = model_id
        saved["profile"] = profile
        return Path("/tmp/profiles.yaml")

    manager = SimpleNamespace(
        hub_root=Path("/models"),
        runtime=runtime,
        discover_models=lambda: [model],
        active_containers=lambda: [
            ContainerRecord(
                container_id="abc123",
                name="overdrive-qwen",
                status="running",
                host_port=8000,
                model_id=model.model_id,
                image="nvcr.io/nvidia/vllm:26.04-py3",
                memory_reservation_gb=42.0,
            )
        ],
        preflight_launch=lambda model_id, **kwargs: report,
        launch_model=lambda model_id, **kwargs: SimpleNamespace(
            model_dump=lambda mode="json": {
                "model_id": model_id,
                "container_name": "overdrive-qwen-8000",
                "host_port": 8000,
                "status": "running",
                "image": "nvcr.io/nvidia/vllm:26.04-py3",
                "command": ["--model", "/models/current"],
            }
        ),
        get_model=lambda model_id: model,
        save_profile=save_profile,
        stop_model=lambda model_id: True,
        cleanup=lambda: 1,
    )
    return manager, saved


def test_display_dtype_falls_back_to_text_config() -> None:
    model = _model(size=35.0)

    assert web_module._display_dtype(model) == "bfloat16"


def test_format_preflight_report_includes_reason() -> None:
    report = PreflightReport(
        model_id="qwen-35b-moe",
        requested_port=8000,
        estimated_model_memory_gb=42.0,
        active_reserved_memory_gb=10.0,
        total_reserved_memory_gb=52.0,
        gpu_memory_budget_gb=48.0,
        allowed=False,
        reason="Projected reserved memory 52.0 GiB exceeds budget 48.0 GiB.",
    )

    formatted = web_module._format_preflight_report(report)

    assert "allowed=False" in formatted
    assert "port=8000" in formatted
    assert "reason=Projected reserved memory 52.0 GiB exceeds budget 48.0 GiB." in formatted


def test_models_endpoint_includes_recommendations(monkeypatch) -> None:
    model = _model(size=35.0)
    manager, _ = _manager(model)

    monkeypatch.setattr(web_module, "detect_gpus", lambda: [])
    monkeypatch.setattr(web_module, "recommended_max_model_len", lambda model, gpus: 4096)
    monkeypatch.setattr(web_module, "recommended_tensor_parallel_size", lambda model, gpus: 1)
    monkeypatch.setattr(web_module, "recommended_kv_cache_dtype", lambda model, gpus: "auto")
    monkeypatch.setattr(web_module, "recommended_gpu_budget_gb", lambda model, gpus: 110.0)

    client = TestClient(web_module.create_app(manager))

    response = client.get("/api/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload[0]["model_id"] == model.model_id
    assert payload[0]["recommendations"]["preferred_port"] == 8000
    assert payload[0]["recommendations"]["kv_cache_dtype"] == "auto"
    assert payload[0]["command_preview"]["image"] == "nvcr.io/nvidia/vllm:26.04-py3"
    assert payload[0]["command_preview"]["host_port"] == 8000
    assert payload[0]["command_preview"]["model_source_path"] == str(model.snapshot_path)
    assert payload[0]["command_preview"]["model_container_path"] == "/models/current"
    assert payload[0]["command_preview"]["shell"].startswith("vllm serve --model")
    assert payload[0]["command_preview"]["docker_shell"] is None


def test_models_diagnostics_endpoint_reports_discovery(monkeypatch) -> None:
    model = _model(size=35.0)
    manager, _ = _manager(model)

    monkeypatch.setattr(web_module, "detect_gpus", lambda: [])

    client = TestClient(web_module.create_app(manager))

    response = client.get("/api/models/diagnostics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["discovered_count"] == 1
    assert payload["discovered_model_ids"] == [model.model_id]
    assert payload["hub_root"] == str(manager.hub_root)


def test_plan_endpoint_returns_display_report(monkeypatch) -> None:
    model = _model(size=35.0)
    manager, _ = _manager(model)

    monkeypatch.setattr(web_module, "detect_gpus", lambda: [])

    client = TestClient(web_module.create_app(manager))
    response = client.post(
        f"/api/models/{model.model_id}/plan",
        json={
            "preferred_port": 8000,
            "max_model_len": 4096,
            "tensor_parallel_size": 1,
            "kv_cache_dtype": "auto",
            "gpu_memory_budget_gb": 110.0,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["requested_port"] == 8000
    assert "allowed=True" in payload["display"]
    assert payload["command_preview"]["host_port"] == 8000
    assert payload["command_preview"]["model_mount_source"] == str(model.snapshot_path)
    assert payload["command_preview"]["model_mount_target"] == "/models/current"
    assert payload["command_preview"]["args"][0:2] == ["--model", "/models/current"]


def test_profile_endpoint_persists_visible_settings(monkeypatch) -> None:
    model = _model(
        size=35.0,
        profile=ModelProfile(
            gpu_memory_utilization=0.92,
            extra_args=["--enable-prefix-caching"],
        ),
    )
    manager, saved = _manager(model)

    monkeypatch.setattr(web_module, "detect_gpus", lambda: [])

    client = TestClient(web_module.create_app(manager))
    response = client.post(
        f"/api/models/{model.model_id}/profile",
        json={
            "preferred_port": 8001,
            "max_model_len": 4096,
            "tensor_parallel_size": 1,
            "kv_cache_dtype": "auto",
            "gpu_memory_budget_gb": 110.0,
        },
    )

    assert response.status_code == 200
    assert saved["model_id"] == model.model_id
    saved_profile = saved["profile"]
    assert isinstance(saved_profile, ModelProfile)
    assert saved_profile.preferred_port == 8001
    assert saved_profile.gpu_memory_utilization == 0.92
    assert saved_profile.extra_args == ["--enable-prefix-caching"]


def test_runtime_and_logs_endpoints_expose_container_data(monkeypatch) -> None:
    model = _model(size=35.0)
    manager, _ = _manager(model)

    monkeypatch.setattr(web_module, "detect_gpus", lambda: [])

    client = TestClient(web_module.create_app(manager))

    runtime_response = client.get("/api/runtime")
    logs_response = client.get(f"/api/logs/{model.model_id}")

    assert runtime_response.status_code == 200
    assert runtime_response.json()["containers"][0]["model_id"] == model.model_id
    assert logs_response.status_code == 200
    assert logs_response.json()["lines"] == ["first line", "second line"]


def test_model_routes_support_slash_separated_model_ids(monkeypatch) -> None:
    model = ModelMetadata(
        model_id="meta-llama/meta-llama/Meta-Llama-3.1-8B-Instruct",
        model_name="Meta-Llama-3.1-8B-Instruct",
        architecture="LlamaForCausalLM",
        model_type="llama",
        parameter_size_billions=8.0,
        dtype="bfloat16",
        snapshot_path=Path("/models/meta-llama/meta-llama/Meta-Llama-3.1-8B-Instruct"),
        config_path=Path(
            "/models/meta-llama/meta-llama/Meta-Llama-3.1-8B-Instruct/config.json"
        ),
        config_data={"architectures": ["LlamaForCausalLM"], "torch_dtype": "bfloat16"},
        profile=ModelProfile(),
    )
    manager, saved = _manager(model)

    monkeypatch.setattr(web_module, "detect_gpus", lambda: [])

    client = TestClient(web_module.create_app(manager))

    logs_response = client.get("/api/logs/meta-llama/meta-llama/Meta-Llama-3.1-8B-Instruct")
    profile_response = client.post(
        "/api/models/meta-llama/meta-llama/Meta-Llama-3.1-8B-Instruct/profile",
        json={
            "preferred_port": 8001,
            "max_model_len": 4096,
            "tensor_parallel_size": 1,
            "kv_cache_dtype": "auto",
            "gpu_memory_budget_gb": 110.0,
        },
    )

    assert logs_response.status_code == 200
    assert logs_response.json()["lines"] == ["first line", "second line"]
    assert profile_response.status_code == 200
    assert saved["model_id"] == model.model_id


def test_benchmark_routes_expose_jobs(monkeypatch) -> None:
    model = _model(size=35.0)
    manager, _ = _manager(model)
    job = BenchmarkJob(
        job_id="job-1",
        config=BenchmarkConfig(model_ids=[model.model_id]),
        model_runs=[
            BenchmarkModelRun(
                model_id=model.model_id,
                display_name=model.display_name,
                status="completed",
                submitted_instances=4,
                resolved_instances=2,
                resolution_rate=50.0,
            )
        ],
        status="completed",
    )
    captured: dict[str, object] = {}

    def create_job(config: BenchmarkConfig) -> BenchmarkJob:
        captured["config"] = config
        return job

    benchmark_service = SimpleNamespace(
        list_jobs=lambda: [job],
        get_job=lambda job_id: job,
        get_model_run_log=lambda job_id, model_id: {
            "job_id": job_id,
            "model_id": model_id,
            "display_name": model.display_name,
            "status": "completed",
            "evaluation_log_path": "/tmp/evaluation.log",
            "content": "full log",
        },
        create_job=create_job,
    )

    monkeypatch.setattr(web_module, "detect_gpus", lambda: [])

    client = TestClient(web_module.create_app(manager, benchmark_service=benchmark_service))

    page_response = client.get("/benchmarks")
    jobs_response = client.get("/api/benchmarks/jobs")
    log_response = client.get(f"/api/benchmarks/jobs/{job.job_id}/logs/{model.model_id}")
    create_response = client.post(
        "/api/benchmarks/jobs",
        json={
            "model_ids": [model.model_id],
            "dataset_name": "princeton-nlp/SWE-bench_Lite",
            "split": "test",
            "instance_limit": 10,
            "max_eval_workers": 2,
        },
    )

    assert page_response.status_code == 200
    assert "SWE-bench" in page_response.text
    assert web_module.__version__ in page_response.text
    assert jobs_response.status_code == 200
    assert jobs_response.json()[0]["job_id"] == "job-1"
    assert log_response.status_code == 200
    assert log_response.json()["content"] == "full log"
    assert create_response.status_code == 202
    assert captured["config"].model_ids == [model.model_id]


def test_index_page_renders_app_version(monkeypatch) -> None:
    model = _model(size=35.0)
    manager, _ = _manager(model)

    monkeypatch.setattr(web_module, "detect_gpus", lambda: [])

    client = TestClient(web_module.create_app(manager))
    response = client.get("/")

    assert response.status_code == 200
    assert web_module.__version__ in response.text


def test_model_search_page_renders(monkeypatch) -> None:
    model = _model(size=35.0)
    manager, _ = _manager(model)

    monkeypatch.setattr(web_module, "detect_gpus", lambda: [])

    client = TestClient(web_module.create_app(manager))
    response = client.get("/models-search")

    assert response.status_code == 200
    assert "Model Search" in response.text
    assert web_module.__version__ in response.text


def test_hub_search_route_returns_results(monkeypatch) -> None:
    model = _model(size=35.0)
    manager, _ = _manager(model)

    monkeypatch.setattr(web_module, "detect_gpus", lambda: [])

    captured: dict[str, object] = {}

    def fake_search(options):
        captured["options"] = options
        return [
            {
                "id": "Qwen/Qwen3-8B",
                "downloads": 123,
                "likes": 45,
                "pipeline_tag": "text-generation",
                "library_name": "transformers",
                "tags": ["vllm", "nvfp4"],
                "dgx_tags": ["vllm", "nvfp4"],
            }
        ]

    monkeypatch.setattr(web_module, "search_hub_models", fake_search)

    client = TestClient(web_module.create_app(manager))
    response = client.post(
        "/api/hub/search",
        json={
            "query": "qwen",
            "quantization": "nvfp4",
            "limit": 20,
            "dgx_ready_only": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["hub_root"] == str(manager.hub_root)
    assert payload["models"][0]["id"] == "Qwen/Qwen3-8B"
    assert captured["options"].query == "qwen"
    assert captured["options"].quantization == "nvfp4"
    assert captured["options"].dgx_ready_only is True


def test_hub_download_route_targets_hub_root(monkeypatch) -> None:
    model = _model(size=35.0)
    manager, _ = _manager(model)
    manager.hub_root = Path("/tmp/models")

    monkeypatch.setattr(web_module, "detect_gpus", lambda: [])

    captured: dict[str, object] = {}

    def fake_download(
        model_id,
        *,
        local_dir,
        cache_dir=None,
        revision=None,
        includes=None,
        excludes=None,
        token=None,
        max_workers=None,
        force_download=False,
        dry_run=False,
    ):
        captured["model_id"] = model_id
        captured["local_dir"] = local_dir
        captured["exists"] = local_dir.exists()
        captured["includes"] = includes
        captured["excludes"] = excludes
        return {
            "model_id": model_id,
            "download_path": str(local_dir),
            "stdout": "ok",
        }

    monkeypatch.setattr(web_module, "download_model", fake_download)

    client = TestClient(web_module.create_app(manager))
    response = client.post(
        "/api/hub/download",
        json={
            "model_id": "Qwen/Qwen3-8B",
            "include": ["*.safetensors"],
            "exclude": ["*.bin"],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    expected_dir = manager.hub_root / "Qwen" / "Qwen3-8B"
    assert payload["local_dir"] == str(expected_dir)
    assert captured["model_id"] == "Qwen/Qwen3-8B"
    assert str(captured["local_dir"]) == str(expected_dir)
    assert captured["exists"] is True
    assert captured["includes"] == ["*.safetensors"]
    assert captured["excludes"] == ["*.bin"]


def test_hub_download_route_reports_unwritable_hub_root(monkeypatch) -> None:
    model = _model(size=35.0)
    manager, _ = _manager(model)
    manager.hub_root = Path("/read-only/models")

    monkeypatch.setattr(web_module, "detect_gpus", lambda: [])

    original_mkdir = Path.mkdir

    def fake_mkdir(self, parents=False, exist_ok=False):
        if self == manager.hub_root / "Qwen" / "Qwen3-8B":
            raise OSError("read-only file system")
        return original_mkdir(self, parents=parents, exist_ok=exist_ok)

    monkeypatch.setattr(Path, "mkdir", fake_mkdir)

    client = TestClient(web_module.create_app(manager))
    response = client.post(
        "/api/hub/download",
        json={
            "model_id": "Qwen/Qwen3-8B",
        },
    )

    assert response.status_code == 400
    assert "mounted writable" in response.json()["detail"]