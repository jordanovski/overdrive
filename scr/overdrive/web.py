"""FastAPI-based web interface for Overdrive orchestration."""

from __future__ import annotations

from pathlib import Path
import shlex

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict

from overdrive import __version__
from overdrive.benchmarks import BenchmarkService
from overdrive.docker_runtime import INTERNAL_VLLM_PORT, VLLM_IMAGE
from overdrive.hardware import (
    GPUDevice,
    detect_gpus,
    recommended_gpu_budget_gb,
    recommended_kv_cache_dtype,
    recommended_max_model_len,
    recommended_tensor_parallel_size,
)
from overdrive.models import (
    BenchmarkConfig,
    LaunchConfig,
    ModelMetadata,
    ModelProfile,
    PreflightReport,
)
from overdrive.state import EngineStateManager

PACKAGE_ROOT = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(PACKAGE_ROOT / "templates"))


class LaunchSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preferred_port: int | None = None
    max_model_len: int | None = None
    tensor_parallel_size: int | None = None
    kv_cache_dtype: str | None = None
    gpu_memory_budget_gb: float | None = None


def _display_dtype(model: ModelMetadata) -> str:
    if model.dtype != "unknown":
        return model.dtype
    text_config = model.config_data.get("text_config", {})
    if isinstance(text_config, dict):
        nested_dtype = text_config.get("dtype")
        if isinstance(nested_dtype, str) and nested_dtype:
            return nested_dtype
    return "unknown"


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


def _recommended_launch_settings(
    manager: EngineStateManager,
    model: ModelMetadata,
    gpus: list[GPUDevice],
) -> LaunchSettings:
    return LaunchSettings(
        preferred_port=manager.runtime.reserve_port(model.profile.preferred_port),
        max_model_len=recommended_max_model_len(model, gpus),
        tensor_parallel_size=recommended_tensor_parallel_size(model, gpus),
        kv_cache_dtype=recommended_kv_cache_dtype(model, gpus),
        gpu_memory_budget_gb=recommended_gpu_budget_gb(model, gpus),
    )


def _hardware_summary(gpus: list[GPUDevice]) -> str:
    if not gpus:
        return "No GPU telemetry available"
    return "; ".join(
        f"{gpu.name}: free {gpu.free_memory_gb:g}/{gpu.total_memory_gb:g} GiB" for gpu in gpus
    )


def _serialize_model(
    manager: EngineStateManager,
    model: ModelMetadata,
    gpus: list[GPUDevice],
) -> dict[str, object]:
    recommendations = _recommended_settings(manager, model, gpus)
    preview = _preview_launch_command(manager, model, _recommended_launch_settings(manager, model, gpus))
    return {
        "model_id": model.model_id,
        "model_name": model.model_name,
        "display_name": model.display_name,
        "architecture": model.architecture,
        "model_type": model.model_type,
        "dtype": model.dtype,
        "dtype_display": _display_dtype(model),
        "parameter_size_billions": model.parameter_size_billions,
        "snapshot_path": str(model.snapshot_path),
        "hardware_summary": _hardware_summary(gpus),
        "profile": model.profile.model_dump(mode="json"),
        "recommendations": recommendations,
        "command_preview": preview,
    }


def _serialize_runtime_item(item: object) -> dict[str, object]:
    if isinstance(item, BaseModel):
        return item.model_dump(mode="json")
    raise TypeError(f"Unsupported runtime item: {type(item)!r}")


def _format_preflight_report(report: PreflightReport) -> str:
    parts = [
        f"allowed={report.allowed}",
        f"port={report.requested_port}",
        f"estimated_model_memory_gb={report.estimated_model_memory_gb}",
        f"active_reserved_memory_gb={report.active_reserved_memory_gb}",
        f"total_reserved_memory_gb={report.total_reserved_memory_gb}",
    ]
    if report.reason:
        parts.append(f"reason={report.reason}")
    return " | ".join(parts)


def _profile_from_settings(model: ModelMetadata, settings: LaunchSettings) -> ModelProfile:
    return ModelProfile(
        preferred_port=settings.preferred_port,
        max_model_len=settings.max_model_len,
        tensor_parallel_size=settings.tensor_parallel_size or 1,
        kv_cache_dtype=settings.kv_cache_dtype,
        gpu_memory_utilization=model.profile.gpu_memory_utilization,
        gpu_memory_budget_gb=settings.gpu_memory_budget_gb,
        extra_args=list(model.profile.extra_args),
    )


def _launch_config_from_settings(
    manager: EngineStateManager,
    model: ModelMetadata,
    settings: LaunchSettings,
) -> LaunchConfig:
    profile = model.profile
    return LaunchConfig(
        model_id=model.model_id,
        snapshot_path=model.snapshot_path,
        host_port=manager.runtime.reserve_port(settings.preferred_port or profile.preferred_port),
        max_model_len=settings.max_model_len if settings.max_model_len is not None else profile.max_model_len,
        tensor_parallel_size=settings.tensor_parallel_size or profile.tensor_parallel_size,
        kv_cache_dtype=settings.kv_cache_dtype if settings.kv_cache_dtype is not None else profile.kv_cache_dtype,
        gpu_memory_utilization=profile.gpu_memory_utilization,
        gpu_memory_budget_gb=(
            settings.gpu_memory_budget_gb
            if settings.gpu_memory_budget_gb is not None
            else profile.gpu_memory_budget_gb
        ),
        extra_args=list(profile.extra_args),
    )


def _preview_launch_command(
    manager: EngineStateManager,
    model: ModelMetadata,
    settings: LaunchSettings,
) -> dict[str, object]:
    launch = _launch_config_from_settings(manager, model, settings)
    args = manager.runtime.build_command(launch)
    return {
        "image": VLLM_IMAGE,
        "host_port": launch.host_port,
        "container_port": INTERNAL_VLLM_PORT,
        "args": args,
        "shell": shlex.join(["vllm", "serve", *args]),
    }


def create_app(
    manager: EngineStateManager,
    benchmark_service: BenchmarkService | None = None,
) -> FastAPI:
    app = FastAPI(title="Overdrive", version=__version__)
    app.state.manager = manager
    app.state.gpus = detect_gpus()
    app.state.benchmark_service = benchmark_service or BenchmarkService(
        manager,
        gpus=app.state.gpus,
    )
    app.mount("/static", StaticFiles(directory=str(PACKAGE_ROOT / "static")), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(
            request,
            "index.html",
            {"hub_root": str(manager.hub_root)},
        )

    @app.get("/benchmarks", response_class=HTMLResponse)
    async def benchmarks_page(request: Request) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(
            request,
            "benchmarks.html",
            {"hub_root": str(manager.hub_root)},
        )

    @app.get("/api/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/models")
    async def models() -> list[dict[str, object]]:
        gpus: list[GPUDevice] = app.state.gpus
        return [_serialize_model(manager, model, gpus) for model in manager.discover_models()]

    @app.get("/api/runtime")
    async def runtime() -> dict[str, list[dict[str, object]]]:
        containers = manager.active_containers()
        stats = manager.runtime.list_managed_stats()
        return {
            "containers": [_serialize_runtime_item(item) for item in containers],
            "stats": [_serialize_runtime_item(item) for item in stats],
        }

    @app.get("/api/logs/{model_id:path}")
    async def logs(model_id: str, tail: int = 50) -> dict[str, object]:
        container = next(
            (item for item in manager.active_containers() if item.model_id == model_id),
            None,
        )
        if container is None:
            return {"container_name": None, "lines": []}
        return {
            "container_name": container.name,
            "lines": manager.runtime.stream_logs(container.name, tail=tail),
        }

    @app.post("/api/models/{model_id:path}/plan")
    async def plan(model_id: str, settings: LaunchSettings) -> dict[str, object]:
        try:
            model = manager.get_model(model_id)
            report = manager.preflight_launch(model_id, **settings.model_dump())
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            **report.model_dump(mode="json"),
            "display": _format_preflight_report(report),
            "command_preview": _preview_launch_command(manager, model, settings),
        }

    @app.post("/api/models/{model_id:path}/launch")
    async def launch(model_id: str, settings: LaunchSettings) -> dict[str, object]:
        try:
            result = manager.launch_model(model_id, **settings.model_dump())
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return result.model_dump(mode="json")

    @app.post("/api/models/{model_id:path}/profile")
    async def save_profile(model_id: str, settings: LaunchSettings) -> dict[str, object]:
        try:
            model = manager.get_model(model_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        profile = _profile_from_settings(model, settings)
        path = manager.save_profile(model_id, profile)
        return {
            "saved": True,
            "model_id": model_id,
            "path": str(path),
            "profile": profile.model_dump(mode="json"),
        }

    @app.post("/api/models/{model_id:path}/stop")
    async def stop(model_id: str) -> dict[str, object]:
        return {"model_id": model_id, "stopped": manager.stop_model(model_id)}

    @app.post("/api/cleanup")
    async def cleanup() -> dict[str, int]:
        return {"stopped_count": manager.cleanup()}

    @app.get("/api/benchmarks/jobs")
    async def benchmark_jobs() -> list[dict[str, object]]:
        service: BenchmarkService = app.state.benchmark_service
        return [job.model_dump(mode="json") for job in service.list_jobs()]

    @app.get("/api/benchmarks/jobs/{job_id}")
    async def benchmark_job(job_id: str) -> dict[str, object]:
        service: BenchmarkService = app.state.benchmark_service
        try:
            job = service.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown benchmark job: {job_id}") from exc
        return job.model_dump(mode="json")

    @app.post("/api/benchmarks/jobs", status_code=202)
    async def start_benchmark_job(config: BenchmarkConfig) -> dict[str, object]:
        service: BenchmarkService = app.state.benchmark_service
        try:
            job = service.create_job(config)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return job.model_dump(mode="json")

    return app


def run_web(manager: EngineStateManager, *, host: str = "127.0.0.1", port: int = 8080) -> None:
    uvicorn.run(create_app(manager), host=host, port=port)