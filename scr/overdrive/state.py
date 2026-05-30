"""Application state manager for orchestration and discovery."""

from __future__ import annotations

from pathlib import Path

from overdrive.docker_runtime import DockerRuntime
from overdrive.models import (
    LaunchConfig,
    LaunchResult,
    ModelMetadata,
    ModelProfile,
    PreflightReport,
)
from overdrive.paths import default_hub_root
from overdrive.profiles import load_profiles, upsert_profile
from overdrive.scanner import scan_model_cache


class EngineStateManager:
    def __init__(
        self,
        hub_root: Path | None = None,
        runtime: DockerRuntime | None = None,
        profiles_path: Path | None = None,
    ) -> None:
        self.hub_root = hub_root or default_hub_root()
        self.runtime = runtime or DockerRuntime()
        self.profiles_path = profiles_path

    def discover_models(self) -> list[ModelMetadata]:
        return scan_model_cache(self.hub_root, self.profiles_path)

    def active_containers(self):
        return self.runtime.list_managed_containers()

    def get_model(self, model_id: str) -> ModelMetadata:
        for model in self.discover_models():
            if model.model_id == model_id or model.model_name == model_id:
                return model
        raise KeyError(f"Model '{model_id}' not found in {self.hub_root}.")

    def save_profile(self, model_id: str, profile: ModelProfile) -> Path:
        return upsert_profile(model_id, profile, self.profiles_path)

    def current_profile(self, model_id: str) -> ModelProfile:
        return load_profiles(self.profiles_path).models.get(model_id, ModelProfile())

    def launch_model(
        self,
        model_id: str,
        *,
        preferred_port: int | None = None,
        max_model_len: int | None = None,
        tensor_parallel_size: int | None = None,
        kv_cache_dtype: str | None = None,
        gpu_memory_utilization: float | None = None,
        gpu_memory_budget_gb: float | None = None,
        extra_args: list[str] | None = None,
        keep_alive: bool = False,
        dry_run: bool = False,
    ) -> LaunchResult:
        model = self.get_model(model_id)
        profile = model.profile
        host_port = self.runtime.reserve_port(preferred_port or profile.preferred_port)
        launch = {
            "model_id": model.model_id,
            "snapshot_path": model.snapshot_path,
            "host_port": host_port,
            "max_model_len": max_model_len if max_model_len is not None else profile.max_model_len,
            "tensor_parallel_size": tensor_parallel_size or profile.tensor_parallel_size,
            "kv_cache_dtype": (
                kv_cache_dtype if kv_cache_dtype is not None else profile.kv_cache_dtype
            ),
            "gpu_memory_utilization": (
                gpu_memory_utilization
                if gpu_memory_utilization is not None
                else profile.gpu_memory_utilization
            ),
            "gpu_memory_budget_gb": (
                gpu_memory_budget_gb
                if gpu_memory_budget_gb is not None
                else profile.gpu_memory_budget_gb
            ),
            "extra_args": extra_args if extra_args is not None else list(profile.extra_args),
            "keep_alive": keep_alive,
            "dry_run": dry_run,
        }
        return self.runtime.launch_model(model, LaunchConfig(**launch))

    def preflight_launch(
        self,
        model_id: str,
        *,
        preferred_port: int | None = None,
        max_model_len: int | None = None,
        tensor_parallel_size: int | None = None,
        kv_cache_dtype: str | None = None,
        gpu_memory_utilization: float | None = None,
        gpu_memory_budget_gb: float | None = None,
        extra_args: list[str] | None = None,
    ) -> PreflightReport:
        model = self.get_model(model_id)
        profile = model.profile
        host_port = self.runtime.reserve_port(preferred_port or profile.preferred_port)
        launch = LaunchConfig(
            model_id=model.model_id,
            snapshot_path=model.snapshot_path,
            host_port=host_port,
            max_model_len=max_model_len if max_model_len is not None else profile.max_model_len,
            tensor_parallel_size=tensor_parallel_size or profile.tensor_parallel_size,
            kv_cache_dtype=kv_cache_dtype if kv_cache_dtype is not None else profile.kv_cache_dtype,
            gpu_memory_utilization=(
                gpu_memory_utilization
                if gpu_memory_utilization is not None
                else profile.gpu_memory_utilization
            ),
            gpu_memory_budget_gb=(
                gpu_memory_budget_gb
                if gpu_memory_budget_gb is not None
                else profile.gpu_memory_budget_gb
            ),
            extra_args=extra_args if extra_args is not None else list(profile.extra_args),
        )
        return self.runtime.preflight_launch(model, launch)

    def stop_model(self, model_id: str) -> bool:
        return self.runtime.stop_model(model_id=model_id)

    def cleanup(self) -> int:
        return self.runtime.cleanup()