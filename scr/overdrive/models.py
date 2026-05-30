"""Core domain models for Overdrive."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preferred_port: int | None = None
    max_model_len: int | None = None
    tensor_parallel_size: int = 1
    kv_cache_dtype: str | None = None
    gpu_memory_utilization: float | None = None
    gpu_memory_budget_gb: float | None = None
    extra_args: list[str] = Field(default_factory=list)


class ModelMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    model_name: str
    architecture: str = "unknown"
    model_type: str = "unknown"
    parameter_size_billions: float | None = None
    dtype: str = "unknown"
    snapshot_path: Path
    config_path: Path
    config_data: dict[str, Any] = Field(default_factory=dict)
    profile: ModelProfile = Field(default_factory=ModelProfile)

    @property
    def display_name(self) -> str:
        size = (
            f" {self.parameter_size_billions:g}B"
            if self.parameter_size_billions is not None
            else ""
        )
        return f"{self.model_name}{size} [{self.dtype}]"


class LaunchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    snapshot_path: Path
    host_port: int
    max_model_len: int | None = None
    tensor_parallel_size: int = 1
    kv_cache_dtype: str | None = None
    gpu_memory_utilization: float | None = None
    gpu_memory_budget_gb: float | None = None
    extra_args: list[str] = Field(default_factory=list)
    keep_alive: bool = False
    dry_run: bool = False


class ContainerRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    container_id: str
    name: str
    status: str
    host_port: int | None = None
    model_id: str | None = None
    image: str
    memory_reservation_gb: float | None = None


class PreflightReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    requested_port: int
    estimated_model_memory_gb: float | None = None
    active_reserved_memory_gb: float = 0.0
    total_reserved_memory_gb: float | None = None
    gpu_memory_budget_gb: float | None = None
    allowed: bool = True
    reason: str | None = None


class ContainerStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    container_id: str
    name: str
    model_id: str | None = None
    cpu_percent: float | None = None
    memory_usage_gb: float | None = None
    memory_limit_gb: float | None = None
    memory_percent: float | None = None
    network_rx_mb: float | None = None
    network_tx_mb: float | None = None


class LaunchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    container_name: str
    host_port: int
    status: Literal["dry-run", "running"]
    image: str
    command: list[str]


class OverdriveProfiles(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: dict[str, ModelProfile] = Field(default_factory=dict)