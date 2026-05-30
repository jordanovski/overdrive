"""Local GPU hardware discovery and launch recommendations."""

from __future__ import annotations

import csv
import math
import subprocess
from dataclasses import dataclass

from overdrive.docker_runtime import estimate_required_memory_gb
from overdrive.models import LaunchConfig, ModelMetadata

GPU_QUERY = [
    "nvidia-smi",
    "--query-gpu=name,memory.total,memory.free",
    "--format=csv,noheader,nounits",
]

MAX_MODEL_LEN_CANDIDATES = [32768, 16384, 8192, 4096, 2048, 1024]


@dataclass(frozen=True)
class GPUDevice:
    name: str
    total_memory_gb: float
    free_memory_gb: float


def detect_gpus() -> list[GPUDevice]:
    try:
        completed = subprocess.run(
            GPU_QUERY,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return _parse_nvidia_smi_csv(completed.stdout)


def _parse_nvidia_smi_csv(stdout: str) -> list[GPUDevice]:
    devices: list[GPUDevice] = []
    reader = csv.reader(line for line in stdout.splitlines() if line.strip())
    for row in reader:
        if len(row) != 3:
            continue
        try:
            total_memory_gb = round(float(row[1].strip()) / 1024, 2)
            free_memory_gb = round(float(row[2].strip()) / 1024, 2)
        except ValueError:
            continue
        devices.append(
            GPUDevice(
                name=row[0].strip(),
                total_memory_gb=total_memory_gb,
                free_memory_gb=free_memory_gb,
            )
        )
    return devices


def recommended_gpu_budget_gb(model: ModelMetadata, devices: list[GPUDevice]) -> float:
    if model.profile.gpu_memory_budget_gb is not None:
        return model.profile.gpu_memory_budget_gb
    if not devices:
        return 110.0
    total_free = sum(device.free_memory_gb for device in devices)
    reserve = max(4.0, 2.0 * len(devices))
    return round(max(1.0, min(total_free * 0.9, total_free - reserve)), 1)


def recommended_tensor_parallel_size(model: ModelMetadata, devices: list[GPUDevice]) -> int:
    if model.profile.tensor_parallel_size > 0:
        return model.profile.tensor_parallel_size
    if not devices:
        return 1

    estimate = _estimated_model_memory_gb(model, tensor_parallel_size=1)
    if estimate is None:
        return 1

    free_memories = sorted((device.free_memory_gb for device in devices), reverse=True)
    for candidate in range(1, len(free_memories) + 1):
        per_gpu_budget = min(free_memories[:candidate]) * 0.9
        if (estimate / candidate) <= per_gpu_budget:
            return candidate
    return len(free_memories)


def recommended_max_model_len(model: ModelMetadata, devices: list[GPUDevice]) -> int:
    if model.profile.max_model_len is not None:
        return model.profile.max_model_len

    budget_gb = recommended_gpu_budget_gb(model, devices)
    tensor_parallel = recommended_tensor_parallel_size(model, devices)
    model_memory_gb = _estimated_model_memory_gb(model, tensor_parallel_size=tensor_parallel) or 0.0
    available_kv_gb = budget_gb - model_memory_gb
    if available_kv_gb <= 0:
        return 1024

    kv_bytes_per_token = _estimated_kv_bytes_per_token(model)
    if kv_bytes_per_token is None or kv_bytes_per_token <= 0:
        parameter_size = model.parameter_size_billions or 0.0
        if parameter_size >= 30:
            return 4096
        if parameter_size >= 14:
            return 8192
        return 16384

    capacity = int((available_kv_gb * (1024**3)) / kv_bytes_per_token)
    for candidate in MAX_MODEL_LEN_CANDIDATES:
        if capacity >= candidate:
            return candidate
    return 1024


def recommended_kv_cache_dtype(model: ModelMetadata, devices: list[GPUDevice]) -> str:
    if model.profile.kv_cache_dtype:
        return model.profile.kv_cache_dtype
    budget_gb = recommended_gpu_budget_gb(model, devices)
    tensor_parallel = recommended_tensor_parallel_size(model, devices)
    estimate = _estimated_model_memory_gb(model, tensor_parallel_size=tensor_parallel)
    if estimate is None or budget_gb <= 0:
        return "auto"
    if estimate / budget_gb >= 0.7:
        return "fp8"
    return "auto"


def _estimated_model_memory_gb(model: ModelMetadata, *, tensor_parallel_size: int) -> float | None:
    launch = LaunchConfig(
        model_id=model.model_id,
        snapshot_path=model.snapshot_path,
        host_port=8000,
        tensor_parallel_size=tensor_parallel_size,
    )
    return estimate_required_memory_gb(model, launch)


def _estimated_kv_bytes_per_token(model: ModelMetadata) -> int | None:
    text_config = model.config_data.get("text_config")
    config = text_config if isinstance(text_config, dict) else model.config_data
    if not isinstance(config, dict):
        return None

    num_layers = _int_value(config.get("num_hidden_layers"))
    num_key_value_heads = _int_value(config.get("num_key_value_heads"))
    num_attention_heads = _int_value(config.get("num_attention_heads"))
    head_dim = _int_value(config.get("head_dim"))
    hidden_size = _int_value(config.get("hidden_size"))
    bytes_per_element = 2 if model.dtype in {"bf16", "bfloat16", "float16", "torch.bfloat16"} else 4

    if num_layers is None:
        return None
    if head_dim is not None and (num_key_value_heads or num_attention_heads):
        kv_heads = num_key_value_heads or num_attention_heads
        return 2 * num_layers * kv_heads * head_dim * bytes_per_element
    if hidden_size is not None:
        return 2 * num_layers * hidden_size * bytes_per_element
    return None


def _int_value(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return math.floor(value)
    return None