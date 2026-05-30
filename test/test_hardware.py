from pathlib import Path

from overdrive.hardware import (
    GPUDevice,
    _parse_nvidia_smi_csv,
    recommended_gpu_budget_gb,
    recommended_kv_cache_dtype,
    recommended_max_model_len,
    recommended_tensor_parallel_size,
)
from overdrive.models import ModelMetadata, ModelProfile


def _model(*, size: float | None, dtype: str = "bfloat16", profile: ModelProfile | None = None):
    return ModelMetadata(
        model_id="qwen-35b-moe",
        model_name="qwen-35b-moe",
        architecture="Qwen",
        model_type="qwen",
        parameter_size_billions=size,
        dtype=dtype,
        snapshot_path=Path("/models/qwen-35b-moe"),
        config_path=Path("/models/qwen-35b-moe/config.json"),
        config_data={
            "text_config": {
                "dtype": dtype,
                "num_hidden_layers": 40,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "hidden_size": 4096,
            }
        },
        profile=profile or ModelProfile(),
    )


def test_parse_nvidia_smi_csv() -> None:
    devices = _parse_nvidia_smi_csv("GPU 0, 32768, 28672\nGPU 1, 32768, 24576\n")

    assert len(devices) == 2
    assert devices[0].name == "GPU 0"
    assert devices[0].free_memory_gb == 28.0


def test_hardware_aware_recommendations() -> None:
    devices = [GPUDevice(name="RTX", total_memory_gb=32.0, free_memory_gb=28.0)]
    model = _model(size=7.0)

    assert recommended_gpu_budget_gb(model, devices) == 24.0
    assert recommended_tensor_parallel_size(model, devices) == 1
    assert recommended_max_model_len(model, devices) >= 1024
    assert recommended_kv_cache_dtype(model, devices) == "auto"


def test_large_model_prefers_fp8_kv_cache_under_pressure() -> None:
    devices = [GPUDevice(name="RTX", total_memory_gb=32.0, free_memory_gb=24.0)]
    model = _model(size=35.0)

    assert recommended_tensor_parallel_size(model, devices) == 1
    assert recommended_max_model_len(model, devices) == 1024
    assert recommended_kv_cache_dtype(model, devices) == "fp8"


def test_profile_values_override_hardware_recommendations() -> None:
    devices = [GPUDevice(name="RTX", total_memory_gb=32.0, free_memory_gb=28.0)]
    model = _model(
        size=7.0,
        profile=ModelProfile(
            max_model_len=32768,
            tensor_parallel_size=2,
            kv_cache_dtype="fp8_e4m3",
            gpu_memory_budget_gb=96.0,
        ),
    )

    assert recommended_max_model_len(model, devices) == 32768
    assert recommended_tensor_parallel_size(model, devices) == 2
    assert recommended_kv_cache_dtype(model, devices) == "fp8_e4m3"
    assert recommended_gpu_budget_gb(model, devices) == 96.0