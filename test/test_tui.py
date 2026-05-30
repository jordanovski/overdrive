from pathlib import Path
from types import SimpleNamespace

from overdrive.models import ModelMetadata, ModelProfile, PreflightReport
from overdrive.tui import OverdriveApp, _display_dtype


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
def test_display_dtype_falls_back_to_text_config() -> None:
    model = _model(size=35.0)

    assert _display_dtype(model) == "bfloat16"


def test_format_preflight_report_includes_reason() -> None:
    app = OverdriveApp(SimpleNamespace())
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

    formatted = app._format_preflight_report(report)

    assert "allowed=False" in formatted
    assert "port=8000" in formatted
    assert "reason=Projected reserved memory 52.0 GiB exceeds budget 48.0 GiB." in formatted


def test_action_plan_selected_logs_preflight() -> None:
    model = _model(size=35.0)
    report = PreflightReport(
        model_id=model.model_id,
        requested_port=8000,
        estimated_model_memory_gb=42.0,
        active_reserved_memory_gb=10.0,
        total_reserved_memory_gb=52.0,
        gpu_memory_budget_gb=110.0,
        allowed=True,
    )
    manager = SimpleNamespace(preflight_launch=lambda model_id, **kwargs: report)
    app = OverdriveApp(manager)
    app.selected = model
    logged: list[str] = []
    app._log = logged.append
    app._current_launch_settings = lambda: {
        "preferred_port": 8000,
        "max_model_len": 4096,
        "tensor_parallel_size": 1,
        "kv_cache_dtype": "auto",
        "gpu_memory_budget_gb": 110.0,
    }

    app.action_plan_selected()

    assert logged
    assert "allowed=True" in logged[0]


def test_action_save_profile_persists_current_fields() -> None:
    model = _model(size=35.0)
    saved: dict[str, object] = {}
    refreshed = _model(size=35.0, profile=ModelProfile(preferred_port=8001, max_model_len=4096))

    def save_profile(model_id: str, profile: ModelProfile):
        saved["model_id"] = model_id
        saved["profile"] = profile
        return Path("/tmp/profiles.yaml")

    manager = SimpleNamespace(
        save_profile=save_profile,
        get_model=lambda model_id: refreshed,
    )
    app = OverdriveApp(manager)
    app.selected = model
    logged: list[str] = []
    app._log = logged.append
    app.action_refresh = lambda: None
    populated: list[ModelMetadata] = []
    app._populate_fields = populated.append
    app._current_launch_settings = lambda: {
        "preferred_port": 8001,
        "max_model_len": 4096,
        "tensor_parallel_size": 1,
        "kv_cache_dtype": "auto",
        "gpu_memory_budget_gb": 110.0,
    }

    app.action_save_profile()

    assert saved["model_id"] == model.model_id
    saved_profile = saved["profile"]
    assert isinstance(saved_profile, ModelProfile)
    assert saved_profile.preferred_port == 8001
    assert saved_profile.kv_cache_dtype == "auto"
    assert populated == [refreshed]
    assert "saved profile" in logged[0]