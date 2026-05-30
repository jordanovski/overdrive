from pathlib import Path

from overdrive.docker_runtime import DockerRuntime
from overdrive.models import ContainerRecord, ModelMetadata, ModelProfile
from overdrive.state import EngineStateManager


class PreflightRuntime(DockerRuntime):
    def __init__(self) -> None:
        super().__init__(client=None)

    def reserve_port(
        self,
        preferred: int | None = None,
        start: int = 8000,
        stop: int = 8100,
    ) -> int:
        return preferred or 8002

    def list_managed_containers(self) -> list[ContainerRecord]:
        return [
            ContainerRecord(
                container_id="abc123",
                name="overdrive-existing",
                status="running",
                host_port=8000,
                model_id="org/existing-13b",
                image="nvcr.io/nvidia/vllm:26.04-py3",
                memory_reservation_gb=20.0,
            )
        ]


class PreflightManager(EngineStateManager):
    def __init__(self, model: ModelMetadata) -> None:
        super().__init__(hub_root=Path("/tmp/hub"), runtime=PreflightRuntime())
        self._model = model

    def get_model(self, model_id: str) -> ModelMetadata:
        return self._model


def test_preflight_accounts_for_active_reservations() -> None:
    model = ModelMetadata(
        model_id="org/gemma-4-27b",
        model_name="gemma-4-27b",
        architecture="GemmaForCausalLM",
        model_type="gemma",
        parameter_size_billions=7,
        dtype="bfloat16",
        snapshot_path="/tmp/model",
        config_path="/tmp/model/config.json",
        profile=ModelProfile(gpu_memory_budget_gb=30.0),
    )
    manager = PreflightManager(model)

    report = manager.preflight_launch("org/gemma-4-27b")

    assert report.estimated_model_memory_gb is not None
    assert report.active_reserved_memory_gb == 20.0
    assert report.total_reserved_memory_gb is not None
    assert report.total_reserved_memory_gb > 30.0
    assert report.allowed is False
    assert report.reason is not None