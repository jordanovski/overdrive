import socket
from types import SimpleNamespace

from overdrive.docker_runtime import DockerRuntime, estimate_required_memory_gb
from overdrive.models import LaunchConfig, ModelMetadata, ModelProfile


class FakeRuntime(DockerRuntime):
    def used_ports(self) -> set[int]:
        return {8000, 8001}


class FakeSocket:
    def __init__(self, *args, **kwargs) -> None:
        self.closed = False

    def setsockopt(self, *args, **kwargs) -> None:
        return None

    def connect_ex(self, address: tuple[str, int]) -> int:
        return 0 if address[1] == 8003 else 1

    def close(self) -> None:
        self.closed = True


def test_reserve_port_skips_used_ports(monkeypatch) -> None:
    monkeypatch.setattr(socket, "socket", FakeSocket)
    runtime = FakeRuntime(client=None)
    assert runtime.reserve_port(start=8000, stop=8005) == 8002


def test_estimate_required_memory_gb_uses_dtype() -> None:
    metadata = ModelMetadata(
        model_id="org/model-7b",
        model_name="model-7b",
        architecture="LlamaForCausalLM",
        model_type="llama",
        parameter_size_billions=7,
        dtype="bf16",
        snapshot_path="/tmp/model",
        config_path="/tmp/model/config.json",
        profile=ModelProfile(),
    )
    launch = LaunchConfig(
        model_id=metadata.model_id,
        snapshot_path=metadata.snapshot_path,
        host_port=8002,
    )

    estimate = estimate_required_memory_gb(metadata, launch)

    assert estimate is not None
    assert estimate > 10


def test_get_container_stats_handles_invalid_payload() -> None:
    class BrokenContainer:
        id = "cid"
        name = "broken"
        labels = {"model": "org/model"}

        def stats(self, stream=False):
            raise ValueError("invalid json")

    fake_client = SimpleNamespace(
        containers=SimpleNamespace(
            get=lambda name: BrokenContainer(),
        )
    )
    runtime = DockerRuntime(client=fake_client)

    assert runtime.get_container_stats("broken") is None


def test_list_managed_stats_skips_broken_container_stats() -> None:
    records = [
        SimpleNamespace(name="good", host_port=8000),
        SimpleNamespace(name="bad", host_port=8001),
    ]
    runtime = DockerRuntime(client=None)
    runtime.list_managed_containers = lambda: records  # type: ignore[method-assign]
    runtime.get_container_stats = lambda name: (_ for _ in ()).throw(RuntimeError("boom")) if name == "bad" else SimpleNamespace()  # type: ignore[method-assign]

    stats = runtime.list_managed_stats()

    assert len(stats) == 1