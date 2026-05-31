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


def test_launch_environment_includes_hf_tokens(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_token")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hub_token")

    runtime = DockerRuntime(client=None)
    env = runtime._launch_environment()

    assert env["HF_HOME"] == "/models"
    assert env["NVIDIA_VISIBLE_DEVICES"] == "all"
    assert env["HF_TOKEN"] == "hf_token"
    assert env["HUGGING_FACE_HUB_TOKEN"] == "hub_token"


def test_launch_environment_skips_empty_hf_tokens(monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    runtime = DockerRuntime(client=None)
    env = runtime._launch_environment()

    assert "HF_TOKEN" not in env
    assert "HUGGING_FACE_HUB_TOKEN" not in env


def test_build_docker_run_command_redacts_tokens(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    metadata = ModelMetadata(
        model_id="meta-llama/meta-llama/Meta-Llama-3.1-8B-Instruct",
        model_name="Meta-Llama-3.1-8B-Instruct",
        architecture="LlamaForCausalLM",
        model_type="llama",
        parameter_size_billions=8.0,
        dtype="bfloat16",
        snapshot_path="/models/meta-llama/meta-llama/Meta-Llama-3.1-8B-Instruct",
        config_path="/models/meta-llama/meta-llama/Meta-Llama-3.1-8B-Instruct/config.json",
        profile=ModelProfile(),
    )
    launch = LaunchConfig(
        model_id=metadata.model_id,
        snapshot_path=metadata.snapshot_path,
        host_port=8000,
        tensor_parallel_size=1,
        max_model_len=32768,
        kv_cache_dtype="auto",
    )

    runtime = DockerRuntime(client=None)
    command = runtime.build_docker_run_command(metadata, launch)

    assert "docker run" in command
    assert "vllm serve --model /models/current" in command
    assert "HF_TOKEN=<set>" in command
    assert "secret-token" not in command