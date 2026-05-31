import json
from pathlib import Path
from types import SimpleNamespace

import overdrive.cli as cli_module
from click.testing import CliRunner
from overdrive.cli import cli
from overdrive.models import ContainerStats


def test_scan_command_lists_model(tmp_path: Path) -> None:
    hub_root = tmp_path / "hub"
    snapshot = hub_root / "models--org--gemma-4-27b" / "snapshots" / "rev1"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps({"architectures": ["GemmaForCausalLM"], "torch_dtype": "bfloat16"}),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["--hub-root", str(hub_root), "scan"])

    assert result.exit_code == 0
    assert "org/gemma-4-27b" in result.output


def test_scan_command_json_output(tmp_path: Path) -> None:
    hub_root = tmp_path / "hub"
    snapshot = hub_root / "models--org--gemma-4-27b" / "snapshots" / "rev1"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps({"architectures": ["GemmaForCausalLM"], "torch_dtype": "bfloat16"}),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["--hub-root", str(hub_root), "scan", "--json-output"])

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload[0]["model_id"] == "org/gemma-4-27b"


def test_plan_command_reports_memory_budget_failure(tmp_path: Path) -> None:
    hub_root = tmp_path / "hub"
    snapshot = hub_root / "models--org--gemma-4-27b" / "snapshots" / "rev1"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps({"architectures": ["GemmaForCausalLM"], "torch_dtype": "bfloat16"}),
        encoding="utf-8",
    )
    profiles = tmp_path / "profiles.yaml"
    profiles.write_text(
        "models:\n  org/gemma-4-27b:\n    gpu_memory_budget_gb: 1\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--hub-root", str(hub_root), "--profiles", str(profiles), "plan", "org/gemma-4-27b"],
    )

    assert result.exit_code == 0
    assert "allowed=False" in result.output
    assert "reason=" in result.output


def test_plan_command_json_output(tmp_path: Path) -> None:
    hub_root = tmp_path / "hub"
    snapshot = hub_root / "models--org--gemma-4-27b" / "snapshots" / "rev1"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps({"architectures": ["GemmaForCausalLM"], "torch_dtype": "bfloat16"}),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--hub-root", str(hub_root), "plan", "org/gemma-4-27b", "--json-output"],
    )

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload["model_id"] == "org/gemma-4-27b"
    assert payload["requested_port"] >= 8000


def test_profile_command_json_output(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles.yaml"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--profiles",
            str(profiles),
            "profile",
            "org/gemma-4-27b",
            "--json-output",
            "--preferred-port",
            "8002",
            "--max-model-len",
            "32768",
        ],
    )

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload["saved"] is True
    assert payload["model_id"] == "org/gemma-4-27b"
    assert payload["profile"]["preferred_port"] == 8002


def test_up_command_json_output(tmp_path: Path) -> None:
    hub_root = tmp_path / "hub"
    snapshot = hub_root / "models--org--gemma-4-27b" / "snapshots" / "rev1"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps({"architectures": ["GemmaForCausalLM"], "torch_dtype": "bfloat16"}),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--hub-root", str(hub_root), "up", "org/gemma-4-27b", "--json-output", "--dry-run"],
    )

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload["model_id"] == "org/gemma-4-27b"
    assert payload["status"] == "dry-run"


def test_stop_command_json_output(monkeypatch) -> None:
    fake_manager = SimpleNamespace(stop_model=lambda model_id: True)

    monkeypatch.setattr(cli_module, "build_manager", lambda hub_root, profiles: fake_manager)

    runner = CliRunner()
    result = runner.invoke(cli, ["stop", "org/gemma-4-27b", "--json-output"])

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload == {"model_id": "org/gemma-4-27b", "stopped": True}


def test_cleanup_command_json_output(monkeypatch) -> None:
    fake_manager = SimpleNamespace(cleanup=lambda: 2)

    monkeypatch.setattr(cli_module, "build_manager", lambda hub_root, profiles: fake_manager)

    runner = CliRunner()
    result = runner.invoke(cli, ["cleanup", "--json-output"])

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload == {"stopped_count": 2}


def test_models_search_command_json_output(monkeypatch) -> None:
    monkeypatch.setattr(
        cli_module,
        "search_models",
        lambda *args, **kwargs: [{"id": "Qwen/Qwen3-8B", "downloads": 123, "likes": 45}],
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["models-search", "qwen", "--json-output"])

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload[0]["id"] == "Qwen/Qwen3-8B"


def test_models_download_command_json_output(monkeypatch) -> None:
    monkeypatch.setattr(
        cli_module,
        "download_model",
        lambda *args, **kwargs: {
            "model_id": "Qwen/Qwen3-8B",
            "download_path": "E:/models/Qwen/Qwen3-8B",
            "dry_run": False,
            "stdout": "E:/models/Qwen/Qwen3-8B",
        },
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["models-download", "Qwen/Qwen3-8B", "--json-output"])

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload["download_path"] == "E:/models/Qwen/Qwen3-8B"


def test_stats_command_watch_outputs_multiple_samples(monkeypatch) -> None:
    calls = {"count": 0}

    class FakeRuntime:
        def list_managed_stats(self) -> list[ContainerStats]:
            calls["count"] += 1
            return [
                ContainerStats(
                    container_id="abc123",
                    name="overdrive-qwen",
                    model_id="org/qwen2.5-7b",
                    cpu_percent=25.0,
                    memory_usage_gb=8.0,
                    memory_limit_gb=16.0,
                    memory_percent=50.0,
                    network_rx_mb=12.0,
                    network_tx_mb=4.0,
                )
            ]

    fake_manager = SimpleNamespace(runtime=FakeRuntime())

    monkeypatch.setattr(cli_module, "build_manager", lambda hub_root, profiles: fake_manager)
    monkeypatch.setattr(cli_module.time, "sleep", lambda interval: None)

    runner = CliRunner()
    result = runner.invoke(cli, ["stats", "--watch", "--interval", "0", "--samples", "2"])

    assert result.exit_code == 0
    assert calls["count"] == 2
    assert "sample=1" in result.output
    assert "sample=2" in result.output
    assert "overdrive-qwen" in result.output


def test_stats_command_json_output(monkeypatch) -> None:
    class FakeRuntime:
        def list_managed_stats(self) -> list[ContainerStats]:
            return [
                ContainerStats(
                    container_id="abc123",
                    name="overdrive-qwen",
                    model_id="org/qwen2.5-7b",
                    cpu_percent=25.0,
                    memory_usage_gb=8.0,
                    memory_limit_gb=16.0,
                    memory_percent=50.0,
                    network_rx_mb=12.0,
                    network_tx_mb=4.0,
                )
            ]

    fake_manager = SimpleNamespace(runtime=FakeRuntime())

    monkeypatch.setattr(cli_module, "build_manager", lambda hub_root, profiles: fake_manager)

    runner = CliRunner()
    result = runner.invoke(cli, ["stats", "--json-output"])

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload[0]["name"] == "overdrive-qwen"


def test_stats_command_json_watch_output(monkeypatch) -> None:
    class FakeRuntime:
        def __init__(self) -> None:
            self.calls = 0

        def list_managed_stats(self) -> list[ContainerStats]:
            self.calls += 1
            return [
                ContainerStats(
                    container_id="abc123",
                    name="overdrive-qwen",
                    model_id="org/qwen2.5-7b",
                    cpu_percent=25.0,
                    memory_usage_gb=8.0,
                    memory_limit_gb=16.0,
                    memory_percent=50.0,
                    network_rx_mb=12.0,
                    network_tx_mb=4.0,
                )
            ]

    runtime = FakeRuntime()
    fake_manager = SimpleNamespace(runtime=runtime)

    monkeypatch.setattr(cli_module, "build_manager", lambda hub_root, profiles: fake_manager)
    monkeypatch.setattr(cli_module.time, "sleep", lambda interval: None)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["stats", "--json-output", "--watch", "--interval", "0", "--samples", "2"],
    )

    decoder = json.JSONDecoder()
    payloads: list[dict[str, object]] = []
    raw_output = result.output.strip()
    position = 0
    while position < len(raw_output):
        while position < len(raw_output) and raw_output[position].isspace():
            position += 1
        if position >= len(raw_output):
            break
        payload, offset = decoder.raw_decode(raw_output, position)
        payloads.append(payload)
        position = offset

    assert result.exit_code == 0
    assert runtime.calls == 2
    assert payloads[0]["sample"] == 1
    assert payloads[1]["sample"] == 2


def test_web_command_invokes_runner(monkeypatch) -> None:
    fake_manager = SimpleNamespace()
    captured: dict[str, object] = {}

    monkeypatch.setattr(cli_module, "build_manager", lambda hub_root, profiles: fake_manager)
    monkeypatch.setattr(
        cli_module,
        "run_web",
        lambda manager, host, port: captured.update(
            {"manager": manager, "host": host, "port": port}
        ),
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["web", "--host", "0.0.0.0", "--port", "9090"])

    assert result.exit_code == 0
    assert captured == {"manager": fake_manager, "host": "0.0.0.0", "port": 9090}