from types import SimpleNamespace

from overdrive.docker_runtime import _parse_container_stats


def test_parse_container_stats_builds_snapshot() -> None:
    container = SimpleNamespace(
        id="abc123",
        name="overdrive-qwen",
        labels={"model": "org/qwen2.5-7b"},
        image=SimpleNamespace(tags=["nvcr.io/nvidia/vllm:26.04-py3"]),
    )
    raw_stats = {
        "cpu_stats": {
            "cpu_usage": {"total_usage": 2_000_000_000},
            "system_cpu_usage": 10_000_000_000,
            "online_cpus": 2,
        },
        "precpu_stats": {
            "cpu_usage": {"total_usage": 1_000_000_000},
            "system_cpu_usage": 8_000_000_000,
        },
        "memory_stats": {
            "usage": 8 * 1024**3,
            "limit": 16 * 1024**3,
        },
        "networks": {
            "eth0": {
                "rx_bytes": 10 * 1024**2,
                "tx_bytes": 4 * 1024**2,
            }
        },
    }

    snapshot = _parse_container_stats(container, raw_stats)

    assert snapshot.name == "overdrive-qwen"
    assert snapshot.model_id == "org/qwen2.5-7b"
    assert snapshot.cpu_percent == 100.0
    assert snapshot.memory_usage_gb == 8.0
    assert snapshot.memory_limit_gb == 16.0
    assert snapshot.memory_percent == 50.0
    assert snapshot.network_rx_mb == 10.0
    assert snapshot.network_tx_mb == 4.0