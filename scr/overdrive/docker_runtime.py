"""Docker orchestration for vLLM containers."""

from __future__ import annotations

import contextlib
import logging
import socket

import docker
from docker.errors import DockerException, NotFound

from overdrive.models import (
    ContainerRecord,
    ContainerStats,
    LaunchConfig,
    LaunchResult,
    ModelMetadata,
    PreflightReport,
)

VLLM_IMAGE = "nvcr.io/nvidia/vllm:26.04-py3"
OVERDRIVE_OWNER = "overdrive"
INTERNAL_VLLM_PORT = 8000
MEMORY_LABEL = "memory_reservation_gb"
LOGGER = logging.getLogger(__name__)


def _slugify_model_id(model_id: str) -> str:
    return model_id.replace("/", "-").replace("_", "-").lower()


def estimate_required_memory_gb(metadata: ModelMetadata, launch: LaunchConfig) -> float | None:
    if metadata.parameter_size_billions is None:
        return None
    bytes_per_param = (
        2.0
        if metadata.dtype in {"bf16", "bfloat16", "float16", "torch.bfloat16"}
        else 4.0
    )
    required = metadata.parameter_size_billions * 1_000_000_000 * bytes_per_param / (1024**3)
    if launch.max_model_len:
        required *= 1.1
    return round(required * 1.15, 2)


class DockerRuntime:
    def __init__(self, client: docker.DockerClient | None = None) -> None:
        self._client = client

    @property
    def client(self) -> docker.DockerClient:
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    def list_managed_containers(self) -> list[ContainerRecord]:
        try:
            containers = self.client.containers.list(
                all=True,
                filters={"label": f"owner={OVERDRIVE_OWNER}"},
            )
        except DockerException:
            return []

        results: list[ContainerRecord] = []
        for container in containers:
            port_bindings = container.attrs.get("NetworkSettings", {}).get("Ports", {})
            host_port = None
            bindings = port_bindings.get(f"{INTERNAL_VLLM_PORT}/tcp") or []
            if bindings:
                host_port = int(bindings[0]["HostPort"])
            results.append(
                ContainerRecord(
                    container_id=container.id,
                    name=container.name,
                    status=container.status,
                    host_port=host_port,
                    model_id=container.labels.get("model"),
                    image=container.image.tags[0] if container.image.tags else VLLM_IMAGE,
                    memory_reservation_gb=_parse_label_float(container.labels.get(MEMORY_LABEL)),
                )
            )
        return results

    def reserved_memory_gb(self) -> float:
        return round(
            sum(record.memory_reservation_gb or 0.0 for record in self.list_managed_containers()),
            2,
        )

    def preflight_launch(self, metadata: ModelMetadata, launch: LaunchConfig) -> PreflightReport:
        estimate = estimate_required_memory_gb(metadata, launch)
        active_reserved = self.reserved_memory_gb()
        total_reserved = None if estimate is None else round(active_reserved + estimate, 2)
        allowed = True
        reason = None

        if (
            total_reserved is not None
            and launch.gpu_memory_budget_gb is not None
            and total_reserved > launch.gpu_memory_budget_gb
        ):
            allowed = False
            reason = (
                f"Projected reserved memory {total_reserved} GiB exceeds budget "
                f"{launch.gpu_memory_budget_gb} GiB."
            )

        return PreflightReport(
            model_id=metadata.model_id,
            requested_port=launch.host_port,
            estimated_model_memory_gb=estimate,
            active_reserved_memory_gb=active_reserved,
            total_reserved_memory_gb=total_reserved,
            gpu_memory_budget_gb=launch.gpu_memory_budget_gb,
            allowed=allowed,
            reason=reason,
        )

    def used_ports(self) -> set[int]:
        return {
            record.host_port
            for record in self.list_managed_containers()
            if record.host_port is not None
        }

    def reserve_port(
        self,
        preferred: int | None = None,
        start: int = 8000,
        stop: int = 8100,
    ) -> int:
        occupied = self.used_ports()
        candidates = [preferred] if preferred is not None else []
        candidates.extend(port for port in range(start, stop) if port != preferred)

        for port in candidates:
            if port is None or port in occupied:
                continue
            with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                if sock.connect_ex(("127.0.0.1", port)) != 0:
                    return port
        raise RuntimeError("No free port available in the configured range.")

    def build_command(self, launch: LaunchConfig) -> list[str]:
        command = [
            "--model",
            "/models/current",
            "--port",
            str(INTERNAL_VLLM_PORT),
            "--tensor-parallel-size",
            str(launch.tensor_parallel_size),
        ]
        if launch.max_model_len is not None:
            command.extend(["--max-model-len", str(launch.max_model_len)])
        if launch.kv_cache_dtype:
            command.extend(["--kv-cache-dtype", launch.kv_cache_dtype])
        if launch.gpu_memory_utilization is not None:
            command.extend(["--gpu-memory-utilization", str(launch.gpu_memory_utilization)])
        command.extend(launch.extra_args)
        return command

    def launch_model(self, metadata: ModelMetadata, launch: LaunchConfig) -> LaunchResult:
        preflight = self.preflight_launch(metadata, launch)
        if not preflight.allowed:
            raise RuntimeError(preflight.reason or "Launch preflight failed.")

        container_name = f"overdrive-{_slugify_model_id(metadata.model_id)}-{launch.host_port}"
        command = self.build_command(launch)

        if launch.dry_run:
            return LaunchResult(
                model_id=metadata.model_id,
                container_name=container_name,
                host_port=launch.host_port,
                status="dry-run",
                image=VLLM_IMAGE,
                command=command,
            )

        labels = {
            "owner": OVERDRIVE_OWNER,
            "model": metadata.model_id,
            "architecture": metadata.architecture,
            MEMORY_LABEL: str(preflight.estimated_model_memory_gb or ""),
        }
        self.client.containers.run(
            VLLM_IMAGE,
            command=command,
            name=container_name,
            detach=True,
            remove=not launch.keep_alive,
            labels=labels,
            ports={f"{INTERNAL_VLLM_PORT}/tcp": launch.host_port},
            volumes={str(metadata.snapshot_path): {"bind": "/models/current", "mode": "ro"}},
            environment={"HF_HOME": "/models", "NVIDIA_VISIBLE_DEVICES": "all"},
            device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])],
        )
        return LaunchResult(
            model_id=metadata.model_id,
            container_name=container_name,
            host_port=launch.host_port,
            status="running",
            image=VLLM_IMAGE,
            command=command,
        )

    def stop_model(self, *, model_id: str | None = None, container_name: str | None = None) -> bool:
        if not model_id and not container_name:
            raise ValueError("model_id or container_name is required")
        filters = {"label": f"owner={OVERDRIVE_OWNER}"}
        containers = self.client.containers.list(all=True, filters=filters)
        for container in containers:
            if container_name and container.name == container_name:
                container.stop()
                return True
            if model_id and container.labels.get("model") == model_id:
                container.stop()
                return True
        return False

    def cleanup(self) -> int:
        count = 0
        for container in self.client.containers.list(
            all=True,
            filters={"label": f"owner={OVERDRIVE_OWNER}"},
        ):
            with contextlib.suppress(NotFound, DockerException):
                container.stop()
                count += 1
        return count

    def stream_logs(self, container_name: str, tail: int = 50) -> list[str]:
        try:
            container = self.client.containers.get(container_name)
        except DockerException:
            return []
        return [
            line.decode("utf-8", errors="replace").rstrip()
            for line in container.logs(tail=tail).splitlines()
        ]

    def get_container_stats(self, container_name: str) -> ContainerStats | None:
        try:
            container = self.client.containers.get(container_name)
            raw_stats = container.stats(stream=False)
        except (DockerException, ValueError, TypeError) as exc:
            LOGGER.warning("Failed to read stats for container %s: %s", container_name, exc)
            return None
        return _parse_container_stats(container, raw_stats)

    def list_managed_stats(self) -> list[ContainerStats]:
        stats: list[ContainerStats] = []
        for record in self.list_managed_containers():
            try:
                snapshot = self.get_container_stats(record.name)
            except Exception as exc:  # Defensive guard against third-party client failures.
                LOGGER.warning("Skipping stats for container %s: %s", record.name, exc)
                continue
            if snapshot is not None:
                stats.append(snapshot)
        return stats


def _parse_label_float(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_container_stats(container: object, raw_stats: dict[str, object]) -> ContainerStats:
    cpu_stats = raw_stats.get("cpu_stats", {}) if isinstance(raw_stats, dict) else {}
    precpu_stats = raw_stats.get("precpu_stats", {}) if isinstance(raw_stats, dict) else {}
    memory_stats = raw_stats.get("memory_stats", {}) if isinstance(raw_stats, dict) else {}
    networks = raw_stats.get("networks", {}) if isinstance(raw_stats, dict) else {}

    cpu_percent = _calculate_cpu_percent(cpu_stats, precpu_stats)
    memory_usage = _bytes_to_gb(memory_stats.get("usage"))
    memory_limit = _bytes_to_gb(memory_stats.get("limit"))
    memory_percent = None
    if memory_usage is not None and memory_limit:
        memory_percent = round((memory_usage / memory_limit) * 100, 2)

    rx_total, tx_total = _network_totals(networks)
    labels = getattr(container, "labels", {}) or {}
    return ContainerStats(
        container_id=getattr(container, "id", ""),
        name=getattr(container, "name", "unknown"),
        model_id=labels.get("model"),
        cpu_percent=cpu_percent,
        memory_usage_gb=memory_usage,
        memory_limit_gb=memory_limit,
        memory_percent=memory_percent,
        network_rx_mb=_bytes_to_mb(rx_total),
        network_tx_mb=_bytes_to_mb(tx_total),
    )


def _calculate_cpu_percent(
    cpu_stats: dict[str, object],
    precpu_stats: dict[str, object],
) -> float | None:
    current_total = _nested_int(cpu_stats, "cpu_usage", "total_usage")
    previous_total = _nested_int(precpu_stats, "cpu_usage", "total_usage")
    current_system = _nested_int(cpu_stats, "system_cpu_usage")
    previous_system = _nested_int(precpu_stats, "system_cpu_usage")
    online_cpus = _nested_int(cpu_stats, "online_cpus") or 1

    if None in {current_total, previous_total, current_system, previous_system}:
        return None

    cpu_delta = current_total - previous_total
    system_delta = current_system - previous_system
    if cpu_delta <= 0 or system_delta <= 0:
        return 0.0
    return round((cpu_delta / system_delta) * online_cpus * 100, 2)


def _nested_int(mapping: dict[str, object], *keys: str) -> int | None:
    current: object = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if isinstance(current, int):
        return current
    return None


def _bytes_to_gb(value: object) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return round(float(value) / (1024**3), 2)


def _bytes_to_mb(value: object) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return round(float(value) / (1024**2), 2)


def _network_totals(networks: object) -> tuple[int, int]:
    if not isinstance(networks, dict):
        return 0, 0
    rx_total = 0
    tx_total = 0
    for stats in networks.values():
        if not isinstance(stats, dict):
            continue
        rx_total += int(stats.get("rx_bytes", 0) or 0)
        tx_total += int(stats.get("tx_bytes", 0) or 0)
    return rx_total, tx_total