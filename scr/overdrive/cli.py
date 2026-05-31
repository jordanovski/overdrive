"""Click-based CLI for Overdrive."""

from __future__ import annotations

import json
import time
from pathlib import Path

import click

from overdrive.hf_cli import HfCliError, download_model, search_models
from overdrive.models import ContainerStats, ModelProfile
from overdrive.state import EngineStateManager
from overdrive.web import run_web


def build_manager(hub_root: str | None, profiles: str | None) -> EngineStateManager:
    return EngineStateManager(
        hub_root=Path(hub_root).expanduser() if hub_root else None,
        profiles_path=Path(profiles).expanduser() if profiles else None,
    )


def _format_stats_row(item: ContainerStats) -> str:
    return "\t".join(
        [
            item.name,
            item.model_id or "",
            f"cpu={item.cpu_percent}",
            f"mem={item.memory_usage_gb}/{item.memory_limit_gb}GiB",
            f"mem_pct={item.memory_percent}",
            f"rx={item.network_rx_mb}MB",
            f"tx={item.network_tx_mb}MB",
        ]
    )


def _emit_stats_snapshot(manager: EngineStateManager, *, sample_index: int | None = None) -> None:
    snapshots = manager.runtime.list_managed_stats()
    if sample_index is not None:
        click.echo(f"sample={sample_index}")
    if not snapshots:
        click.echo("No managed containers.")
        return
    for item in snapshots:
        click.echo(_format_stats_row(item))


def _echo_json(payload: object) -> None:
    click.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _raise_click_error(exc: HfCliError) -> None:
    raise click.ClickException(str(exc)) from exc


@click.group()
@click.option(
    "--hub-root",
    type=click.Path(path_type=str),
    default=None,
    envvar="OVERDRIVE_HUB_ROOT",
    help="Hugging Face hub root.",
)
@click.option(
    "--profiles",
    type=click.Path(path_type=str),
    default=None,
    envvar="OVERDRIVE_PROFILES",
    help="Override profiles YAML path.",
)
@click.pass_context
def cli(ctx: click.Context, hub_root: str | None, profiles: str | None) -> None:
    ctx.obj = build_manager(hub_root, profiles)


@cli.command("scan")
@click.option("--json-output", is_flag=True, help="Emit machine-readable JSON.")
@click.pass_obj
def scan(manager: EngineStateManager, json_output: bool) -> None:
    models = manager.discover_models()
    if json_output:
        _echo_json([model.model_dump(mode="json") for model in models])
        return
    if not models:
        click.echo(f"No models found under {manager.hub_root}")
        return
    for model in models:
        click.echo(
            f"{model.model_id}\t{model.architecture}\t{model.dtype}\t{model.snapshot_path}"
        )


@cli.command("models-search")
@click.argument("query")
@click.option("--json-output", is_flag=True, help="Emit machine-readable JSON.")
@click.option("--limit", type=int, default=10, show_default=True)
@click.option("--author", type=str, default=None)
@click.option(
    "--sort",
    type=click.Choice(["created_at", "downloads", "last_modified", "likes", "trending_score"]),
    default=None,
)
@click.option("--filter", "filters", multiple=True)
@click.option("--num-parameters", type=str, default=None)
@click.option("--token", type=str, default=None)
def models_search(
    query: str,
    json_output: bool,
    limit: int,
    author: str | None,
    sort: str | None,
    filters: tuple[str, ...],
    num_parameters: str | None,
    token: str | None,
) -> None:
    try:
        results = search_models(
            query,
            limit=limit,
            author=author,
            sort=sort,
            filters=list(filters),
            num_parameters=num_parameters,
            token=token,
        )
    except HfCliError as exc:
        _raise_click_error(exc)

    if json_output:
        _echo_json(results)
        return

    if not results:
        click.echo("No Hugging Face models matched your search.")
        return
    for item in results:
        click.echo(
            "\t".join(
                [
                    str(item.get("id", "")),
                    f"downloads={item.get('downloads')}",
                    f"likes={item.get('likes')}",
                    f"pipeline={item.get('pipeline_tag')}",
                ]
            )
        )


@cli.command("models-download")
@click.argument("model_id")
@click.option("--json-output", is_flag=True, help="Emit machine-readable JSON.")
@click.option("--local-dir", type=click.Path(path_type=Path), default=None)
@click.option("--cache-dir", type=click.Path(path_type=Path), default=None)
@click.option("--revision", type=str, default=None)
@click.option("--include", "includes", multiple=True)
@click.option("--exclude", "excludes", multiple=True)
@click.option("--token", type=str, default=None)
@click.option("--max-workers", type=int, default=None)
@click.option("--force-download", is_flag=True)
@click.option("--dry-run", is_flag=True)
def models_download(
    model_id: str,
    json_output: bool,
    local_dir: Path | None,
    cache_dir: Path | None,
    revision: str | None,
    includes: tuple[str, ...],
    excludes: tuple[str, ...],
    token: str | None,
    max_workers: int | None,
    force_download: bool,
    dry_run: bool,
) -> None:
    try:
        result = download_model(
            model_id,
            local_dir=local_dir,
            cache_dir=cache_dir,
            revision=revision,
            includes=list(includes),
            excludes=list(excludes),
            token=token,
            max_workers=max_workers,
            force_download=force_download,
            dry_run=dry_run,
        )
    except HfCliError as exc:
        _raise_click_error(exc)

    if json_output:
        _echo_json(result)
        return
    if dry_run:
        click.echo(result["stdout"])
        return
    click.echo(f"Downloaded {model_id} to {result['download_path']}")


@cli.command("profile")
@click.argument("model_id")
@click.option("--json-output", is_flag=True, help="Emit machine-readable JSON.")
@click.option("--preferred-port", type=int, default=None)
@click.option("--max-model-len", type=int, default=None)
@click.option("--tensor-parallel-size", type=int, default=1)
@click.option("--kv-cache-dtype", type=str, default=None)
@click.option("--gpu-memory-utilization", type=float, default=None)
@click.option("--gpu-memory-budget-gb", type=float, default=None)
@click.option("--extra-arg", "extra_args", multiple=True)
@click.pass_obj
def profile(
    manager: EngineStateManager,
    model_id: str,
    json_output: bool,
    preferred_port: int | None,
    max_model_len: int | None,
    tensor_parallel_size: int,
    kv_cache_dtype: str | None,
    gpu_memory_utilization: float | None,
    gpu_memory_budget_gb: float | None,
    extra_args: tuple[str, ...],
) -> None:
    path = manager.save_profile(
        model_id,
        ModelProfile(
            preferred_port=preferred_port,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            kv_cache_dtype=kv_cache_dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            gpu_memory_budget_gb=gpu_memory_budget_gb,
            extra_args=list(extra_args),
        ),
    )
    if json_output:
        _echo_json(
            {
                "model_id": model_id,
                "path": str(path),
                "profile": {
                    "preferred_port": preferred_port,
                    "max_model_len": max_model_len,
                    "tensor_parallel_size": tensor_parallel_size,
                    "kv_cache_dtype": kv_cache_dtype,
                    "gpu_memory_utilization": gpu_memory_utilization,
                    "gpu_memory_budget_gb": gpu_memory_budget_gb,
                    "extra_args": list(extra_args),
                },
                "saved": True,
            }
        )
        return
    click.echo(f"Saved profile for {model_id} to {path}")


@cli.command("up")
@click.argument("model_id")
@click.option("--json-output", is_flag=True, help="Emit machine-readable JSON.")
@click.option("--port", type=int, default=None)
@click.option("--max-model-len", type=int, default=None)
@click.option("--tensor-parallel-size", type=int, default=None)
@click.option("--kv-cache-dtype", type=str, default=None)
@click.option("--gpu-memory-utilization", type=float, default=None)
@click.option("--gpu-memory-budget-gb", type=float, default=None)
@click.option("--extra-arg", "extra_args", multiple=True)
@click.option("--keep-alive/--ephemeral", default=False)
@click.option("--dry-run/--launch", default=True)
@click.pass_obj
def up(
    manager: EngineStateManager,
    model_id: str,
    json_output: bool,
    port: int | None,
    max_model_len: int | None,
    tensor_parallel_size: int | None,
    kv_cache_dtype: str | None,
    gpu_memory_utilization: float | None,
    gpu_memory_budget_gb: float | None,
    extra_args: tuple[str, ...],
    keep_alive: bool,
    dry_run: bool,
) -> None:
    result = manager.launch_model(
        model_id,
        preferred_port=port,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        kv_cache_dtype=kv_cache_dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        gpu_memory_budget_gb=gpu_memory_budget_gb,
        extra_args=list(extra_args),
        keep_alive=keep_alive,
        dry_run=dry_run,
    )
    if json_output:
        _echo_json(result.model_dump(mode="json"))
        return
    click.echo(
        f"{result.status}: {result.model_id} => {result.container_name} port={result.host_port}"
    )


@cli.command("plan")
@click.argument("model_id")
@click.option("--json-output", is_flag=True, help="Emit machine-readable JSON.")
@click.option("--port", type=int, default=None)
@click.option("--max-model-len", type=int, default=None)
@click.option("--tensor-parallel-size", type=int, default=None)
@click.option("--kv-cache-dtype", type=str, default=None)
@click.option("--gpu-memory-utilization", type=float, default=None)
@click.option("--gpu-memory-budget-gb", type=float, default=None)
@click.option("--extra-arg", "extra_args", multiple=True)
@click.pass_obj
def plan(
    manager: EngineStateManager,
    model_id: str,
    json_output: bool,
    port: int | None,
    max_model_len: int | None,
    tensor_parallel_size: int | None,
    kv_cache_dtype: str | None,
    gpu_memory_utilization: float | None,
    gpu_memory_budget_gb: float | None,
    extra_args: tuple[str, ...],
) -> None:
    report = manager.preflight_launch(
        model_id,
        preferred_port=port,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        kv_cache_dtype=kv_cache_dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        gpu_memory_budget_gb=gpu_memory_budget_gb,
        extra_args=list(extra_args),
    )
    if json_output:
        _echo_json(report.model_dump(mode="json"))
        return
    click.echo(f"allowed={report.allowed}")
    click.echo(f"port={report.requested_port}")
    click.echo(f"estimated_model_memory_gb={report.estimated_model_memory_gb}")
    click.echo(f"active_reserved_memory_gb={report.active_reserved_memory_gb}")
    click.echo(f"total_reserved_memory_gb={report.total_reserved_memory_gb}")
    if report.reason:
        click.echo(f"reason={report.reason}")


@cli.command("stop")
@click.argument("model_id")
@click.option("--json-output", is_flag=True, help="Emit machine-readable JSON.")
@click.pass_obj
def stop(manager: EngineStateManager, model_id: str, json_output: bool) -> None:
    stopped = manager.stop_model(model_id)
    if not stopped:
        raise click.ClickException(f"No managed container found for {model_id}")
    if json_output:
        _echo_json({"model_id": model_id, "stopped": True})
        return
    click.echo(f"Stopped {model_id}")


@cli.command("cleanup")
@click.option("--json-output", is_flag=True, help="Emit machine-readable JSON.")
@click.pass_obj
def cleanup(manager: EngineStateManager, json_output: bool) -> None:
    count = manager.cleanup()
    if json_output:
        _echo_json({"stopped_count": count})
        return
    click.echo(f"Stopped {count} managed container(s)")


@cli.command("ps")
@click.pass_obj
def ps(manager: EngineStateManager) -> None:
    for container in manager.active_containers():
        click.echo(f"{container.name}\t{container.status}\t{container.host_port}\t{container.model_id}")


@cli.command("logs")
@click.argument("container_name")
@click.option("--tail", type=int, default=50)
@click.pass_obj
def logs(manager: EngineStateManager, container_name: str, tail: int) -> None:
    for line in manager.runtime.stream_logs(container_name, tail=tail):
        click.echo(line)


@cli.command("stats")
@click.option("--json-output", is_flag=True, help="Emit machine-readable JSON.")
@click.option("--watch", is_flag=True, help="Refresh stats continuously.")
@click.option("--interval", type=float, default=2.0, show_default=True)
@click.option("--samples", type=int, default=None, help="Stop after N snapshots when watching.")
@click.pass_obj
def stats(
    manager: EngineStateManager,
    json_output: bool,
    watch: bool,
    interval: float,
    samples: int | None,
) -> None:
    if json_output and watch and samples is None:
        raise click.ClickException("--json-output with --watch requires --samples")

    if not watch:
        if json_output:
            _echo_json(
                [item.model_dump(mode="json") for item in manager.runtime.list_managed_stats()]
            )
            return
        _emit_stats_snapshot(manager)
        return

    if interval < 0:
        raise click.ClickException("--interval must be non-negative")
    if samples is not None and samples <= 0:
        raise click.ClickException("--samples must be greater than zero")

    emitted = 0
    while samples is None or emitted < samples:
        emitted += 1
        if json_output:
            _echo_json(
                {
                    "sample": emitted,
                    "stats": [
                        item.model_dump(mode="json")
                        for item in manager.runtime.list_managed_stats()
                    ],
                }
            )
        else:
            _emit_stats_snapshot(manager, sample_index=emitted)
        if samples is not None and emitted >= samples:
            break
        time.sleep(interval)


@cli.command("web")
@click.option("--host", type=str, default="127.0.0.1", show_default=True)
@click.option("--port", type=int, default=8080, show_default=True)
@click.pass_obj
def web(manager: EngineStateManager, host: str, port: int) -> None:
    run_web(manager, host=host, port=port)


def main() -> int:
    cli.main(prog_name="overdrive", standalone_mode=False)
    return 0