"""Thin wrapper around the Hugging Face CLI."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


class HfCliError(RuntimeError):
    """Raised when the Hugging Face CLI is unavailable or returns an error."""


def search_models(
    query: str,
    *,
    limit: int = 10,
    author: str | None = None,
    sort: str | None = None,
    filters: list[str] | None = None,
    num_parameters: str | None = None,
    token: str | None = None,
) -> list[dict[str, object]]:
    command = [
        *_resolve_hf_command(),
        "models",
        "list",
        "--search",
        query,
        "--limit",
        str(limit),
        "--format",
        "json",
    ]
    if author:
        command.extend(["--author", author])
    if sort:
        command.extend(["--sort", sort])
    if num_parameters:
        command.extend(["--num-parameters", num_parameters])
    for item in filters or []:
        command.extend(["--filter", item])
    if token:
        command.extend(["--token", token])

    completed = _run_hf_command(command)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise HfCliError("hf models list returned invalid JSON output.") from exc
    if not isinstance(payload, list):
        raise HfCliError("hf models list did not return a JSON array.")
    return payload


def download_model(
    model_id: str,
    *,
    local_dir: Path | None = None,
    cache_dir: Path | None = None,
    revision: str | None = None,
    includes: list[str] | None = None,
    excludes: list[str] | None = None,
    token: str | None = None,
    max_workers: int | None = None,
    force_download: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    command = [*_resolve_hf_command(), "download", model_id]
    if revision:
        command.extend(["--revision", revision])
    if local_dir:
        command.extend(["--local-dir", str(local_dir)])
    if cache_dir:
        command.extend(["--cache-dir", str(cache_dir)])
    for pattern in includes or []:
        command.extend(["--include", pattern])
    for pattern in excludes or []:
        command.extend(["--exclude", pattern])
    if token:
        command.extend(["--token", token])
    if max_workers is not None:
        command.extend(["--max-workers", str(max_workers)])
    if force_download:
        command.append("--force-download")
    if dry_run:
        command.append("--dry-run")
    else:
        command.append("--quiet")

    completed = _run_hf_command(command)
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    download_path = None if dry_run or not lines else lines[-1]
    return {
        "model_id": model_id,
        "local_dir": str(local_dir) if local_dir else None,
        "cache_dir": str(cache_dir) if cache_dir else None,
        "revision": revision,
        "include": includes or [],
        "exclude": excludes or [],
        "force_download": force_download,
        "dry_run": dry_run,
        "download_path": download_path,
        "stdout": completed.stdout.strip(),
    }


def _resolve_hf_command() -> list[str]:
    executable = shutil.which("hf")
    if executable:
        return [executable]

    sibling = Path(sys.executable).with_name("hf.exe" if sys.platform.startswith("win") else "hf")
    if sibling.exists():
        return [str(sibling)]

    raise HfCliError(
        "Could not find the Hugging Face CLI. "
        "Install the 'huggingface_hub' package or the standalone 'hf' CLI."
    )


def _run_hf_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise HfCliError("The Hugging Face CLI executable could not be started.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "Unknown hf CLI error."
        raise HfCliError(stderr) from exc