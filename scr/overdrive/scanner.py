"""Model cache scanning and metadata extraction."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from overdrive.models import ModelMetadata, ModelProfile
from overdrive.profiles import load_profiles

SIZE_PATTERN = re.compile(r"(?P<size>\d+(?:\.\d+)?)\s*(?P<suffix>[bBmM])")
LOGGER = logging.getLogger(__name__)


def _parse_parameter_size(model_name: str) -> float | None:
    match = SIZE_PATTERN.search(model_name)
    if not match:
        return None
    value = float(match.group("size"))
    suffix = match.group("suffix").lower()
    if suffix == "m":
        return round(value / 1000, 3)
    return value


def _infer_model_id(snapshot_path: Path, hub_root: Path) -> str:
    relative = snapshot_path.relative_to(hub_root)
    storage_root = relative.parts[0]
    if storage_root.startswith("models--"):
        return storage_root.removeprefix("models--").replace("--", "/")
    if relative == Path("."):
        return snapshot_path.name
    return relative.as_posix()


def _load_config(config_path: Path) -> dict[str, object]:
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _looks_like_model_config(config_data: dict[str, object]) -> bool:
    return any(
        key in config_data
        for key in ("architectures", "model_type", "torch_dtype", "text_config")
    )


def _discover_config_paths(hub_root: Path) -> list[Path]:
    config_paths: list[Path] = []
    seen: set[Path] = set()

    def add_config_path(config_path: Path) -> None:
        if config_path in seen:
            return
        seen.add(config_path)
        config_paths.append(config_path)

    root_config = hub_root / "config.json"
    if root_config.exists():
        add_config_path(root_config)

    for config_path in sorted(hub_root.glob("**/config.json")):
        add_config_path(config_path)

    return config_paths


def scan_model_cache(hub_root: Path, profiles_path: Path | None = None) -> list[ModelMetadata]:
    if not hub_root.exists():
        LOGGER.warning("Hub root does not exist: %s", hub_root)
        return []

    LOGGER.info("Scanning model cache under %s", hub_root)
    profiles = load_profiles(profiles_path)
    discovered: list[ModelMetadata] = []
    candidate_paths = _discover_config_paths(hub_root)
    LOGGER.info("Found %d config.json candidate(s) under %s", len(candidate_paths), hub_root)

    for config_path in candidate_paths:
        LOGGER.debug("Inspecting config candidate %s", config_path)
        snapshot_path = config_path.parent
        try:
            config_data = _load_config(config_path)
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("Skipping unreadable config %s: %s", config_path, exc)
            continue
        if not _looks_like_model_config(config_data):
            LOGGER.debug("Skipping non-model config %s", config_path)
            continue
        model_id = _infer_model_id(snapshot_path, hub_root)
        model_name = model_id.split("/")[-1]
        architecture = next(iter(config_data.get("architectures", [])), "unknown")
        model_type = str(config_data.get("model_type", architecture)).lower()
        dtype = str(config_data.get("torch_dtype", "unknown"))
        profile = (
            profiles.models.get(model_id)
            or profiles.models.get(model_name)
            or profiles.models.get("default")
            or ModelProfile()
        )
        discovered.append(
            ModelMetadata(
                model_id=model_id,
                model_name=model_name,
                architecture=architecture,
                model_type=model_type,
                parameter_size_billions=_parse_parameter_size(model_name),
                dtype=dtype,
                snapshot_path=snapshot_path,
                config_path=config_path,
                config_data=config_data,
                profile=profile,
            )
        )
        LOGGER.debug("Discovered model %s at %s", model_id, snapshot_path)

    if discovered:
        LOGGER.info("Discovered %d model(s) under %s", len(discovered), hub_root)
    else:
        LOGGER.warning("No models discovered under %s", hub_root)

    return discovered