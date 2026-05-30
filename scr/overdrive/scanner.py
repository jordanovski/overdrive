"""Model cache scanning and metadata extraction."""

from __future__ import annotations

import json
import re
from pathlib import Path

from overdrive.models import ModelMetadata, ModelProfile
from overdrive.profiles import load_profiles

SIZE_PATTERN = re.compile(r"(?P<size>\d+(?:\.\d+)?)\s*(?P<suffix>[bBmM])")


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


def _discover_config_paths(hub_root: Path) -> list[Path]:
    config_paths: list[Path] = []

    root_config = hub_root / "config.json"
    if root_config.exists():
        config_paths.append(root_config)

    # Hugging Face cache layout: models--org--repo/snapshots/<revision>/config.json
    config_paths.extend(sorted(hub_root.glob("**/snapshots/*/config.json")))

    # Plain model library layouts:
    # - <hub_root>/<model-dir>/config.json
    # - <hub_root>/<org>/<model-dir>/config.json
    for pattern in ("*/config.json", "*/*/config.json"):
        for config_path in sorted(hub_root.glob(pattern)):
            if config_path not in config_paths:
                config_paths.append(config_path)

    return config_paths


def scan_model_cache(hub_root: Path, profiles_path: Path | None = None) -> list[ModelMetadata]:
    if not hub_root.exists():
        return []

    profiles = load_profiles(profiles_path)
    discovered: list[ModelMetadata] = []

    for config_path in _discover_config_paths(hub_root):
        snapshot_path = config_path.parent
        config_data = _load_config(config_path)
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

    return discovered