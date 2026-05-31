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
WEIGHT_GLOBS = (
    "*.safetensors",
    "*.gguf",
    "pytorch_model*.bin",
    "model*.bin",
)
METADATA_MARKERS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
)


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


def _is_model_like_directory(directory: Path) -> bool:
    # Fast checks first.
    for marker in METADATA_MARKERS:
        if (directory / marker).exists():
            return True
    for pattern in WEIGHT_GLOBS:
        if any(directory.glob(pattern)):
            return True
    # Some exports keep files one level down from the top-level model folder.
    for child in directory.iterdir():
        if not child.is_dir():
            continue
        for marker in METADATA_MARKERS:
            if (child / marker).exists():
                return True
        for pattern in WEIGHT_GLOBS:
            if any(child.glob(pattern)):
                return True
    return False


def _discover_fallback_model_roots(hub_root: Path, config_paths: list[Path]) -> list[Path]:
    roots_with_config = {path.parent for path in config_paths}
    fallback: list[Path] = []
    for child in sorted(path for path in hub_root.iterdir() if path.is_dir()):
        if child in roots_with_config:
            continue
        if _is_model_like_directory(child):
            fallback.append(child)
    return fallback


def model_cache_diagnostics(
    hub_root: Path,
    candidate_paths: list[Path] | None = None,
    fallback_roots: list[Path] | None = None,
) -> dict[str, object]:
    candidates = candidate_paths if candidate_paths is not None else _discover_config_paths(hub_root)
    fallback = (
        fallback_roots
        if fallback_roots is not None
        else (_discover_fallback_model_roots(hub_root, candidates) if hub_root.exists() else [])
    )
    fallback_set = {path.resolve() for path in fallback}
    top_level: list[dict[str, object]] = []
    if hub_root.exists():
        for child in sorted(path for path in hub_root.iterdir() if path.is_dir()):
            child_candidates = [path for path in candidates if path.is_relative_to(child)]
            heuristic = child.resolve() in fallback_set
            top_level.append(
                {
                    "name": child.name,
                    "path": str(child),
                    "config_count": len(child_candidates),
                    "sample_configs": [str(path) for path in child_candidates[:3]],
                    "heuristic_model_like": heuristic,
                }
            )
    missing = [
        item["name"]
        for item in top_level
        if int(item["config_count"]) == 0 and not bool(item["heuristic_model_like"])
    ]
    return {
        "hub_root": str(hub_root),
        "exists": hub_root.exists(),
        "candidate_count": len(candidates),
        "candidate_paths": [str(path) for path in candidates[:25]],
        "fallback_root_count": len(fallback),
        "fallback_roots": [str(path) for path in fallback],
        "top_level_directory_count": len(top_level),
        "top_level": top_level,
        "missing_config_directories": missing,
    }


def scan_model_cache(hub_root: Path, profiles_path: Path | None = None) -> list[ModelMetadata]:
    if not hub_root.exists():
        LOGGER.warning("Hub root does not exist: %s", hub_root)
        return []

    LOGGER.info("Scanning model cache under %s", hub_root)
    profiles = load_profiles(profiles_path)
    discovered: list[ModelMetadata] = []
    candidate_paths = _discover_config_paths(hub_root)
    fallback_roots = _discover_fallback_model_roots(hub_root, candidate_paths)
    LOGGER.info("Found %d config.json candidate(s) under %s", len(candidate_paths), hub_root)
    diagnostics = model_cache_diagnostics(hub_root, candidate_paths, fallback_roots)
    if fallback_roots:
        LOGGER.info(
            "Discovered %d model-like directories without config.json under %s: %s",
            len(fallback_roots),
            hub_root,
            ", ".join(str(path) for path in fallback_roots),
        )
    missing_dirs = diagnostics["missing_config_directories"]
    if missing_dirs:
        LOGGER.warning(
            "Top-level directories with no config.json under %s: %s",
            hub_root,
            ", ".join(str(item) for item in missing_dirs),
        )
    candidate_preview = diagnostics["candidate_paths"]
    if candidate_preview:
        LOGGER.info("Config candidates under %s: %s", hub_root, ", ".join(candidate_preview))

    config_seen_snapshots: set[Path] = set()

    for config_path in candidate_paths:
        LOGGER.debug("Inspecting config candidate %s", config_path)
        snapshot_path = config_path.parent
        config_seen_snapshots.add(snapshot_path.resolve())
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

    for snapshot_path in fallback_roots:
        if snapshot_path.resolve() in config_seen_snapshots:
            continue
        model_id = _infer_model_id(snapshot_path, hub_root)
        model_name = model_id.split("/")[-1]
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
                architecture="unknown",
                model_type="unknown",
                parameter_size_billions=_parse_parameter_size(model_name),
                dtype="unknown",
                snapshot_path=snapshot_path,
                config_path=snapshot_path / "config.json",
                config_data={},
                profile=profile,
            )
        )
        LOGGER.info(
            "Discovered model %s at %s via artifact fallback (no config.json).",
            model_id,
            snapshot_path,
        )

    if discovered:
        LOGGER.info("Discovered %d model(s) under %s", len(discovered), hub_root)
    else:
        LOGGER.warning("No models discovered under %s", hub_root)

    return discovered