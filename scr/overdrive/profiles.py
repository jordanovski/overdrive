"""Load and save model profile overrides."""

from __future__ import annotations

from pathlib import Path

import yaml

from overdrive.models import ModelProfile, OverdriveProfiles
from overdrive.paths import profiles_path as default_profiles_path


def load_profiles(path: Path | None = None) -> OverdriveProfiles:
    target = path or default_profiles_path()
    if not target.exists():
        return OverdriveProfiles()

    data = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    return OverdriveProfiles.model_validate(data)


def save_profiles(profiles: OverdriveProfiles, path: Path | None = None) -> Path:
    target = path or default_profiles_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(profiles.model_dump(mode="python"), sort_keys=True),
        encoding="utf-8",
    )
    return target


def upsert_profile(model_id: str, profile: ModelProfile, path: Path | None = None) -> Path:
    profiles = load_profiles(path)
    profiles.models[model_id] = profile
    return save_profiles(profiles, path)