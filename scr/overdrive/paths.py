"""Filesystem path helpers for Overdrive."""

from __future__ import annotations

import os
from pathlib import Path


def config_home() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg)
    return Path.home() / ".config"


def profiles_path() -> Path:
    return config_home() / "overdrive" / "profiles.yaml"


def default_hub_root() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"