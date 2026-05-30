from pathlib import Path

from overdrive.models import ModelProfile
from overdrive.profiles import load_profiles, upsert_profile


def test_profile_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "profiles.yaml"

    upsert_profile(
        "org/model-7b",
        ModelProfile(preferred_port=8002, max_model_len=32768, extra_args=["--trust-remote-code"]),
        target,
    )

    profiles = load_profiles(target)
    assert profiles.models["org/model-7b"].preferred_port == 8002
    assert profiles.models["org/model-7b"].max_model_len == 32768