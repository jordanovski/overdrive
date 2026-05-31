import json
import logging
from pathlib import Path

from overdrive.models import ModelProfile
from overdrive.profiles import upsert_profile
from overdrive.scanner import scan_model_cache


def test_scan_model_cache_extracts_metadata_and_profiles(tmp_path: Path) -> None:
    hub_root = tmp_path / "hub"
    snapshot = hub_root / "models--org--qwen2.5-7b" / "snapshots" / "123abc"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen2ForCausalLM"],
                "model_type": "qwen2",
                "torch_dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )
    profiles_path = tmp_path / "profiles.yaml"
    upsert_profile("org/qwen2.5-7b", ModelProfile(preferred_port=8009), profiles_path)

    models = scan_model_cache(hub_root, profiles_path)

    assert len(models) == 1
    assert models[0].model_id == "org/qwen2.5-7b"
    assert models[0].architecture == "Qwen2ForCausalLM"
    assert models[0].profile.preferred_port == 8009
    assert models[0].parameter_size_billions == 7.0


def test_scan_model_cache_supports_direct_model_directories(tmp_path: Path) -> None:
    hub_root = tmp_path / "models"
    model_dir = hub_root / "qwen-35b-moe"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen2MoeForCausalLM"],
                "model_type": "qwen2_moe",
                "torch_dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )

    models = scan_model_cache(hub_root)

    assert len(models) == 1
    assert models[0].model_id == "qwen-35b-moe"
    assert models[0].snapshot_path == model_dir
    assert models[0].architecture == "Qwen2MoeForCausalLM"
    assert models[0].parameter_size_billions == 35.0


def test_scan_model_cache_supports_nested_org_model_directories(tmp_path: Path) -> None:
    hub_root = tmp_path / "models"
    model_dir = hub_root / "google" / "gemma-4-31B-it"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Gemma3ForConditionalGeneration"],
                "model_type": "gemma3",
                "torch_dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )

    models = scan_model_cache(hub_root)

    assert len(models) == 1
    assert models[0].model_id == "google/gemma-4-31B-it"
    assert models[0].snapshot_path == model_dir
    assert models[0].architecture == "Gemma3ForConditionalGeneration"
    assert models[0].parameter_size_billions == 31.0


def test_scan_model_cache_supports_deeply_nested_model_directories(tmp_path: Path) -> None:
    hub_root = tmp_path / "models"
    model_dir = hub_root / "foundation" / "google" / "gemma-4-31B-it"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Gemma3ForConditionalGeneration"],
                "model_type": "gemma3",
                "torch_dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )

    models = scan_model_cache(hub_root)

    assert len(models) == 1
    assert models[0].model_id == "foundation/google/gemma-4-31B-it"
    assert models[0].snapshot_path == model_dir


def test_scan_model_cache_ignores_unrelated_recursive_config_files(tmp_path: Path) -> None:
    hub_root = tmp_path / "models"
    model_dir = hub_root / "google" / "gemma-4-31B-it"
    unrelated_dir = hub_root / "docs" / "site"
    model_dir.mkdir(parents=True)
    unrelated_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Gemma3ForConditionalGeneration"],
                "model_type": "gemma3",
                "torch_dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )
    (unrelated_dir / "config.json").write_text(
        json.dumps({"theme": "docs"}),
        encoding="utf-8",
    )

    models = scan_model_cache(hub_root)

    assert len(models) == 1
    assert models[0].model_id == "google/gemma-4-31B-it"


def test_scan_model_cache_logs_summary_and_discovery(tmp_path: Path, caplog) -> None:
    hub_root = tmp_path / "models"
    model_dir = hub_root / "google" / "gemma-4-31B-it"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Gemma3ForConditionalGeneration"],
                "model_type": "gemma3",
                "torch_dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )

    with caplog.at_level(logging.DEBUG, logger="overdrive.scanner"):
        models = scan_model_cache(hub_root)

    assert len(models) == 1
    assert "Scanning model cache under" in caplog.text
    assert "Discovered model google/gemma-4-31B-it" in caplog.text
    assert "Discovered 1 model(s) under" in caplog.text


def test_scan_model_cache_logs_missing_hub_root(tmp_path: Path, caplog) -> None:
    missing_root = tmp_path / "missing"

    with caplog.at_level(logging.WARNING, logger="overdrive.scanner"):
        models = scan_model_cache(missing_root)

    assert models == []
    assert f"Hub root does not exist: {missing_root}" in caplog.text