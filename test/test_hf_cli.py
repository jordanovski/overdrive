import subprocess
from pathlib import Path

import pytest
from overdrive.hf_cli import HfCliError, download_model, search_models


def test_search_models_parses_json(monkeypatch) -> None:
    monkeypatch.setattr("overdrive.hf_cli.shutil.which", lambda name: "hf")

    def fake_run(command, check, capture_output, text):
        assert command[:3] == ["hf", "models", "list"]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout='[{"id":"Qwen/Qwen3-8B","downloads":123,"likes":45}]',
            stderr="",
        )

    monkeypatch.setattr("overdrive.hf_cli.subprocess.run", fake_run)

    payload = search_models("qwen", limit=5)

    assert payload[0]["id"] == "Qwen/Qwen3-8B"


def test_download_model_returns_last_output_line(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("overdrive.hf_cli.shutil.which", lambda name: "hf")

    def fake_run(command, check, capture_output, text):
        assert command[:2] == ["hf", "download"]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="warning line\nE:/models/Qwen/Qwen3-8B\n",
            stderr="",
        )

    monkeypatch.setattr("overdrive.hf_cli.subprocess.run", fake_run)

    payload = download_model("Qwen/Qwen3-8B", local_dir=tmp_path)

    assert payload["download_path"] == "E:/models/Qwen/Qwen3-8B"


def test_search_models_raises_on_hf_error(monkeypatch) -> None:
    monkeypatch.setattr("overdrive.hf_cli.shutil.which", lambda name: "hf")

    def fake_run(command, check, capture_output, text):
        raise subprocess.CalledProcessError(1, command, stderr="hf failed")

    monkeypatch.setattr("overdrive.hf_cli.subprocess.run", fake_run)

    with pytest.raises(HfCliError, match="hf failed"):
        search_models("qwen")