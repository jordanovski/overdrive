from pathlib import Path
import tomllib

from overdrive import __version__


def test_version_is_defined() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    parsed = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    assert __version__ == parsed["project"]["version"]