from types import SimpleNamespace

from overdrive.hf_catalog import HubSearchOptions, search_hub_models


class _CompatApi:
    def __init__(self, token=None):
        self.token = token

    def list_models(self, **kwargs):
        if "direction" in kwargs:
            raise TypeError("HfApi.list_models() got an unexpected keyword argument 'direction'")
        return [
            SimpleNamespace(
                id="Qwen/Qwen3.6-27B",
                author="Qwen",
                downloads=123,
                likes=12,
                pipeline_tag="text-generation",
                library_name="transformers",
                tags=["nvfp4", "vllm"],
                cardData={"tags": ["blackwell"]},
                last_modified="2026-05-31T00:00:00",
            )
        ]


def test_search_hub_models_compat_with_missing_direction(monkeypatch) -> None:
    monkeypatch.setattr("overdrive.hf_catalog.HfApi", _CompatApi)

    results = search_hub_models(
        HubSearchOptions(
            query="qwen3.6",
            quantization="nvfp4",
            dgx_ready_only=True,
            limit=10,
        )
    )

    assert len(results) == 1
    assert results[0]["id"] == "Qwen/Qwen3.6-27B"
    assert "nvfp4" in results[0]["tags"]
    assert "blackwell" in results[0]["dgx_tags"]
