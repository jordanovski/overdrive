"""Hugging Face Hub catalog search helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from huggingface_hub import HfApi


DGX_HINT_TAGS = {
    "vllm",
    "tensorrt-llm",
    "triton",
    "blackwell",
    "hopper",
    "h100",
    "h200",
    "b200",
    "fp8",
    "nvfp4",
    "compressed-tensors",
    "modelopt",
    "safetensors",
}


@dataclass(slots=True)
class HubSearchOptions:
    query: str = ""
    quantization: str | None = None
    author: str | None = None
    pipeline_tag: str | None = None
    library: str | None = None
    min_downloads: int | None = None
    sort: str = "downloads"
    limit: int = 25
    dgx_ready_only: bool = False
    token: str | None = None


def _split_csv_tokens(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [token.strip().lower() for token in raw.split(",") if token.strip()]


def _extract_card_tags(card_data: object) -> set[str]:
    if not isinstance(card_data, dict):
        return set()
    tags = card_data.get("tags")
    if not isinstance(tags, list):
        return set()
    return {
        str(tag).strip().lower()
        for tag in tags
        if isinstance(tag, str) and str(tag).strip()
    }


def _as_positive_limit(limit: int) -> int:
    if limit < 1:
        raise ValueError("limit must be greater than 0")
    return min(limit, 100)


def _list_models_compat(api: HfApi, **kwargs: Any):
    try:
        return api.list_models(**kwargs)
    except TypeError as exc:
        message = str(exc)
        # Older huggingface_hub versions may not support some newer kwargs.
        for optional_kwarg in ("direction", "cardData"):
            if optional_kwarg in kwargs and optional_kwarg in message:
                fallback = dict(kwargs)
                fallback.pop(optional_kwarg, None)
                return api.list_models(**fallback)
        raise


def search_hub_models(options: HubSearchOptions) -> list[dict[str, object]]:
    limit = _as_positive_limit(options.limit)
    query = options.query.strip()
    quant_tokens = _split_csv_tokens(options.quantization)
    api = HfApi(token=options.token)

    # Pull extra rows before local filtering so the final page remains populated.
    request_limit = min(max(limit * 4, 40), 200)
    records = _list_models_compat(
        api,
        search=query or None,
        author=options.author or None,
        pipeline_tag=options.pipeline_tag or None,
        sort=options.sort,
        direction=-1,
        limit=request_limit,
        full=True,
        cardData=True,
    )

    results: list[dict[str, object]] = []
    for model in records:
        tags = {
            str(tag).strip().lower()
            for tag in (model.tags or [])
            if isinstance(tag, str) and str(tag).strip()
        }
        tags.update(_extract_card_tags(getattr(model, "cardData", None)))
        model_id = str(getattr(model, "id", ""))
        text_search_haystack = f"{model_id.lower()} {' '.join(sorted(tags))}"

        if options.library:
            library_value = str(getattr(model, "library_name", "") or "").strip().lower()
            wanted_library = options.library.strip().lower()
            if library_value != wanted_library and wanted_library not in tags:
                continue

        downloads = int(getattr(model, "downloads", 0) or 0)
        if options.min_downloads is not None and downloads < options.min_downloads:
            continue

        if quant_tokens and not any(token in text_search_haystack for token in quant_tokens):
            continue

        dgx_tags = sorted(tag for tag in tags if tag in DGX_HINT_TAGS)
        if options.dgx_ready_only and not dgx_tags:
            continue

        results.append(
            {
                "id": model_id,
                "author": str(getattr(model, "author", "") or ""),
                "downloads": downloads,
                "likes": int(getattr(model, "likes", 0) or 0),
                "pipeline_tag": getattr(model, "pipeline_tag", None),
                "library_name": getattr(model, "library_name", None),
                "last_modified": str(getattr(model, "last_modified", "") or ""),
                "tags": sorted(tags),
                "dgx_tags": dgx_tags,
                "quantization_match": bool(quant_tokens),
            }
        )

        if len(results) >= limit:
            break

    return results
