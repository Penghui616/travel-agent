from copy import deepcopy
from typing import Any, Dict, List


_TOKEN_USAGE_RECORDS: List[Dict[str, Any]] = []


def _get_usage_value(usage: Any, key: str) -> int:
    if usage is None:
        return 0
    if isinstance(usage, dict):
        value = usage.get(key, 0)
    else:
        value = getattr(usage, key, 0)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def record_token_usage(stage: str, response: Any, model: str = "") -> None:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")

    prompt_tokens = _get_usage_value(usage, "prompt_tokens")
    completion_tokens = _get_usage_value(usage, "completion_tokens")
    total_tokens = _get_usage_value(usage, "total_tokens")
    if not total_tokens:
        total_tokens = prompt_tokens + completion_tokens

    _TOKEN_USAGE_RECORDS.append(
        {
            "stage": stage,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
    )


def reset_token_usage() -> None:
    _TOKEN_USAGE_RECORDS.clear()


def get_token_usage_records() -> List[Dict[str, Any]]:
    return deepcopy(_TOKEN_USAGE_RECORDS)


def summarize_token_usage(records: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    source_records = records if records is not None else _TOKEN_USAGE_RECORDS
    return {
        "prompt_tokens": sum(record.get("prompt_tokens", 0) for record in source_records),
        "completion_tokens": sum(record.get("completion_tokens", 0) for record in source_records),
        "total_tokens": sum(record.get("total_tokens", 0) for record in source_records),
        "call_count": len(source_records),
    }
