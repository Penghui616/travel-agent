import os
from functools import lru_cache
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from utils.config import get_required_setting


def _load_chat_zhipuai():
    try:
        from langchain_community.chat_models import ChatZhipuAI
    except ImportError as exc:
        raise ImportError(
            "LangChain ZhipuAI support is missing. Install langchain-community."
        ) from exc
    return ChatZhipuAI


def _to_langchain_messages(messages: Iterable[Dict[str, str]]) -> List[BaseMessage]:
    converted: List[BaseMessage] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "system":
            converted.append(SystemMessage(content=content))
        elif role == "assistant":
            converted.append(AIMessage(content=content))
        else:
            converted.append(HumanMessage(content=content))
    return converted


def _extract_usage(message: AIMessage) -> Dict[str, int]:
    usage_metadata = getattr(message, "usage_metadata", None) or {}
    response_metadata = getattr(message, "response_metadata", None) or {}
    token_usage = response_metadata.get("token_usage", {}) if isinstance(response_metadata, dict) else {}

    prompt_tokens = (
        usage_metadata.get("input_tokens")
        or token_usage.get("prompt_tokens")
        or token_usage.get("input_tokens")
        or 0
    )
    completion_tokens = (
        usage_metadata.get("output_tokens")
        or token_usage.get("completion_tokens")
        or token_usage.get("output_tokens")
        or 0
    )
    total_tokens = (
        usage_metadata.get("total_tokens")
        or token_usage.get("total_tokens")
        or int(prompt_tokens or 0) + int(completion_tokens or 0)
    )

    return {
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "total_tokens": int(total_tokens or 0),
    }


@lru_cache(maxsize=16)
def _get_chat_model(model: str, temperature: float):
    api_key = get_required_setting("ZHIPU_API_KEY")
    os.environ["ZHIPUAI_API_KEY"] = api_key
    os.environ["ZHIPU_API_KEY"] = api_key

    ChatZhipuAI = _load_chat_zhipuai()
    try:
        return ChatZhipuAI(model=model, temperature=temperature)
    except TypeError:
        return ChatZhipuAI(model_name=model, temperature=temperature)


class _LangChainCompletions:
    def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0,
        **_: Any,
    ) -> Any:
        chat_model = _get_chat_model(model, float(temperature or 0))
        result = chat_model.invoke(_to_langchain_messages(messages))
        content = result.content if isinstance(result.content, str) else str(result.content)

        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=_extract_usage(result),
        )


class _LangChainChat:
    def __init__(self) -> None:
        self.completions = _LangChainCompletions()


class LangChainChatClient:
    def __init__(self) -> None:
        self.chat = _LangChainChat()


def get_langchain_chat_client() -> LangChainChatClient:
    return LangChainChatClient()
