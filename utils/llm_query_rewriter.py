import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from utils.config import get_setting
from utils.langchain_llm import LangChainChatClient, get_langchain_chat_client
from utils.token_usage import record_token_usage

load_dotenv()

INITIAL_REWRITE_PROMPT = """
你是一个旅行需求 Query Rewriting 助手。
请把用户原始输入改写得更清晰、更完整、更适合后续 LLM 解析。
保持原意，不要编造新的城市、天数、预算或偏好。
只输出自然语言，不要输出 JSON、解释或 markdown。
"""

FOLLOWUP_REWRITE_PROMPT = """
你是一个旅行 Agent 的 follow-up Query Rewriting 助手。
请把用户最新补充改写成清晰、完整、可独立理解的修改指令，供后续 LLM 更新旅行需求。
只输出自然语言，不要输出 JSON、解释或 markdown。
"""


def get_client() -> LangChainChatClient:
    return get_langchain_chat_client()


def get_model_name() -> str:
    return get_setting("ZHIPU_MODEL", "glm-4-flash") or "glm-4-flash"


@lru_cache(maxsize=128)
def _rewrite_with_llm_cached(system_prompt: str, payload_json: str, fallback: str) -> str:
    if not get_setting("ZHIPU_API_KEY"):
        return fallback

    try:
        model_name = get_model_name()
        response = get_client().chat.completions.create(
            model=model_name,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": payload_json},
            ],
        )
        record_token_usage("query_rewrite", response, model_name)
        content = response.choices[0].message.content.strip()
        return content or fallback
    except Exception:
        return fallback


def _rewrite_with_llm(system_prompt: str, payload: Dict[str, Any], fallback: str) -> str:
    return _rewrite_with_llm_cached(
        system_prompt,
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        fallback,
    )


def rewrite_initial_query(user_input: str) -> str:
    fallback = user_input.strip()
    payload = {"user_input": fallback}
    return _rewrite_with_llm(INITIAL_REWRITE_PROMPT, payload, fallback)


def rewrite_followup_query(
    user_message: str,
    parsed_request: Optional[Dict[str, Any]] = None,
    current_itinerary: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    fallback = user_message.strip()
    payload = {
        "parsed_request": parsed_request or {},
        "current_itinerary": current_itinerary or {},
        "conversation_history": (conversation_history or [])[-8:],
        "user_message": fallback,
    }
    return _rewrite_with_llm(FOLLOWUP_REWRITE_PROMPT, payload, fallback)
