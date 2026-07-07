import json
import os
from copy import deepcopy
from functools import lru_cache
from typing import Any, Dict

from dotenv import load_dotenv

from utils.config import get_setting
from utils.langchain_llm import LangChainChatClient, get_langchain_chat_client
from utils.token_usage import record_token_usage

load_dotenv()

SYSTEM_PROMPT = """
你是一个旅行需求解析助手。
请把用户的自然语言旅行需求提取成结构化 JSON。
只输出合法 JSON，不要输出解释、markdown 或代码块。

输出字段：
{
  "city": "目的地城市，没有就填空字符串",
  "start_date": "YYYY-MM-DD，没有就填空字符串",
  "days": 0,
  "budget": "低/中/高",
  "travel_group": "独自旅行/情侣/朋友/家庭亲子/带老人",
  "transport_preference": "步行优先/公共交通优先/打车优先/都可以",
  "preferences": ["美食","夜景","历史文化","购物","拍照","亲子","博物馆","自然风景","citywalk"],
  "special_requirements": "其他特殊要求"
}
"""

AFTERNOON_KEYWORDS = ["下午出门", "下午开始", "中午后出门", "晚点出门", "不想早起", "睡到自然醒"]


def get_client() -> LangChainChatClient:
    return get_langchain_chat_client()


def get_model_name() -> str:
    return get_setting("ZHIPU_MODEL", "glm-4-flash") or "glm-4-flash"


def _append_special_requirement(original: str, addition: str) -> str:
    original = (original or "").strip()
    addition = (addition or "").strip()
    if not addition:
        return original
    if not original:
        return addition
    if addition in original:
        return original
    return f"{original}；{addition}"


def _apply_deterministic_postprocess(
    parsed_request: Dict[str, Any],
    user_input: str,
) -> Dict[str, Any]:
    updated = deepcopy(parsed_request)
    if any(keyword in user_input for keyword in AFTERNOON_KEYWORDS):
        updated["special_requirements"] = _append_special_requirement(
            updated.get("special_requirements", ""),
            "用户习惯下午出门，尽量把每天第一段行程安排在 12:00 以后。",
        )
    return updated


def extract_json_from_text(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])

    raise ValueError("模型返回内容不是合法 JSON")


@lru_cache(maxsize=64)
def _parse_request_cached(user_input: str) -> Dict[str, Any]:
    model_name = get_model_name()
    response = get_client().chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
        temperature=0,
    )
    record_token_usage("parse_request", response, model_name)

    content = response.choices[0].message.content
    return extract_json_from_text(content)


def parse_travel_request_with_llm(
    user_input: str,
    original_user_input: str | None = None,
) -> Dict[str, Any]:
    parsed_request = _parse_request_cached(user_input.strip())
    return _apply_deterministic_postprocess(
        parsed_request,
        original_user_input or user_input,
    )
