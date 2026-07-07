import copy
import json
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from utils.config import get_setting
from utils.langchain_llm import LangChainChatClient, get_langchain_chat_client
from utils.token_usage import record_token_usage

load_dotenv()

UPDATE_PROMPT = """
你是一个旅游需求更新助手。

你的任务是根据：
1. 当前已经解析好的旅游需求 JSON
2. 当前已有的行程 JSON（可能为空）
3. 最近的对话历史
4. 用户刚刚提出的新要求

输出一份“更新后的旅游需求 JSON”。

要求：
- 只输出合法 JSON，不要输出解释，不要输出 markdown。
- 保留原来没有被用户修改的字段。
- 如果用户提出了新的偏好、禁忌、节奏要求、想去/不想去的内容，优先写入 `special_requirements`。
- 如果用户明确修改了天数、预算、出行人群、交通偏好、城市、偏好类型等字段，就同步更新对应字段。
- `preferences` 只能保留与旅行偏好相关的短标签列表。
- 如果用户只是补充约束，不要随意覆盖已有信息。

输出字段必须严格保持下面这个结构：
{
  "city": "字符串",
  "start_date": "YYYY-MM-DD 或空字符串",
  "days": 0,
  "budget": "低/中/高",
  "travel_group": "独自旅行/情侣/朋友/家庭亲子/带老人",
  "transport_preference": "步行优先/公共交通优先/打车优先/都可以",
  "preferences": [],
  "special_requirements": "字符串"
}
"""


QA_PROMPT = """
你是一个旅游助手，正在基于当前已有的旅行方案继续和用户对话。

你会收到：
1. 当前解析后的旅行需求
2. 当前执行计划
3. 当前工具结果
4. 当前最终行程
5. 最近对话历史
6. 用户最新问题

请直接回答用户问题：
- 语气自然、简洁、友好
- 优先根据已有上下文回答，不要编造上下文里没有的信息
- 如果用户的问题本质上是在要求“修改行程”，明确提醒他可以继续描述想怎么改
- 不要输出 JSON，不要输出 markdown 代码块
"""


PREFERENCE_KEYWORDS = {
    "美食": ["美食", "好吃", "火锅", "小吃", "吃", "餐厅", "咖啡"],
    "夜景": ["夜景", "夜游", "看夜景", "夜晚"],
    "历史文化": ["历史", "文化", "古镇", "老街", "人文"],
    "购物": ["购物", "逛街", "商场", "商圈", "买东西"],
    "拍照": ["拍照", "出片", "摄影", "打卡"],
    "亲子": ["亲子", "孩子", "小朋友"],
    "博物馆": ["博物馆", "展馆", "美术馆"],
    "自然风景": ["自然", "公园", "江边", "爬山", "风景"],
    "citywalk": ["citywalk", "散步", "步行", "漫步", "压马路"],
}


DAY_FOCUS_KEYWORDS = {
    "美食": ["美食", "逛吃", "吃吃喝喝", "火锅", "小吃"],
    "夜景": ["夜景", "夜游"],
    "购物": ["购物", "逛街", "商圈"],
    "拍照": ["拍照", "出片", "打卡", "摄影"],
    "博物馆": ["博物馆", "展馆", "美术馆"],
    "历史文化": ["历史", "文化", "古镇", "老街", "人文"],
    "自然风景": ["自然", "公园", "江边", "爬山", "风景"],
    "citywalk": ["citywalk", "散步", "漫步", "步行"],
}


def get_client() -> LangChainChatClient:
    return get_langchain_chat_client()


def get_model_name() -> str:
    return get_setting("ZHIPU_MODEL", "glm-4-flash") or "glm-4-flash"


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


def should_update_trip(user_message: str) -> bool:
    explicit_update_keywords = (
        "改",
        "调整",
        "换",
        "不要",
        "别去",
        "增加",
        "减少",
        "删掉",
        "修改",
        "重新",
        "优化",
        "改成",
        "换成",
        "再来",
        "重做",
        "想去",
        "不想去",
        "加上",
        "去掉",
        "多一天",
        "少一天",
        "加一天",
        "减一天",
        "轻松一点",
        "紧凑一点",
        "预算低",
        "预算高",
        "住在",
        "安排成",
        "延长",
        "缩短",
        "下午出门",
        "下午开始",
        "中午后出门",
        "晚点出门",
        "不想早起",
        "每天至少",
        "每一天至少",
        "四个景点",
        "4个景点",
        "至少四个",
        "至少4个",
        "从早到晚",
        "早到晚",
        "晚上也",
        "晚上的",
        "夜景",
    )
    if any(keyword in user_message for keyword in explicit_update_keywords):
        return True

    if _extract_days_update(user_message, 1) is not None:
        return True

    return any(
        re.search(pattern, user_message) is not None
        for pattern in [
            r"再?加[一二两三四五六七八九十\d]+天",
            r"多[一二两三四五六七八九十\d]+天",
            r"延长[一二两三四五六七八九十\d]+天",
            r"减少[一二两三四五六七八九十\d]+天",
            r"缩短[一二两三四五六七八九十\d]+天",
            r"第[一二两三四五六七八九十\d]+天",
        ]
    )


def _trim_history(
    conversation_history: List[Dict[str, str]],
    limit: int = 8,
) -> List[Dict[str, str]]:
    if len(conversation_history) <= limit:
        return conversation_history
    return conversation_history[-limit:]


def _chinese_number_to_int(text: str) -> Optional[int]:
    text = (text or "").strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)

    zero = chr(0x96F6)
    one = chr(0x4E00)
    two = chr(0x4E8C)
    liang = chr(0x4E24)
    three = chr(0x4E09)
    four = chr(0x56DB)
    five = chr(0x4E94)
    six = chr(0x516D)
    seven = chr(0x4E03)
    eight = chr(0x516B)
    nine = chr(0x4E5D)
    ten = chr(0x5341)

    digit_map = {
        zero: 0,
        one: 1,
        two: 2,
        liang: 2,
        three: 3,
        four: 4,
        five: 5,
        six: 6,
        seven: 7,
        eight: 8,
        nine: 9,
        ten: 10,
    }

    if text == ten:
        return 10
    if text.startswith(ten) and len(text) == 2:
        return 10 + digit_map.get(text[1], 0)
    if text.endswith(ten) and len(text) == 2:
        return digit_map.get(text[0], 0) * 10
    if ten in text and len(text) == 3:
        left, _, right = text.partition(ten)
        return digit_map.get(left, 0) * 10 + digit_map.get(right, 0)
    if len(text) == 1 and text in digit_map:
        return digit_map[text]
    return None

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


def _prefers_afternoon_start(text: str) -> bool:
    return any(
        keyword in text
        for keyword in [
            "下午出门",
            "下午开始",
            "中午后出门",
            "晚点出门",
            "不想早起",
            "睡到自然醒",
            "下午再出去",
        ]
    )


def _extract_day_index(text: str) -> Optional[int]:
    match = re.search(r"第([一二两三四五六七八九十\d]+)天", text)
    if not match:
        return None
    return _chinese_number_to_int(match.group(1))


def _collect_itinerary_names(current_itinerary: Optional[Dict[str, Any]]) -> List[str]:
    if not current_itinerary:
        return []
    names: List[str] = []
    for day in current_itinerary.get("days", []):
        for item in day.get("items", []):
            name = (item.get("name", "") or "").strip()
            if name and name not in names:
                names.append(name)
    return names


def _extract_mentioned_places(
    user_message: str,
    current_itinerary: Optional[Dict[str, Any]],
) -> List[str]:
    candidate_names = sorted(
        _collect_itinerary_names(current_itinerary),
        key=len,
        reverse=True,
    )
    mentioned = []
    for name in candidate_names:
        if name in user_message and name not in mentioned:
            mentioned.append(name)
    return mentioned


def _extract_day_focus(user_message: str) -> Optional[tuple[int, str]]:
    day_index = _extract_day_index(user_message)
    if not day_index:
        return None

    for focus, keywords in DAY_FOCUS_KEYWORDS.items():
        if any(keyword in user_message for keyword in keywords):
            return day_index, focus
    return None


def _merge_unique_str_list(existing: List[str], additions: List[str]) -> List[str]:
    merged = list(existing or [])
    for item in additions:
        if item and item not in merged:
            merged.append(item)
    return merged


def _extract_days_update(user_message: str, current_days: int) -> Optional[int]:
    absolute_patterns = [
        r"改成([一二两三四五六七八九十\d]+)天",
        r"变成([一二两三四五六七八九十\d]+)天",
        r"调整为([一二两三四五六七八九十\d]+)天",
        r"安排([一二两三四五六七八九十\d]+)天",
        r"玩([一二两三四五六七八九十\d]+)天",
        r"([一二两三四五六七八九十\d]+)天就够",
    ]
    for pattern in absolute_patterns:
        match = re.search(pattern, user_message)
        if match:
            value = _chinese_number_to_int(match.group(1))
            if value and value > 0:
                return value

    delta_patterns = [
        (r"再?加([一二两三四五六七八九十\d]+)天", 1),
        (r"多([一二两三四五六七八九十\d]+)天", 1),
        (r"增加([一二两三四五六七八九十\d]+)天", 1),
        (r"延长([一二两三四五六七八九十\d]+)天", 1),
        (r"少([一二两三四五六七八九十\d]+)天", -1),
        (r"减([一二两三四五六七八九十\d]+)天", -1),
        (r"减少([一二两三四五六七八九十\d]+)天", -1),
        (r"缩短([一二两三四五六七八九十\d]+)天", -1),
    ]
    for pattern, direction in delta_patterns:
        match = re.search(pattern, user_message)
        if match:
            value = _chinese_number_to_int(match.group(1))
            if value:
                return max(1, current_days + direction * value)

    if "加一天" in user_message or "多一天" in user_message:
        return max(1, current_days + 1)
    if "减一天" in user_message or "少一天" in user_message:
        return max(1, current_days - 1)
    return None


def _apply_deterministic_updates(
    base_request: Dict[str, Any],
    user_message: str,
    reference_days: Optional[int] = None,
    current_itinerary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    updated = copy.deepcopy(base_request)
    current_days = int(reference_days if reference_days is not None else (updated.get("days", 0) or 0))
    constraints = copy.deepcopy(updated.get("hard_constraints", {}) or {})
    constraints.setdefault("avoid_places_global", [])
    constraints.setdefault("avoid_places_by_day", {})
    constraints.setdefault("day_focus", {})

    new_days = _extract_days_update(user_message, current_days)
    if new_days is not None:
        updated["days"] = new_days

    if any(keyword in user_message for keyword in ["预算低", "便宜点", "省钱", "预算少", "穷游"]):
        updated["budget"] = "低"
    elif any(keyword in user_message for keyword in ["预算高", "住好点", "贵一点", "高端"]):
        updated["budget"] = "高"
    elif any(keyword in user_message for keyword in ["预算中", "中等预算", "适中"]):
        updated["budget"] = "中"

    if any(keyword in user_message for keyword in ["地铁", "公共交通", "公交"]):
        updated["transport_preference"] = "公共交通优先"
    elif any(keyword in user_message for keyword in ["打车", "出租车", "网约车"]):
        updated["transport_preference"] = "打车优先"
    elif any(keyword in user_message for keyword in ["步行", "走路"]):
        updated["transport_preference"] = "步行优先"

    preferences = list(updated.get("preferences", []) or [])
    for preference, keywords in PREFERENCE_KEYWORDS.items():
        if preference in preferences:
            continue
        if any(keyword in user_message for keyword in keywords):
            preferences.append(preference)
    updated["preferences"] = preferences

    if any(keyword in user_message for keyword in ["轻松一点", "别太赶", "慢一点"]):
        updated["special_requirements"] = _append_special_requirement(
            updated.get("special_requirements", ""),
            "行程节奏希望更轻松，不要太赶。",
        )
    elif any(keyword in user_message for keyword in ["紧凑一点", "安排满一点"]):
        updated["special_requirements"] = _append_special_requirement(
            updated.get("special_requirements", ""),
            "行程可以更紧凑一些。",
        )

    if _prefers_afternoon_start(user_message):
        updated["special_requirements"] = _append_special_requirement(
            updated.get("special_requirements", ""),
            "用户习惯下午出门，尽量把每天第一段行程安排在 12:00 以后。",
        )

    if any(
        keyword in user_message
        for keyword in [
            "每天至少",
            "每一天至少",
            "四个景点",
            "4个景点",
            "至少四个",
            "至少4个",
            "从早到晚",
            "早到晚",
            "晚上也",
            "晚上的",
        ]
    ):
        updated["special_requirements"] = _append_special_requirement(
            updated.get("special_requirements", ""),
            "用户要求每天至少 4 个游玩景点，并从早到晚覆盖上午、下午和晚上，晚上也要安排夜景或夜游点。",
        )

    if new_days is not None:
        updated["special_requirements"] = _append_special_requirement(
            updated.get("special_requirements", ""),
            f"用户最新要求把行程调整为 {updated['days']} 天。",
        )

    mentioned_places = _extract_mentioned_places(user_message, current_itinerary)
    if mentioned_places:
        day_index = _extract_day_index(user_message)
        if any(keyword in user_message for keyword in ["不要", "别去", "去掉", "删掉", "删去", "移除"]):
            if day_index:
                day_key = str(day_index)
                existing = constraints["avoid_places_by_day"].get(day_key, [])
                constraints["avoid_places_by_day"][day_key] = _merge_unique_str_list(existing, mentioned_places)
                updated["special_requirements"] = _append_special_requirement(
                    updated.get("special_requirements", ""),
                    f"第{day_index}天不要安排：" + "、".join(mentioned_places),
                )
            else:
                constraints["avoid_places_global"] = _merge_unique_str_list(
                    constraints["avoid_places_global"],
                    mentioned_places,
                )
                updated["special_requirements"] = _append_special_requirement(
                    updated.get("special_requirements", ""),
                    "不要再安排：" + "、".join(mentioned_places),
                )

    day_focus = _extract_day_focus(user_message)
    if day_focus and any(keyword in user_message for keyword in ["为主", "改成", "调整成", "主要", "重点"]):
        day_index, focus = day_focus
        constraints["day_focus"][str(day_index)] = focus
        updated["special_requirements"] = _append_special_requirement(
            updated.get("special_requirements", ""),
            f"第{day_index}天改为以{focus}为主。",
        )

    updated["hard_constraints"] = constraints

    return updated


def update_parsed_request_with_llm(
    current_request: Dict[str, Any],
    user_message: str,
    conversation_history: List[Dict[str, str]],
    current_itinerary: Optional[Dict[str, Any]] = None,
    rewritten_user_message: Optional[str] = None,
) -> Dict[str, Any]:
    merged_request = copy.deepcopy(current_request)
    reference_days = int(current_request.get("days", 0) or 0)
    effective_user_message = (rewritten_user_message or user_message).strip()

    if get_setting("ZHIPU_API_KEY"):
        payload = {
            "current_request": current_request,
            "current_itinerary": current_itinerary or {},
            "conversation_history": _trim_history(conversation_history),
            "user_message": effective_user_message,
        }

        try:
            model_name = get_model_name()
            response = get_client().chat.completions.create(
                model=model_name,
                temperature=0,
                messages=[
                    {"role": "system", "content": UPDATE_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
            )
            record_token_usage("update_request", response, model_name)
            content = response.choices[0].message.content
            llm_request = extract_json_from_text(content)
            merged_request.update(llm_request)
        except Exception:
            pass

    merged_request = _apply_deterministic_updates(
        merged_request,
        user_message,
        reference_days=reference_days,
        current_itinerary=current_itinerary,
    )
    return merged_request


def answer_followup_with_llm(
    parsed_request: Dict[str, Any],
    execution_plan: Dict[str, Any],
    tool_results: Dict[str, Any],
    final_itinerary: Dict[str, Any],
    conversation_history: List[Dict[str, str]],
    user_message: str,
    rewritten_user_message: Optional[str] = None,
) -> str:
    effective_user_message = (rewritten_user_message or user_message).strip()
    payload = {
        "parsed_request": parsed_request,
        "execution_plan": execution_plan,
        "tool_results": tool_results,
        "final_itinerary": final_itinerary,
        "conversation_history": _trim_history(conversation_history),
        "user_message": effective_user_message,
    }

    model_name = get_model_name()
    response = get_client().chat.completions.create(
        model=model_name,
        temperature=0.5,
        messages=[
            {"role": "system", "content": QA_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    record_token_usage("followup_answer", response, model_name)

    return response.choices[0].message.content.strip()
