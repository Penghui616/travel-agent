import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from utils.config import get_setting
from utils.langchain_llm import LangChainChatClient, get_langchain_chat_client
from utils.token_usage import record_token_usage

load_dotenv()

MAX_CONTEXT_ATTRACTIONS = int(get_setting("TRAVEL_AGENT_MAX_CONTEXT_ATTRACTIONS", "20") or "20")
MAX_CONTEXT_ATTRACTIONS_CAP = int(get_setting("TRAVEL_AGENT_MAX_CONTEXT_ATTRACTIONS_CAP", "40") or "40")
MAX_CONTEXT_RESTAURANTS = int(get_setting("TRAVEL_AGENT_MAX_CONTEXT_RESTAURANTS", "10") or "10")
MAX_CONTEXT_RESTAURANTS_CAP = int(get_setting("TRAVEL_AGENT_MAX_CONTEXT_RESTAURANTS_CAP", "24") or "24")
MAX_CONTEXT_HOTELS = int(get_setting("TRAVEL_AGENT_MAX_CONTEXT_HOTELS", "4") or "4")
MAX_CONTEXT_DISTRICTS = int(get_setting("TRAVEL_AGENT_MAX_CONTEXT_DISTRICTS", "5") or "5")
MAX_RAG_CHUNKS = int(get_setting("TRAVEL_AGENT_MAX_RAG_CHUNKS", "2") or "2")
MAX_RAG_CONTENT_CHARS = int(get_setting("TRAVEL_AGENT_MAX_RAG_CONTENT_CHARS", "320") or "320")
MIN_SIGHTSEEING_ITEMS_PER_DAY = 4
FAST_ITINERARY_ENABLED = str(get_setting("TRAVEL_AGENT_FAST_ITINERARY", "1") or "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}


JSON_REPAIR_PROMPT = """
你是一个 JSON 修复助手。
我会给你一段本应为 JSON 的文本，它可能缺逗号、夹杂解释文字、或有轻微格式错误。

要求：
- 只输出修复后的合法 JSON
- 保持原始字段和值语义不变
- 不要输出解释
- 不要输出 markdown
"""


ITINERARY_PROMPT = """
你是一个专业的旅游行程规划助手。

你需要根据用户需求、执行计划和精简后的工具结果，生成适合前端展示的结构化多日行程 JSON。

硬性要求，必须全部满足：
1. `days` 数组长度必须严格等于 `parsed_request.days`。
2. 每一天的 `theme` 必须不同。
3. 同一个地点的 `name` 在整个行程中只能出现一次，严禁重复景点。
4. 每一天必须至少 4 个游玩类地点/活动，最多 5 个；餐厅、酒店休整不计入 4 个游玩点。
5. 默认每一天都要从早到晚覆盖上午、下午、晚上三个时段，晚上必须安排夜景/夜游/街区散步等游玩点；但如果 `special_requirements` 明确说明用户习惯下午出门或中午后出发，则当天第一段行程要从 12:00 以后开始。
6. 如果真实景点不够，不要重复景点；可以安排附近用餐、城市漫步、自由活动、商圈闲逛、夜景散步、酒店休整等不同活动来补足。
7. 每天路线不能简单复制前一天，应尽量围绕不同片区展开。
8. 每一天尽量围绕一个片区/行政区展开，减少跨区往返；如果提供了 `day_district_plan`，请优先按该分区建议安排每天路线。
9. 优先从 `compact_tool_results` 里的景点、餐厅、酒店候选中挑选地点，并优先使用不同的 `name`。
10. 不要机械使用所有工具结果，只选择合理、顺路、适合用户偏好的地点。
11. 不要安排明显远郊地点，除非用户明确要求。
12. `theme` 和 `route_summary` 最好体现当天核心片区。
13. 输出必须是合法 JSON，不要输出 markdown，不要输出解释。
14. `summary`、`description`、`transport_to_next`、`day_tips` 都要简洁；每个 description 控制在一句话内。
15. 如果输入中提供了 `rag_context`，请把它当作本地攻略知识参考，用来优化片区节奏、避坑提醒和主题安排；地点选择仍优先依据 `compact_tool_results` 中的真实候选。

每个 day 的 items 示例：
[
  {
    "time": "09:00-10:30",
    "name": "景点A",
    "category": "景点",
    "description": "为什么安排这里，适合怎么游玩",
    "transport_to_next": "步行/地铁/打车到下一站，大约多久"
  },
  {
    "time": "11:00-12:30",
    "name": "景点B",
    "category": "景点",
    "description": "作为上午第二个游玩点，和前一站保持顺路",
    "transport_to_next": "就近前往午餐或下午行程"
  },
  {
    "time": "14:30-16:30",
    "name": "景点C",
    "category": "景点",
    "description": "下午继续安排一个核心景点，保证行程充实",
    "transport_to_next": "前往晚间活动"
  },
  {
    "time": "19:00-21:00",
    "name": "夜景散步",
    "category": "夜景",
    "description": "适合傍晚和夜间放松游玩",
    "transport_to_next": ""
  }
]

输出 JSON 格式必须严格如下：
{
  "title": "城市N日游",
  "summary": "一句话总结整体路线",
  "important_tips": ["提醒1", "提醒2"],
  "days": [
    {
      "day": 1,
      "theme": "当天主题",
      "route_summary": "当天路线概括",
      "items": [
        {
          "time": "09:00-10:30",
          "name": "地点名称",
          "category": "景点/餐饮/购物/酒店/其他",
          "description": "为什么安排这里，怎么玩",
          "transport_to_next": "到下一站的交通方式、距离或时间，没有就填空字符串"
        }
      ],
      "day_tips": ["当天建议1", "当天建议2"]
    }
  ],
  "hotel_area_suggestion": "推荐住宿区域",
  "weather_advice": "天气和穿衣建议",
  "transport_advice": "交通建议"
}
"""


CATEGORY_POOL_MAP = {
    "餐饮": ["restaurants"],
    "美食": ["restaurants"],
    "小吃": ["restaurants"],
    "咖啡": ["restaurants"],
    "酒店": ["hotels"],
    "住宿": ["hotels"],
    "民宿": ["hotels"],
    "景点": ["attractions"],
    "夜景": ["attractions"],
    "购物": ["attractions", "restaurants"],
    "商圈": ["attractions", "restaurants"],
    "博物馆": ["attractions"],
    "公园": ["attractions"],
}


AFTERNOON_TIME_TEMPLATES = {
    3: ["13:00-14:30", "15:30-17:30", "19:00-21:00"],
    4: ["13:00-14:30", "15:00-16:30", "17:30-18:30", "19:30-21:00"],
    5: ["13:00-14:00", "14:30-15:45", "16:15-17:30", "18:00-19:00", "19:30-21:00"],
    6: ["13:00-13:50", "14:10-15:00", "15:20-16:10", "16:40-17:30", "18:00-19:00", "19:30-21:00"],
}

FULL_DAY_TIME_TEMPLATES = {
    4: ["09:00-10:30", "11:00-12:30", "14:30-16:30", "19:00-21:00"],
    5: ["09:00-10:15", "10:45-12:00", "14:00-15:30", "16:00-17:30", "19:00-21:00"],
    6: ["09:00-09:50", "10:10-11:00", "11:20-12:10", "14:00-15:20", "16:00-17:30", "19:00-21:00"],
}

SIGHTSEEING_CATEGORY_KEYWORDS = [
    "景点",
    "夜景",
    "夜游",
    "购物",
    "商圈",
    "博物馆",
    "公园",
    "自然",
    "文化",
    "citywalk",
    "拍照",
]

FULL_DAY_REQUIREMENT_KEYWORDS = [
    "从早到晚",
    "早到晚",
    "早上出门",
    "上午出门",
    "上午开始",
    "早点出门",
    "全天",
    "安排满",
    "四个景点",
    "4个景点",
    "至少四个",
    "至少4个",
]


DAY_FOCUS_CATEGORY_PLAN = {
    "美食": ["餐饮", "餐饮", "景点", "夜景"],
    "夜景": ["景点", "景点", "餐饮", "夜景"],
    "购物": ["购物", "餐饮", "购物", "夜景"],
    "拍照": ["景点", "景点", "餐饮", "夜景"],
    "博物馆": ["景点", "景点", "餐饮", "夜景"],
    "历史文化": ["景点", "景点", "餐饮", "夜景"],
    "自然风景": ["景点", "景点", "餐饮", "夜景"],
    "citywalk": ["景点", "餐饮", "景点", "夜景"],
}


def get_client() -> LangChainChatClient:
    return get_langchain_chat_client()


def get_model_name() -> str:
    return get_setting("ZHIPU_MODEL", "glm-4-flash") or "glm-4-flash"


def _strip_code_fences(text: str) -> str:
    cleaned = (text or "").strip().replace("\ufeff", "")
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    return cleaned


def _extract_balanced_json_candidate(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return ""

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]

    end = text.rfind("}")
    if end != -1 and end > start:
        return text[start:end + 1]
    return ""


def _sanitize_json_text(text: str) -> str:
    sanitized = text
    sanitized = sanitized.replace("“", '"').replace("”", '"')
    sanitized = sanitized.replace("‘", "'").replace("’", "'")
    sanitized = sanitized.replace("\u00a0", " ")
    sanitized = re.sub(r",\s*([}\]])", r"\1", sanitized)
    return sanitized.strip()


def _repair_json_with_llm(text: str) -> str:
    model_name = get_model_name()
    response = get_client().chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[
            {"role": "system", "content": JSON_REPAIR_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    record_token_usage("json_repair", response, model_name)
    return response.choices[0].message.content.strip()


def extract_json_from_text(text: str, allow_llm_repair: bool = True) -> Dict[str, Any]:
    cleaned = _strip_code_fences(text)
    candidate = _extract_balanced_json_candidate(cleaned)

    trial_texts = [cleaned]
    if candidate and candidate not in trial_texts:
        trial_texts.append(candidate)

    sanitized_texts = []
    for item in trial_texts:
        sanitized = _sanitize_json_text(item)
        if sanitized and sanitized not in trial_texts and sanitized not in sanitized_texts:
            sanitized_texts.append(sanitized)

    last_error: Optional[Exception] = None
    for item in [*trial_texts, *sanitized_texts]:
        if not item:
            continue
        try:
            return json.loads(item)
        except json.JSONDecodeError as exc:
            last_error = exc

    if allow_llm_repair and get_setting("ZHIPU_API_KEY"):
        repaired = _repair_json_with_llm(candidate or cleaned)
        return extract_json_from_text(repaired, allow_llm_repair=False)

    if last_error is not None:
        raise last_error
    raise ValueError("模型返回内容不是合法 JSON")


def _normalize_name(name: str) -> str:
    return " ".join((name or "").strip().lower().split())


def _prefers_afternoon_start(parsed_request: Dict[str, Any]) -> bool:
    special_requirements = (parsed_request.get("special_requirements", "") or "").strip()
    if any(keyword in special_requirements for keyword in FULL_DAY_REQUIREMENT_KEYWORDS):
        return False
    return any(
        keyword in special_requirements
        for keyword in [
            "下午出门",
            "下午开始",
            "中午后出门",
            "晚点出门",
            "不想早起",
            "12:00 以后",
        ]
    )


def _parse_start_hour(time_range: str) -> Optional[int]:
    match = json.loads(json.dumps({"time": time_range}))["time"] if time_range is not None else ""
    if not isinstance(match, str) or "-" not in match:
        return None
    start = match.split("-", 1)[0].strip()
    if ":" not in start:
        return None
    hour = start.split(":", 1)[0]
    return int(hour) if hour.isdigit() else None


def _is_sightseeing_item(item: Dict[str, Any]) -> bool:
    category = (item or {}).get("category", "")
    name = (item or {}).get("name", "")
    text = f"{category} {name}".lower()
    return any(keyword.lower() in text for keyword in SIGHTSEEING_CATEGORY_KEYWORDS)


def _has_evening_item(items: List[Dict[str, Any]]) -> bool:
    for item in items:
        hour = _parse_start_hour(item.get("time", ""))
        category = (item.get("category", "") or "").strip()
        if hour is not None and hour >= 18:
            return True
        if any(keyword in category for keyword in ["夜景", "夜游", "晚上", "晚间"]):
            return True
    return False


def _day_time_template(item_count: int, parsed_request: Dict[str, Any]) -> Optional[List[str]]:
    capped_count = min(max(item_count, MIN_SIGHTSEEING_ITEMS_PER_DAY), 6)
    if _prefers_afternoon_start(parsed_request):
        return AFTERNOON_TIME_TEMPLATES.get(capped_count)
    return FULL_DAY_TIME_TEMPLATES.get(capped_count)


def _sightseeing_category_for_position(position: int) -> str:
    if position >= MIN_SIGHTSEEING_ITEMS_PER_DAY:
        return "夜景"
    return "景点"


def _apply_day_time_template(
    day: Dict[str, Any],
    parsed_request: Dict[str, Any],
) -> None:
    items = day.get("items", [])
    template = _day_time_template(len(items), parsed_request)
    if not template:
        return
    for item, new_time in zip(items, template):
        item["time"] = new_time


def _normalize_constraint_names(names: List[str]) -> set[str]:
    return {_normalize_name(name) for name in names if _normalize_name(name)}


def _poi_district(poi: Dict[str, Any]) -> str:
    return (
        (poi or {}).get("adname")
        or (poi or {}).get("cityname")
        or (poi or {}).get("pname")
        or ""
    ).strip()


def _extract_poi_pool(tool_results: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    pool_map = {
        "attractions": tool_results.get("attractions", {}).get("attractions", []),
        "restaurants": tool_results.get("restaurants", {}).get("restaurants", []),
        "hotels": tool_results.get("hotels", {}).get("hotels", []),
    }

    cleaned: Dict[str, List[Dict[str, Any]]] = {}
    for pool_name, pois in pool_map.items():
        seen = set()
        unique_pois = []
        for poi in pois:
            name = (poi or {}).get("name", "").strip()
            normalized = _normalize_name(name)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique_pois.append(poi)
        cleaned[pool_name] = unique_pois

    return cleaned


def _build_day_district_plan(
    parsed_request: Dict[str, Any],
    tool_results: Dict[str, Any],
) -> List[Dict[str, Any]]:
    pools = _extract_poi_pool(tool_results)
    expected_days = max(int(parsed_request.get("days", 0) or 0), 1)

    district_scores: Dict[str, int] = {}
    district_candidates: Dict[str, Dict[str, List[str]]] = {}

    pool_weights = {
        "attractions": 3,
        "restaurants": 1,
        "hotels": 1,
    }

    for pool_name, pois in pools.items():
        for poi in pois:
            district = _poi_district(poi)
            name = (poi or {}).get("name", "").strip()
            if not district or not name:
                continue

            district_scores[district] = district_scores.get(district, 0) + pool_weights.get(pool_name, 1)
            district_candidates.setdefault(
                district,
                {"attractions": [], "restaurants": [], "hotels": []},
            )
            if name not in district_candidates[district][pool_name]:
                district_candidates[district][pool_name].append(name)

    ranked_districts = sorted(
        district_scores.keys(),
        key=lambda district: (-district_scores[district], district),
    )

    if not ranked_districts:
        return []

    plan = []
    for index in range(expected_days):
        district = ranked_districts[index] if index < len(ranked_districts) else ranked_districts[index % len(ranked_districts)]
        candidates = district_candidates.get(
            district,
            {"attractions": [], "restaurants": [], "hotels": []},
        )
        plan.append(
            {
                "day": index + 1,
                "district": district,
                "attraction_candidates": candidates["attractions"][:6],
                "restaurant_candidates": candidates["restaurants"][:4],
                "hotel_candidates": candidates["hotels"][:3],
            }
        )

    return plan


def _build_candidate_pool_by_district(
    tool_results: Dict[str, Any],
) -> Dict[str, Dict[str, List[str]]]:
    pools = _extract_poi_pool(tool_results)
    district_map: Dict[str, Dict[str, List[str]]] = {}

    for pool_name, pois in pools.items():
        for poi in pois:
            district = _poi_district(poi)
            name = (poi or {}).get("name", "").strip()
            if not district or not name:
                continue

            district_map.setdefault(
                district,
                {"attractions": [], "restaurants": [], "hotels": []},
            )
            if name not in district_map[district][pool_name]:
                district_map[district][pool_name].append(name)

    return district_map


def _compact_poi(poi: Dict[str, Any]) -> Dict[str, Any]:
    compact = {
        "name": (poi or {}).get("name"),
        "type": (poi or {}).get("type"),
        "district": _poi_district(poi),
        "address": (poi or {}).get("address"),
    }
    return {key: value for key, value in compact.items() if value}


def _compact_poi_list(pois: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    compact_pois: List[Dict[str, Any]] = []
    seen_names = set()
    for poi in pois:
        name = (poi or {}).get("name", "").strip()
        normalized = _normalize_name(name)
        if not normalized or normalized in seen_names:
            continue
        seen_names.add(normalized)
        compact_pois.append(_compact_poi(poi))
        if len(compact_pois) >= limit:
            break
    return compact_pois


def _compact_weather_result(
    weather_result: Dict[str, Any],
    expected_days: int,
) -> Dict[str, Any]:
    weather_items = weather_result.get("weather", []) or []
    compact_weather = []
    for item in weather_items[: max(expected_days, 1)]:
        compact_weather.append(
            {
                "date": item.get("date"),
                "day_weather": item.get("day_weather"),
                "night_weather": item.get("night_weather"),
                "day_temp": item.get("day_temp"),
                "night_temp": item.get("night_temp"),
            }
        )
    return {
        "city": weather_result.get("city", ""),
        "weather": compact_weather,
    }


def _compact_distance_result(distance_result: Dict[str, Any]) -> Dict[str, Any]:
    routes = []
    for route in (distance_result.get("routes", []) or [])[:3]:
        routes.append(
            {
                "from_name": route.get("from_name"),
                "to_name": route.get("to_name"),
                "mode": route.get("mode"),
                "distance_meters": route.get("distance_meters"),
                "duration_seconds": route.get("duration_seconds"),
            }
        )
    return {"routes": routes}


def _compact_candidate_pool_by_district(
    tool_results: Dict[str, Any],
) -> Dict[str, Dict[str, List[str]]]:
    district_map = _build_candidate_pool_by_district(tool_results)
    ranked_districts = sorted(
        district_map,
        key=lambda district: (
            -sum(len(values) for values in district_map[district].values()),
            district,
        ),
    )

    compact_map: Dict[str, Dict[str, List[str]]] = {}
    for district in ranked_districts[:MAX_CONTEXT_DISTRICTS]:
        candidates = district_map[district]
        compact_map[district] = {
            "attractions": candidates.get("attractions", [])[:6],
            "restaurants": candidates.get("restaurants", [])[:4],
            "hotels": candidates.get("hotels", [])[:2],
        }
    return compact_map


def _build_compact_tool_context(
    parsed_request: Dict[str, Any],
    tool_results: Dict[str, Any],
) -> Dict[str, Any]:
    expected_days = max(int(parsed_request.get("days", 0) or 0), 1)
    pools = _extract_poi_pool(tool_results)

    required_sightseeing_candidates = expected_days * MIN_SIGHTSEEING_ITEMS_PER_DAY + max(expected_days, 4)
    attraction_limit = min(
        max(MAX_CONTEXT_ATTRACTIONS, required_sightseeing_candidates),
        max(MAX_CONTEXT_ATTRACTIONS_CAP, MAX_CONTEXT_ATTRACTIONS),
    )
    restaurant_limit = min(
        max(MAX_CONTEXT_RESTAURANTS, expected_days * 2),
        max(MAX_CONTEXT_RESTAURANTS_CAP, MAX_CONTEXT_RESTAURANTS),
    )

    return {
        "weather": _compact_weather_result(
            tool_results.get("weather", {}),
            expected_days,
        ),
        "attractions": {
            "city": tool_results.get("attractions", {}).get("city", ""),
            "keywords": tool_results.get("attractions", {}).get("keywords", ""),
            "attractions": _compact_poi_list(
                pools.get("attractions", []),
                attraction_limit,
            ),
        },
        "restaurants": {
            "city": tool_results.get("restaurants", {}).get("city", ""),
            "keyword": tool_results.get("restaurants", {}).get("keyword", ""),
            "restaurants": _compact_poi_list(
                pools.get("restaurants", []),
                restaurant_limit,
            ),
        },
        "hotels": {
            "city": tool_results.get("hotels", {}).get("city", ""),
            "keyword": tool_results.get("hotels", {}).get("keyword", ""),
            "hotels": _compact_poi_list(
                pools.get("hotels", []),
                MAX_CONTEXT_HOTELS,
            ),
        },
        "distances": _compact_distance_result(tool_results.get("distances", {})),
        "candidate_pool_by_district": _compact_candidate_pool_by_district(tool_results),
    }


def _candidate_pool_order(category: str) -> List[str]:
    category = (category or "").strip()
    for key, pools in CATEGORY_POOL_MAP.items():
        if key in category:
            return pools
    return ["attractions", "restaurants", "hotels"]


def _next_unused_poi(
    pools: Dict[str, List[Dict[str, Any]]],
    pool_names: List[str],
    used_names: set[str],
    preferred_district: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if preferred_district:
        for pool_name in pool_names:
            for poi in pools.get(pool_name, []):
                normalized = _normalize_name(poi.get("name", ""))
                if not normalized or normalized in used_names:
                    continue
                if _poi_district(poi) == preferred_district:
                    return poi

    for pool_name in pool_names:
        for poi in pools.get(pool_name, []):
            normalized = _normalize_name(poi.get("name", ""))
            if normalized and normalized not in used_names:
                return poi
    return None


def _fallback_item_name(
    day_number: int,
    item_index: int,
    category: str,
    preferred_district: Optional[str] = None,
) -> str:
    category = (category or "").strip()
    district_label = f"第{day_number}天{preferred_district}" if preferred_district else f"第{day_number}天"
    if any(keyword in category for keyword in ["餐", "美食", "小吃", "咖啡"]):
        return f"{district_label}特色小吃探索"
    if any(keyword in category for keyword in ["酒店", "住宿", "民宿"]):
        return f"{district_label}住宿休整"
    if any(keyword in category for keyword in ["夜景", "夜游"]):
        return f"{district_label}夜景散步"
    if any(keyword in category for keyword in ["购物", "商圈"]):
        return f"{district_label}商圈闲逛"
    generic_names = ["城市地标漫步", "老街区探索", "滨江休闲散步", "街区文化体验"]
    return f"{district_label}{generic_names[(item_index - 1) % len(generic_names)]}"


def _fallback_description(
    day_number: int,
    category: str,
    preferred_district: Optional[str] = None,
) -> str:
    category = (category or "").strip()
    district_label = preferred_district or "当天片区"
    if any(keyword in category for keyword in ["餐", "美食", "小吃", "咖啡"]):
        return f"作为第{day_number}天的就近用餐安排，避免重复打卡同一餐厅，也减少跨区折返。"
    if any(keyword in category for keyword in ["酒店", "住宿", "民宿"]):
        return f"作为第{day_number}天的休整时段，给行程留出弹性，避免节奏过赶。"
    if any(keyword in category for keyword in ["夜景", "夜游"]):
        return f"安排在{district_label}晚间放松看夜景，避免重复前面已经去过的地点。"
    if any(keyword in category for keyword in ["购物", "商圈"]):
        return f"安排在{district_label}补充逛街和休息时间，可根据体力灵活调整。"
    return f"安排在{district_label}补充城市探索，保证行程丰富且不重复前面地点。"


def _collect_used_names(itinerary: Dict[str, Any]) -> set[str]:
    used_names: set[str] = set()
    for day in itinerary.get("days", []):
        for item in day.get("items", []):
            normalized = _normalize_name(item.get("name", ""))
            if normalized:
                used_names.add(normalized)
    return used_names


def _build_replacement_item(
    item: Dict[str, Any],
    day_number: int,
    item_index: int,
    pools: Dict[str, List[Dict[str, Any]]],
    used_names: set[str],
    preferred_district: Optional[str] = None,
) -> Dict[str, Any]:
    category = item.get("category", "其他")
    pool_names = _candidate_pool_order(category)
    replacement_poi = _next_unused_poi(
        pools,
        pool_names,
        used_names,
        preferred_district=preferred_district,
    )

    if replacement_poi:
        location_hint = replacement_poi.get("adname") or replacement_poi.get("address") or "当前片区"
        updated = dict(item)
        updated["name"] = replacement_poi.get("name", item.get("name", ""))
        updated["description"] = (
            f"安排前往{location_hint}的{updated['name']}，尽量保持当天路线顺路，并避免与前面重复。"
        )
        return updated

    updated = dict(item)
    updated["name"] = _fallback_item_name(day_number, item_index, category, preferred_district)
    updated["description"] = _fallback_description(day_number, category, preferred_district)
    if not updated.get("transport_to_next"):
        updated["transport_to_next"] = "根据当天实际位置灵活前往下一站"
    return updated


def _build_synthetic_day(
    day_number: int,
    preferred_district: str,
    pools: Dict[str, List[Dict[str, Any]]],
    used_names: set[str],
) -> Dict[str, Any]:
    placeholders = [
        {"time": "09:00-10:30", "category": "景点", "transport_to_next": "前往下一站约 15-20 分钟"},
        {"time": "11:00-12:30", "category": "景点", "transport_to_next": "就近午餐后前往下午行程"},
        {"time": "14:30-16:30", "category": "景点", "transport_to_next": "前往晚间活动约 20 分钟"},
        {"time": "19:00-21:00", "category": "夜景", "transport_to_next": ""},
    ]

    items = []
    for index, placeholder in enumerate(placeholders, start=1):
        item = _build_replacement_item(
            item={
                "time": placeholder["time"],
                "name": "",
                "category": placeholder["category"],
                "description": "",
                "transport_to_next": placeholder["transport_to_next"],
            },
            day_number=day_number,
            item_index=index,
            pools=pools,
            used_names=used_names,
            preferred_district=preferred_district or None,
        )
        normalized = _normalize_name(item.get("name", ""))
        if normalized:
            used_names.add(normalized)
        items.append(item)

    district_label = preferred_district or f"第{day_number}天"
    return {
        "day": day_number,
        "theme": f"{district_label}慢游",
        "route_summary": f"以{district_label}为中心安排新增的一天，减少跨区往返。",
        "items": items,
        "day_tips": [
            "这一天是根据你的补充需求新增的，建议灵活调整节奏。",
            "如遇排队或天气变化，可优先保留核心景点与晚间活动。",
        ],
    }


def ensure_daily_sightseeing_density(
    itinerary: Dict[str, Any],
    parsed_request: Dict[str, Any],
    tool_results: Dict[str, Any],
    day_district_plan: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    pools = _extract_poi_pool(tool_results)
    used_names = _collect_used_names(itinerary)
    district_map = {
        plan.get("day"): plan.get("district", "").strip()
        for plan in (day_district_plan or [])
    }

    for day_index, day in enumerate(itinerary.get("days", []), start=1):
        items = day.get("items", [])
        if not isinstance(items, list):
            items = []
            day["items"] = items

        preferred_district = district_map.get(day.get("day") or day_index)

        sightseeing_count = sum(1 for item in items if _is_sightseeing_item(item))
        for item_index, item in enumerate(items, start=1):
            if sightseeing_count >= MIN_SIGHTSEEING_ITEMS_PER_DAY:
                break
            if _is_sightseeing_item(item):
                continue

            old_normalized = _normalize_name(item.get("name", ""))
            if old_normalized:
                used_names.discard(old_normalized)

            target_category = _sightseeing_category_for_position(sightseeing_count + 1)
            item.update(
                _build_replacement_item(
                    item={
                        **item,
                        "category": target_category,
                    },
                    day_number=day_index,
                    item_index=item_index,
                    pools=pools,
                    used_names=used_names,
                    preferred_district=preferred_district or None,
                )
            )
            item["category"] = target_category
            normalized = _normalize_name(item.get("name", ""))
            if normalized:
                used_names.add(normalized)
            sightseeing_count += 1

        while sightseeing_count < MIN_SIGHTSEEING_ITEMS_PER_DAY:
            item_index = len(items) + 1
            target_category = _sightseeing_category_for_position(sightseeing_count + 1)
            template = _day_time_template(
                max(item_index, MIN_SIGHTSEEING_ITEMS_PER_DAY),
                parsed_request,
            ) or FULL_DAY_TIME_TEMPLATES[MIN_SIGHTSEEING_ITEMS_PER_DAY]
            time_range = template[min(item_index - 1, len(template) - 1)]
            new_item = _build_replacement_item(
                item={
                    "time": time_range,
                    "name": "",
                    "category": target_category,
                    "description": "",
                    "transport_to_next": "前往下一站约 15-20 分钟" if target_category != "夜景" else "",
                },
                day_number=day_index,
                item_index=item_index,
                pools=pools,
                used_names=used_names,
                preferred_district=preferred_district or None,
            )
            new_item["category"] = target_category
            normalized = _normalize_name(new_item.get("name", ""))
            if normalized:
                used_names.add(normalized)
            items.append(new_item)
            sightseeing_count += 1

        if items and not _has_evening_item(items):
            evening_item = items[-1]
            old_normalized = _normalize_name(evening_item.get("name", ""))
            if old_normalized:
                used_names.discard(old_normalized)
            evening_item.update(
                _build_replacement_item(
                    item={
                        **evening_item,
                        "time": "19:00-21:00",
                        "category": "夜景",
                        "transport_to_next": "",
                    },
                    day_number=day_index,
                    item_index=len(items),
                    pools=pools,
                    used_names=used_names,
                    preferred_district=preferred_district or None,
                )
            )
            evening_item["time"] = "19:00-21:00"
            evening_item["category"] = "夜景"
            normalized = _normalize_name(evening_item.get("name", ""))
            if normalized:
                used_names.add(normalized)

        _apply_day_time_template(day, parsed_request)
        day["items"] = items

        day_tips = day.get("day_tips", [])
        if not isinstance(day_tips, list):
            day_tips = []
        note = "已自动补齐为每天至少 4 个游玩点，并覆盖上午、下午和晚上。"
        if note not in day_tips:
            day_tips.append(note)
        day["day_tips"] = day_tips

    return itinerary


def apply_hard_constraints(
    itinerary: Dict[str, Any],
    parsed_request: Dict[str, Any],
    tool_results: Dict[str, Any],
    day_district_plan: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    constraints = parsed_request.get("hard_constraints", {}) or {}
    avoid_global = _normalize_constraint_names(constraints.get("avoid_places_global", []))
    avoid_by_day = {
        str(day): _normalize_constraint_names(names)
        for day, names in (constraints.get("avoid_places_by_day", {}) or {}).items()
    }
    day_focus = {
        str(day): focus
        for day, focus in (constraints.get("day_focus", {}) or {}).items()
        if focus
    }

    if not avoid_global and not avoid_by_day and not day_focus:
        return itinerary

    pools = _extract_poi_pool(tool_results)
    used_names = _collect_used_names(itinerary)
    district_map = {
        str(plan.get("day")): plan.get("district", "").strip()
        for plan in (day_district_plan or [])
    }

    for day in itinerary.get("days", []):
        day_number = day.get("day")
        day_key = str(day_number)
        preferred_district = district_map.get(day_key, "")
        day_avoid = set(avoid_global) | set(avoid_by_day.get(day_key, set()))

        if day_avoid:
            for item_index, item in enumerate(day.get("items", []), start=1):
                normalized = _normalize_name(item.get("name", ""))
                if normalized and normalized in day_avoid:
                    item.update(
                        _build_replacement_item(
                            item=item,
                            day_number=day_number,
                            item_index=item_index,
                            pools=pools,
                            used_names=used_names,
                            preferred_district=preferred_district or None,
                        )
                    )
                    replacement_name = _normalize_name(item.get("name", ""))
                    if replacement_name:
                        used_names.add(replacement_name)

        focus = day_focus.get(day_key)
        if not focus:
            continue

        focus_plan = DAY_FOCUS_CATEGORY_PLAN.get(focus, ["景点", "餐饮", "夜景"])
        day["theme"] = f"{focus}主题日"
        if preferred_district:
            day["theme"] = f"{preferred_district}·{focus}主题日"
        day["route_summary"] = f"这一天以{focus}为主，尽量围绕{preferred_district or '当天片区'}展开。"

        items = day.get("items", [])
        for item_index, item in enumerate(items, start=1):
            if item_index > len(focus_plan):
                break
            target_category = focus_plan[item_index - 1]
            if item_index == len(items) and target_category != "夜景":
                continue
            item["category"] = target_category
            item.update(
                _build_replacement_item(
                    item=item,
                    day_number=day_number,
                    item_index=item_index,
                    pools=pools,
                    used_names=used_names,
                    preferred_district=preferred_district or None,
                )
            )
            normalized = _normalize_name(item.get("name", ""))
            if normalized:
                used_names.add(normalized)

        day_tips = day.get("day_tips", [])
        note = f"已按你的要求把第{day_number}天调整为以{focus}为主。"
        if note not in day_tips:
            day_tips.append(note)
        day["day_tips"] = day_tips

    return itinerary


def apply_departure_preference(
    itinerary: Dict[str, Any],
    parsed_request: Dict[str, Any],
) -> Dict[str, Any]:
    if not _prefers_afternoon_start(parsed_request):
        return itinerary

    for day in itinerary.get("days", []):
        items = day.get("items", [])
        if not items:
            continue

        first_hour = _parse_start_hour(items[0].get("time", ""))
        if first_hour is not None and first_hour >= 12:
            continue

        template = AFTERNOON_TIME_TEMPLATES.get(len(items))
        if not template:
            continue

        for item, new_time in zip(items, template):
            item["time"] = new_time

        day_tips = day.get("day_tips", [])
        note = "已按你的习惯调整为下午出门，上午留作休息或自由安排。"
        if note not in day_tips:
            day_tips.append(note)
        day["day_tips"] = day_tips

    return itinerary


def repair_itinerary_duplicates(
    itinerary: Dict[str, Any],
    tool_results: Dict[str, Any],
    day_district_plan: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    pools = _extract_poi_pool(tool_results)
    used_names: set[str] = set()
    district_map = {
        plan.get("day"): plan.get("district", "").strip()
        for plan in (day_district_plan or [])
    }

    for day_index, day in enumerate(itinerary.get("days", []), start=1):
        items = day.get("items", [])
        preferred_district = district_map.get(day_index)

        if preferred_district:
            theme = day.get("theme", "").strip()
            route_summary = day.get("route_summary", "").strip()
            if theme and preferred_district not in theme:
                day["theme"] = f"{preferred_district}·{theme}"
            elif not theme:
                day["theme"] = f"{preferred_district}漫游"

            if route_summary and preferred_district not in route_summary:
                day["route_summary"] = f"以{preferred_district}为中心安排当日路线。{route_summary}"
            elif not route_summary:
                day["route_summary"] = f"以{preferred_district}为中心，减少跨区往返。"

        for item_index, item in enumerate(items, start=1):
            name = item.get("name", "").strip()
            normalized = _normalize_name(name)

            if not normalized:
                repaired_item = _build_replacement_item(
                    item=item,
                    day_number=day_index,
                    item_index=item_index,
                    pools=pools,
                    used_names=used_names,
                    preferred_district=preferred_district,
                )
                item.update(repaired_item)
                normalized = _normalize_name(item.get("name", ""))
                if normalized:
                    used_names.add(normalized)
                continue

            if normalized in used_names:
                repaired_item = _build_replacement_item(
                    item=item,
                    day_number=day_index,
                    item_index=item_index,
                    pools=pools,
                    used_names=used_names,
                    preferred_district=preferred_district,
                )
                item.update(repaired_item)
                normalized = _normalize_name(item.get("name", ""))

            if normalized:
                used_names.add(normalized)

    return itinerary


def ensure_itinerary_day_count(
    itinerary: Dict[str, Any],
    expected_days: int,
    tool_results: Dict[str, Any],
    day_district_plan: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if expected_days <= 0:
        return itinerary

    days = itinerary.get("days", [])
    if len(days) > expected_days:
        itinerary["days"] = days[:expected_days]
        for index, day in enumerate(itinerary["days"], start=1):
            day["day"] = index
        return itinerary

    if len(days) == expected_days:
        return itinerary

    pools = _extract_poi_pool(tool_results)
    used_names = _collect_used_names(itinerary)
    district_map = {
        plan.get("day"): plan.get("district", "").strip()
        for plan in (day_district_plan or [])
    }

    current_days = itinerary.get("days", [])
    while len(current_days) < expected_days:
        next_day_number = len(current_days) + 1
        preferred_district = district_map.get(next_day_number, "")
        synthetic_day = _build_synthetic_day(
            day_number=next_day_number,
            preferred_district=preferred_district,
            pools=pools,
            used_names=used_names,
        )
        current_days.append(synthetic_day)

    itinerary["days"] = current_days
    return itinerary


def _pick_hotel_area(tool_results: Dict[str, Any]) -> str:
    hotels = tool_results.get("hotels", {}).get("hotels", []) or []
    area_counter: Counter[str] = Counter()
    for hotel in hotels:
        area = (
            (hotel or {}).get("adname")
            or (hotel or {}).get("cityname")
            or (hotel or {}).get("pname")
            or ""
        ).strip()
        if area:
            area_counter[area] += 1

    if not area_counter:
        city = tool_results.get("weather", {}).get("city", "") or "市中心"
        return f"建议优先住在{city}交通方便、餐饮集中的区域。"

    top_areas = [name for name, _ in area_counter.most_common(2)]
    if len(top_areas) == 1:
        return f"建议住在{top_areas[0]}附近，吃饭和出行会更方便。"
    return f"建议优先考虑住在{top_areas[0]}或{top_areas[1]}附近，兼顾交通和餐饮便利。"


def _build_weather_advice(tool_results: Dict[str, Any]) -> str:
    weather_items = tool_results.get("weather", {}).get("weather", []) or []
    if not weather_items:
        return "出发前建议再确认一次当地天气，按温差准备外套和舒适步行鞋。"

    first = weather_items[0]
    day_weather = first.get("day_weather", "")
    night_weather = first.get("night_weather", "")
    day_temp = first.get("day_temp", "")
    night_temp = first.get("night_temp", "")

    weather_text = day_weather or night_weather or "天气多变"
    temp_text = ""
    if day_temp and night_temp:
        temp_text = f"白天约 {day_temp}℃，夜间约 {night_temp}℃。"

    if any(keyword in weather_text for keyword in ["雨", "阵雨", "雷"]):
        extra = "建议随身带伞，优先准备防滑鞋。"
    elif any(keyword in weather_text for keyword in ["晴", "多云"]):
        extra = "建议做好防晒，白天活动注意补水。"
    else:
        extra = "建议穿着方便增减的衣物。"

    return f"近期以{weather_text}为主，{temp_text}{extra}".strip()


def _build_transport_advice(
    parsed_request: Dict[str, Any],
    tool_results: Dict[str, Any],
) -> str:
    preference = (parsed_request.get("transport_preference", "") or "").strip()
    distance_routes = tool_results.get("distances", {}).get("routes", []) or []

    route_hint = ""
    if distance_routes:
        valid_durations = []
        for route in distance_routes:
            duration = route.get("duration_seconds")
            if duration is None:
                continue
            try:
                valid_durations.append(int(duration))
            except (TypeError, ValueError):
                continue
        if valid_durations:
            avg_minutes = round(sum(valid_durations) / len(valid_durations) / 60)
            route_hint = f"景点间单段通勤大多在 {avg_minutes} 分钟左右。"

    if "公共交通" in preference:
        base = "建议优先地铁和公交，跨区再配合打车。"
    elif "打车" in preference:
        base = "建议以打车为主，能更稳定控制跨区通勤时间。"
    elif "步行" in preference:
        base = "建议把同片区景点连起来走，减少折返。"
    else:
        base = "建议同片区尽量步行或地铁，跨区时再灵活打车。"

    return f"{base}{route_hint}".strip()


def ensure_itinerary_supporting_fields(
    itinerary: Dict[str, Any],
    parsed_request: Dict[str, Any],
    tool_results: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(itinerary.get("important_tips"), list):
        itinerary["important_tips"] = []

    if not itinerary.get("hotel_area_suggestion"):
        itinerary["hotel_area_suggestion"] = _pick_hotel_area(tool_results)

    if not itinerary.get("weather_advice"):
        itinerary["weather_advice"] = _build_weather_advice(tool_results)

    if not itinerary.get("transport_advice"):
        itinerary["transport_advice"] = _build_transport_advice(parsed_request, tool_results)

    if not itinerary.get("summary"):
        city = parsed_request.get("city", "") or "目的地"
        days = parsed_request.get("days", "") or "多日"
        itinerary["summary"] = f"这是一份围绕{city}安排的 {days} 天旅行路线。"

    if not itinerary.get("title"):
        city = parsed_request.get("city", "") or "城市"
        days = parsed_request.get("days", "") or "多日"
        itinerary["title"] = f"{city}{days}日游"

    return itinerary


def _compact_rag_context(rag_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(rag_context, dict):
        return {}

    compact_chunks = []
    for chunk in (rag_context.get("chunks") or [])[:MAX_RAG_CHUNKS]:
        content = str(chunk.get("content", "") or "").strip()
        if len(content) > MAX_RAG_CONTENT_CHARS:
            content = content[:MAX_RAG_CONTENT_CHARS].rstrip() + "..."
        compact_chunks.append(
            {
                "source": chunk.get("source", ""),
                "title": chunk.get("title", ""),
                "content": content,
                "score": chunk.get("score", 0),
            }
        )

    return {
        "retriever": rag_context.get("retriever", ""),
        "query": rag_context.get("query", ""),
        "chunks": compact_chunks,
        "summary": "\n".join(
            f"- [{chunk['source']}] {chunk['title']}: {chunk['content']}"
            for chunk in compact_chunks
        ),
    }


def _fast_theme_suffix(day_number: int, parsed_request: Dict[str, Any]) -> str:
    preferences = [
        str(item).strip()
        for item in (parsed_request.get("preferences", []) or [])
        if str(item).strip()
    ]
    if preferences:
        preference = preferences[(day_number - 1) % len(preferences)]
        theme_map = {
            "夜景": "夜景漫游",
            "历史文化": "历史文化线",
            "博物馆": "文化展馆线",
            "购物": "商圈休闲线",
            "自然风景": "自然休闲线",
            "美食": "美食街区线",
            "拍照": "拍照打卡线",
            "citywalk": "城市漫步线",
        }
        return theme_map.get(preference, f"{preference}主题线")

    default_suffixes = [
        "经典打卡线",
        "城市漫步线",
        "文化休闲线",
        "滨江夜景线",
        "轻松探索线",
        "街区体验线",
        "收尾慢游线",
    ]
    return default_suffixes[(day_number - 1) % len(default_suffixes)]


def _fast_day_tip(
    day_number: int,
    preferred_district: str,
    parsed_request: Dict[str, Any],
) -> List[str]:
    district_label = preferred_district or "当天片区"
    tips = [f"第{day_number}天尽量围绕{district_label}游玩，减少跨区折返。"]
    if _prefers_afternoon_start(parsed_request):
        tips.append("已按下午出门节奏安排，上午可以休息或自由活动。")
    else:
        tips.append("当天覆盖上午、下午和晚上，节奏偏充实，可按体力删减。")
    return tips


def _fast_important_tips(
    parsed_request: Dict[str, Any],
    rag_context: Optional[Dict[str, Any]],
) -> List[str]:
    tips = [
        "每天至少安排 4 个游玩点，并尽量避免重复地点。",
        "多日路线按片区分天，实际出行时可根据天气和体力微调顺序。",
    ]
    compact_rag = _compact_rag_context(rag_context)
    rag_titles = [
        chunk.get("title", "")
        for chunk in compact_rag.get("chunks", [])
        if chunk.get("title")
    ]
    if rag_titles:
        tips.append(f"已参考本地攻略知识：{'、'.join(rag_titles[:2])}。")
    if _prefers_afternoon_start(parsed_request):
        tips.append("你偏好下午出门，行程已避免上午过早开始。")
    return tips


def _build_fast_itinerary(
    parsed_request: Dict[str, Any],
    tool_results: Dict[str, Any],
    rag_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    city = (parsed_request.get("city", "") or "目的地").strip()
    expected_days = max(int(parsed_request.get("days", 0) or 0), 1)
    pools = _extract_poi_pool(tool_results)
    day_district_plan = _build_day_district_plan(parsed_request, tool_results)
    district_map = {
        plan.get("day"): plan.get("district", "").strip()
        for plan in day_district_plan
    }
    used_names: set[str] = set()
    days = []

    for day_number in range(1, expected_days + 1):
        preferred_district = district_map.get(day_number, "")
        template = _day_time_template(MIN_SIGHTSEEING_ITEMS_PER_DAY, parsed_request)
        template = template or FULL_DAY_TIME_TEMPLATES[MIN_SIGHTSEEING_ITEMS_PER_DAY]
        categories = ["景点", "景点", "景点", "夜景"]
        items = []

        for item_index, (time_range, category) in enumerate(zip(template, categories), start=1):
            item = _build_replacement_item(
                item={
                    "time": time_range,
                    "name": "",
                    "category": category,
                    "description": "",
                    "transport_to_next": (
                        "前往下一站约 15-20 分钟"
                        if item_index < MIN_SIGHTSEEING_ITEMS_PER_DAY
                        else ""
                    ),
                },
                day_number=day_number,
                item_index=item_index,
                pools=pools,
                used_names=used_names,
                preferred_district=preferred_district or None,
            )
            item["time"] = time_range
            item["category"] = category
            normalized = _normalize_name(item.get("name", ""))
            if normalized:
                used_names.add(normalized)
            items.append(item)

        district_label = preferred_district or city
        theme_suffix = _fast_theme_suffix(day_number, parsed_request)
        days.append(
            {
                "day": day_number,
                "theme": f"{district_label}·{theme_suffix}",
                "route_summary": f"以{district_label}为中心串联 4 个游玩点，保证从早到晚且不重复景点。",
                "items": items,
                "day_tips": _fast_day_tip(day_number, district_label, parsed_request),
            }
        )

    itinerary = {
        "title": f"{city}{expected_days}日游",
        "summary": f"这是一份围绕{city}不同片区展开的 {expected_days} 天快速结构化行程。",
        "important_tips": _fast_important_tips(parsed_request, rag_context),
        "days": days,
    }
    return postprocess_itinerary(itinerary, parsed_request, tool_results)


def generate_itinerary_with_llm(
    parsed_request: Dict[str, Any],
    execution_plan: Dict[str, Any],
    tool_results: Dict[str, Any],
    rag_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if FAST_ITINERARY_ENABLED:
        return _build_fast_itinerary(parsed_request, tool_results, rag_context)

    day_district_plan = _build_day_district_plan(parsed_request, tool_results)
    compact_tool_results = _build_compact_tool_context(parsed_request, tool_results)
    candidate_pool = {
        "attractions": [
            poi.get("name")
            for poi in compact_tool_results.get("attractions", {}).get("attractions", [])
            if poi.get("name")
        ],
        "restaurants": [
            poi.get("name")
            for poi in compact_tool_results.get("restaurants", {}).get("restaurants", [])
            if poi.get("name")
        ],
        "hotels": [
            poi.get("name")
            for poi in compact_tool_results.get("hotels", {}).get("hotels", [])
            if poi.get("name")
        ],
    }

    user_content = {
        "parsed_request": parsed_request,
        "execution_plan": execution_plan,
        "compact_tool_results": compact_tool_results,
        "candidate_pool": candidate_pool,
        "candidate_pool_by_district": compact_tool_results.get("candidate_pool_by_district", {}),
        "day_district_plan": day_district_plan,
        "rag_context": _compact_rag_context(rag_context),
    }

    model_name = get_model_name()
    response = get_client().chat.completions.create(
        model=model_name,
        temperature=0.2,
        messages=[
            {"role": "system", "content": ITINERARY_PROMPT},
            {
                "role": "user",
                "content": json.dumps(user_content, ensure_ascii=False),
            },
        ],
    )
    record_token_usage("itinerary_generation", response, model_name)

    content = response.choices[0].message.content
    itinerary = extract_json_from_text(content)
    return postprocess_itinerary(itinerary, parsed_request, tool_results)


def postprocess_itinerary(
    itinerary: Dict[str, Any],
    parsed_request: Dict[str, Any],
    tool_results: Dict[str, Any],
) -> Dict[str, Any]:
    day_district_plan = _build_day_district_plan(parsed_request, tool_results)
    itinerary = ensure_itinerary_day_count(
        itinerary,
        expected_days=int(parsed_request.get("days", 0) or 0),
        tool_results=tool_results,
        day_district_plan=day_district_plan,
    )
    itinerary = apply_hard_constraints(
        itinerary,
        parsed_request,
        tool_results,
        day_district_plan=day_district_plan,
    )
    itinerary = repair_itinerary_duplicates(
        itinerary,
        tool_results,
        day_district_plan=day_district_plan,
    )
    itinerary = ensure_daily_sightseeing_density(
        itinerary,
        parsed_request,
        tool_results,
        day_district_plan=day_district_plan,
    )
    itinerary = repair_itinerary_duplicates(
        itinerary,
        tool_results,
        day_district_plan=day_district_plan,
    )
    itinerary = apply_departure_preference(itinerary, parsed_request)
    return ensure_itinerary_supporting_fields(itinerary, parsed_request, tool_results)
