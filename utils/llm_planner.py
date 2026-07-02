from typing import Any, Dict, List


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _clean_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _append_task(
    tasks: List[Dict[str, Any]],
    tool: str,
    reason: str,
) -> None:
    tasks.append(
        {
            "step": len(tasks) + 1,
            "tool": tool,
            "reason": reason,
        }
    )


def generate_plan(parsed_request: Dict[str, Any]) -> Dict[str, Any]:
    city = _clean_text(parsed_request.get("city")) or "目的地"
    days = parsed_request.get("days") or 0
    preferences = _clean_list(parsed_request.get("preferences"))
    special_requirements = _clean_text(parsed_request.get("special_requirements"))

    tasks: List[Dict[str, Any]] = []
    _append_task(tasks, "weather_tool", "查询目的地天气，用于生成穿衣与室内外安排建议。")
    _append_task(tasks, "attraction_tool", "根据城市和偏好检索候选景点，支撑每日路线编排。")
    _append_task(tasks, "restaurant_tool", "检索餐厅候选，补充午餐、晚餐和美食偏好。")
    _append_task(tasks, "hotel_tool", "检索住宿候选，用于生成住宿区域建议。")
    _append_task(tasks, "distance_tool", "如已启用距离工具，则补充部分景点之间的路线距离。")
    _append_task(tasks, "itinerary_tool", "结合结构化需求和工具结果生成最终行程。")

    return {
        "planner_type": "deterministic",
        "llm_used": False,
        "destination": city,
        "days": days,
        "preferences": preferences,
        "special_requirements": special_requirements,
        "tasks": tasks,
    }


def generate_plan_with_llm(parsed_request: Dict[str, Any]) -> Dict[str, Any]:
    return generate_plan(parsed_request)
