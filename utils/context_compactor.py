from typing import Any, Dict, List, Optional

from utils.config import get_setting


HISTORY_LIMIT = max(1, int(get_setting("TRAVEL_AGENT_FOLLOWUP_HISTORY_LIMIT", "4") or "4"))
HISTORY_MESSAGE_CHARS = max(
    80,
    int(get_setting("TRAVEL_AGENT_FOLLOWUP_HISTORY_CHARS", "260") or "260"),
)
MAX_CONTEXT_DAYS = max(1, int(get_setting("TRAVEL_AGENT_CONTEXT_DAYS", "10") or "10"))
MAX_ITEMS_PER_DAY = max(2, int(get_setting("TRAVEL_AGENT_CONTEXT_ITEMS_PER_DAY", "5") or "5"))
MAX_TOOL_ITEMS = max(3, int(get_setting("TRAVEL_AGENT_CONTEXT_TOOL_ITEMS", "8") or "8"))


def clip_text(text: Any, limit: int) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "..."


def compact_history(
    conversation_history: Optional[List[Dict[str, str]]],
    limit: int = HISTORY_LIMIT,
) -> List[Dict[str, str]]:
    history = conversation_history or []
    compacted = []
    for message in history[-limit:]:
        compacted.append(
            {
                "role": message.get("role", "user"),
                "content": clip_text(message.get("content", ""), HISTORY_MESSAGE_CHARS),
            }
        )
    return compacted


def compact_itinerary(itinerary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(itinerary, dict):
        return {}

    days = []
    for day in (itinerary.get("days") or [])[:MAX_CONTEXT_DAYS]:
        items = []
        for item in (day.get("items") or [])[:MAX_ITEMS_PER_DAY]:
            items.append(
                {
                    "time": item.get("time", ""),
                    "name": item.get("name", ""),
                    "category": item.get("category", ""),
                }
            )
        days.append(
            {
                "day": day.get("day"),
                "theme": day.get("theme", ""),
                "route_summary": clip_text(day.get("route_summary", ""), 120),
                "items": items,
            }
        )

    return {
        "title": itinerary.get("title", ""),
        "summary": clip_text(itinerary.get("summary", ""), 180),
        "days": days,
        "hotel_area_suggestion": clip_text(itinerary.get("hotel_area_suggestion", ""), 120),
        "weather_advice": clip_text(itinerary.get("weather_advice", ""), 120),
        "transport_advice": clip_text(itinerary.get("transport_advice", ""), 120),
    }


def _compact_poi_list(items: List[Dict[str, Any]], limit: int = MAX_TOOL_ITEMS) -> List[Dict[str, Any]]:
    compacted = []
    seen = set()
    for item in items or []:
        name = (item or {}).get("name", "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        compacted.append(
            {
                "name": name,
                "district": (item or {}).get("adname") or (item or {}).get("cityname") or "",
                "type": (item or {}).get("type", ""),
            }
        )
        if len(compacted) >= limit:
            break
    return compacted


def compact_tool_results(tool_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(tool_results, dict):
        return {}

    weather_items = []
    for item in (tool_results.get("weather", {}).get("weather") or [])[:3]:
        weather_items.append(
            {
                "date": item.get("date", ""),
                "day_weather": item.get("day_weather", ""),
                "day_temp": item.get("day_temp", ""),
                "night_temp": item.get("night_temp", ""),
            }
        )

    routes = []
    for route in (tool_results.get("distances", {}).get("routes") or [])[:2]:
        routes.append(
            {
                "from_name": route.get("from_name", ""),
                "to_name": route.get("to_name", ""),
                "duration_seconds": route.get("duration_seconds"),
            }
        )

    return {
        "weather": {
            "city": tool_results.get("weather", {}).get("city", ""),
            "weather": weather_items,
        },
        "attractions": _compact_poi_list(
            tool_results.get("attractions", {}).get("attractions", [])
        ),
        "restaurants": _compact_poi_list(
            tool_results.get("restaurants", {}).get("restaurants", [])
        ),
        "hotels": _compact_poi_list(tool_results.get("hotels", {}).get("hotels", [])),
        "distances": routes,
    }
