import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional, Tuple, TypedDict

from utils.config import get_setting
from utils.langchain_tools import invoke_travel_tool

try:
    from langgraph.graph import END, StateGraph
except ImportError:
    END = None
    StateGraph = None


MAX_TOOL_WORKERS = max(1, int(get_setting("TRAVEL_AGENT_MAX_TOOL_WORKERS", "2") or "2"))
ENABLE_DISTANCE_TOOL = get_setting("TRAVEL_AGENT_ENABLE_DISTANCE", "1").lower() not in {
    "0",
    "false",
    "no",
}
MIN_ATTRACTION_CANDIDATES = max(20, int(get_setting("TRAVEL_AGENT_MIN_ATTRACTION_CANDIDATES", "20") or "20"))


class TravelState(TypedDict):
    parsed_request: Dict[str, Any]
    weather_result: Dict[str, Any]
    attraction_result: Dict[str, Any]
    restaurant_result: Dict[str, Any]
    hotel_result: Dict[str, Any]
    distance_result: Dict[str, Any]
    tool_results: Dict[str, Any]


def _normalize_city(parsed_request: Dict[str, Any]) -> str:
    return (parsed_request.get("city", "") or "").strip()


def _normalize_preferences(parsed_request: Dict[str, Any]) -> Tuple[str, ...]:
    preferences = parsed_request.get("preferences", []) or []
    normalized = sorted({str(item).strip() for item in preferences if str(item).strip()})
    return tuple(normalized)


def _normalize_days(parsed_request: Optional[Dict[str, Any]]) -> int:
    if not isinstance(parsed_request, dict):
        return 0
    try:
        return max(int(parsed_request.get("days", 0) or 0), 0)
    except (TypeError, ValueError):
        return 0


def _tool_signatures(parsed_request: Dict[str, Any]) -> Dict[str, Tuple[Any, ...]]:
    city = _normalize_city(parsed_request)
    preferences = _normalize_preferences(parsed_request)
    return {
        "weather": (city,),
        "attractions": (city, preferences),
        "restaurants": (city,),
        "hotels": (city,),
    }


def _tool_results_complete(tool_results: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(tool_results, dict):
        return False
    required_keys = {"weather", "attractions", "restaurants", "hotels", "distances"}
    return required_keys.issubset(tool_results.keys())


def _clone_result(data: Any) -> Any:
    if isinstance(data, dict):
        return {key: _clone_result(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_clone_result(item) for item in data]
    return data


def _is_empty_limited_result(tool_name: str, result: Dict[str, Any]) -> bool:
    if not isinstance(result, dict) or not result.get("warning"):
        return False
    list_key_map = {
        "weather": "weather",
        "attractions": "attractions",
        "restaurants": "restaurants",
        "hotels": "hotels",
    }
    list_key = list_key_map.get(tool_name)
    return bool(list_key and not result.get(list_key))


def _reuse_previous_on_limited_result(
    tool_name: str,
    result: Dict[str, Any],
    previous_tool_results: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not _is_empty_limited_result(tool_name, result):
        return result

    previous_result = (previous_tool_results or {}).get(tool_name)
    if not previous_result:
        return result

    reused = _clone_result(previous_result)
    warnings = reused.get("warnings", []) if isinstance(reused.get("warnings"), list) else []
    warnings.append(f"本次{tool_name}工具触发高德限流，已复用上一次结果。")
    reused["warnings"] = warnings
    return reused


def _should_refresh_tool(
    tool_name: str,
    parsed_request: Dict[str, Any],
    previous_request: Optional[Dict[str, Any]],
    previous_tool_results: Optional[Dict[str, Any]],
) -> bool:
    if previous_request is None or not _tool_results_complete(previous_tool_results):
        return True

    previous_signatures = _tool_signatures(previous_request)
    current_signatures = _tool_signatures(parsed_request)
    if previous_signatures.get(tool_name) != current_signatures.get(tool_name):
        return True

    current_days = _normalize_days(parsed_request)
    previous_days = _normalize_days(previous_request)
    if tool_name == "attractions":
        attractions = previous_tool_results.get("attractions", {}).get("attractions", []) or []
        required_candidates = max(MIN_ATTRACTION_CANDIDATES, current_days * 4 + max(current_days, 4))
        if current_days > previous_days or len(attractions) < required_candidates:
            return True

    if tool_name == "restaurants":
        restaurants = previous_tool_results.get("restaurants", {}).get("restaurants", []) or []
        if current_days > previous_days and len(restaurants) < max(8, current_days * 2):
            return True

    return not previous_tool_results.get(tool_name)


def _build_tool_results(
    weather_result: Dict[str, Any],
    attraction_result: Dict[str, Any],
    restaurant_result: Dict[str, Any],
    hotel_result: Dict[str, Any],
    distance_result: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "weather": weather_result,
        "attractions": attraction_result,
        "restaurants": restaurant_result,
        "hotels": hotel_result,
        "distances": distance_result,
    }


def run_weather(state: TravelState) -> TravelState:
    city = _normalize_city(state["parsed_request"])
    state["weather_result"] = invoke_travel_tool("weather", {"city": city})
    return state


def run_attractions(state: TravelState) -> TravelState:
    parsed_request = state["parsed_request"]
    state["attraction_result"] = invoke_travel_tool(
        "attractions",
        {
            "city": _normalize_city(parsed_request),
            "preferences": list(_normalize_preferences(parsed_request)),
        },
    )
    return state


def run_restaurants(state: TravelState) -> TravelState:
    city = _normalize_city(state["parsed_request"])
    state["restaurant_result"] = invoke_travel_tool("restaurants", {"city": city})
    return state


def run_hotels(state: TravelState) -> TravelState:
    city = _normalize_city(state["parsed_request"])
    state["hotel_result"] = invoke_travel_tool("hotels", {"city": city})
    return state


def run_distances(state: TravelState) -> TravelState:
    attractions = state.get("attraction_result", {}).get("attractions", [])
    state["distance_result"] = (
        invoke_travel_tool("distances", {"attractions": attractions})
        if ENABLE_DISTANCE_TOOL
        else {"routes": []}
    )
    return state


def collect_results(state: TravelState) -> TravelState:
    state["tool_results"] = _build_tool_results(
        state.get("weather_result", {}),
        state.get("attraction_result", {}),
        state.get("restaurant_result", {}),
        state.get("hotel_result", {}),
        state.get("distance_result", {}),
    )
    return state


def build_travel_graph():
    if StateGraph is None or END is None:
        raise ImportError("langgraph is required to build the travel tool graph.")

    graph = StateGraph(TravelState)
    graph.add_node("weather", run_weather)
    graph.add_node("attractions", run_attractions)
    graph.add_node("restaurants", run_restaurants)
    graph.add_node("hotels", run_hotels)
    graph.add_node("distances", run_distances)
    graph.add_node("collect", collect_results)
    graph.set_entry_point("weather")
    graph.add_edge("weather", "attractions")
    graph.add_edge("attractions", "restaurants")
    graph.add_edge("restaurants", "hotels")
    graph.add_edge("hotels", "distances")
    graph.add_edge("distances", "collect")
    graph.add_edge("collect", END)
    return graph.compile()


def run_travel_agent_tools(
    parsed_request: Dict[str, Any],
    previous_request: Optional[Dict[str, Any]] = None,
    previous_tool_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    city = _normalize_city(parsed_request)
    if not city:
        return _build_tool_results({}, {}, {}, {}, {"routes": []})

    previous_tool_results = previous_tool_results or {}
    weather_result = (
        _clone_result(previous_tool_results.get("weather", {}))
        if not _should_refresh_tool("weather", parsed_request, previous_request, previous_tool_results)
        else None
    )
    attraction_result = (
        _clone_result(previous_tool_results.get("attractions", {}))
        if not _should_refresh_tool("attractions", parsed_request, previous_request, previous_tool_results)
        else None
    )
    restaurant_result = (
        _clone_result(previous_tool_results.get("restaurants", {}))
        if not _should_refresh_tool("restaurants", parsed_request, previous_request, previous_tool_results)
        else None
    )
    hotel_result = (
        _clone_result(previous_tool_results.get("hotels", {}))
        if not _should_refresh_tool("hotels", parsed_request, previous_request, previous_tool_results)
        else None
    )

    future_map = {}
    with ThreadPoolExecutor(max_workers=MAX_TOOL_WORKERS) as executor:
        if weather_result is None:
            future_map[
                executor.submit(invoke_travel_tool, "weather", {"city": city})
            ] = "weather"
        if attraction_result is None:
            future_map[
                executor.submit(
                    invoke_travel_tool,
                    "attractions",
                    {
                        "city": city,
                        "preferences": list(_normalize_preferences(parsed_request)),
                    },
                )
            ] = "attractions"
        if restaurant_result is None:
            future_map[
                executor.submit(invoke_travel_tool, "restaurants", {"city": city})
            ] = "restaurants"
        if hotel_result is None:
            future_map[
                executor.submit(invoke_travel_tool, "hotels", {"city": city})
            ] = "hotels"

        for future in as_completed(future_map):
            tool_name = future_map[future]
            result = future.result()
            result = _reuse_previous_on_limited_result(
                tool_name,
                result,
                previous_tool_results,
            )
            if tool_name == "weather":
                weather_result = result
            elif tool_name == "attractions":
                attraction_result = result
            elif tool_name == "restaurants":
                restaurant_result = result
            elif tool_name == "hotels":
                hotel_result = result

    weather_result = weather_result or {}
    attraction_result = attraction_result or {}
    restaurant_result = restaurant_result or {}
    hotel_result = hotel_result or {}

    should_refresh_distances = ENABLE_DISTANCE_TOOL and (
        attraction_result != (previous_tool_results.get("attractions", {}) if previous_tool_results else {})
        or not previous_tool_results.get("distances")
    )

    if ENABLE_DISTANCE_TOOL:
        if should_refresh_distances:
            distance_result = invoke_travel_tool(
                "distances",
                {"attractions": attraction_result.get("attractions", [])},
            )
        else:
            distance_result = _clone_result(previous_tool_results.get("distances", {"routes": []}))
    else:
        distance_result = {"routes": []}

    return _build_tool_results(
        weather_result,
        attraction_result,
        restaurant_result,
        hotel_result,
        distance_result,
    )
