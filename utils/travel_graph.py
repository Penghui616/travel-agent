import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional, Tuple, TypedDict

from utils.amap_tools import (
    attraction_tool,
    distance_tool,
    hotel_tool,
    restaurant_tool,
    weather_tool,
)


MAX_TOOL_WORKERS = max(2, int(os.getenv("TRAVEL_AGENT_MAX_TOOL_WORKERS", "4")))
ENABLE_DISTANCE_TOOL = os.getenv("TRAVEL_AGENT_ENABLE_DISTANCE", "1").lower() not in {
    "0",
    "false",
    "no",
}


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
    state["weather_result"] = weather_tool(city)
    return state


def run_attractions(state: TravelState) -> TravelState:
    parsed_request = state["parsed_request"]
    state["attraction_result"] = attraction_tool(
        _normalize_city(parsed_request),
        list(_normalize_preferences(parsed_request)),
    )
    return state


def run_restaurants(state: TravelState) -> TravelState:
    city = _normalize_city(state["parsed_request"])
    state["restaurant_result"] = restaurant_tool(city)
    return state


def run_hotels(state: TravelState) -> TravelState:
    city = _normalize_city(state["parsed_request"])
    state["hotel_result"] = hotel_tool(city)
    return state


def run_distances(state: TravelState) -> TravelState:
    attractions = state.get("attraction_result", {}).get("attractions", [])
    state["distance_result"] = distance_tool(attractions) if ENABLE_DISTANCE_TOOL else {"routes": []}
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
    return None


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
            future_map[executor.submit(weather_tool, city)] = "weather"
        if attraction_result is None:
            future_map[
                executor.submit(
                    attraction_tool,
                    city,
                    list(_normalize_preferences(parsed_request)),
                )
            ] = "attractions"
        if restaurant_result is None:
            future_map[executor.submit(restaurant_tool, city)] = "restaurants"
        if hotel_result is None:
            future_map[executor.submit(hotel_tool, city)] = "hotels"

        for future in as_completed(future_map):
            tool_name = future_map[future]
            result = future.result()
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
            distance_result = distance_tool(attraction_result.get("attractions", []))
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
