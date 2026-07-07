from typing import Any, Dict, List

from langchain_core.tools import tool

from utils.amap_tools import (
    attraction_tool as _attraction_tool,
    distance_tool as _distance_tool,
    hotel_tool as _hotel_tool,
    restaurant_tool as _restaurant_tool,
    weather_tool as _weather_tool,
)


@tool("weather_tool")
def langchain_weather_tool(city: str) -> Dict[str, Any]:
    """Query weather information for a destination city."""
    return _weather_tool(city)


@tool("attraction_tool")
def langchain_attraction_tool(city: str, preferences: List[str] | None = None) -> Dict[str, Any]:
    """Search attractions for a destination city and user preferences."""
    return _attraction_tool(city, preferences or [])


@tool("restaurant_tool")
def langchain_restaurant_tool(city: str) -> Dict[str, Any]:
    """Search restaurants for a destination city."""
    return _restaurant_tool(city)


@tool("hotel_tool")
def langchain_hotel_tool(city: str) -> Dict[str, Any]:
    """Search hotels for a destination city."""
    return _hotel_tool(city)


@tool("distance_tool")
def langchain_distance_tool(attractions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Estimate routes between attraction candidates."""
    return _distance_tool(attractions)


LANGCHAIN_TRAVEL_TOOLS = {
    "weather": langchain_weather_tool,
    "attractions": langchain_attraction_tool,
    "restaurants": langchain_restaurant_tool,
    "hotels": langchain_hotel_tool,
    "distances": langchain_distance_tool,
}


def invoke_travel_tool(name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return LANGCHAIN_TRAVEL_TOOLS[name].invoke(payload)
