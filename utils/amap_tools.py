import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

from utils.config import get_setting

load_dotenv()

AMAP_KEY = get_setting("AMAP_KEY")

AMAP_GEO_URL = "https://restapi.amap.com/v3/geocode/geo"
AMAP_WEATHER_URL = "https://restapi.amap.com/v3/weather/weatherInfo"
AMAP_PLACE_URL = "https://restapi.amap.com/v3/place/text"
AMAP_WALKING_URL = "https://restapi.amap.com/v3/direction/walking"
AMAP_DRIVING_URL = "https://restapi.amap.com/v3/direction/driving"

MAX_DISTANCE_ROUTES = max(0, int(get_setting("TRAVEL_AGENT_MAX_DISTANCE_ROUTES", "2")))
MAX_ROUTE_WORKERS = max(1, int(get_setting("TRAVEL_AGENT_MAX_ROUTE_WORKERS", "2")))

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "travel-agent/1.0"})


def _request(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if not AMAP_KEY:
        raise ValueError("请先在 .env 中配置 AMAP_KEY")

    request_params = dict(params)
    request_params["key"] = AMAP_KEY
    request_params["output"] = "JSON"

    resp = _SESSION.get(url, params=request_params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "1":
        raise RuntimeError(f"高德 API 调用失败：{data}")

    return data


def _clone_data(data: Any) -> Any:
    return deepcopy(data)


@lru_cache(maxsize=128)
def _geocode_city_cached(city: str) -> Dict[str, Any]:
    data = _request(AMAP_GEO_URL, {"address": city})

    geocodes = data.get("geocodes", [])
    if not geocodes:
        raise ValueError(f"无法找到城市：{city}")

    first = geocodes[0]
    return {
        "city": city,
        "formatted_address": first.get("formatted_address"),
        "province": first.get("province"),
        "adcode": first.get("adcode"),
        "location": first.get("location"),
    }


def geocode_city(city: str) -> Dict[str, Any]:
    return _clone_data(_geocode_city_cached((city or "").strip()))


@lru_cache(maxsize=128)
def _weather_tool_cached(city: str) -> Dict[str, Any]:
    geo = _geocode_city_cached(city)
    adcode = geo["adcode"]

    data = _request(
        AMAP_WEATHER_URL,
        {
            "city": adcode,
            "extensions": "all",
        },
    )

    forecasts = data.get("forecasts", [])
    if not forecasts:
        return {"city": city, "weather": []}

    casts = forecasts[0].get("casts", [])
    weather_list = []
    for item in casts:
        weather_list.append(
            {
                "date": item.get("date"),
                "week": item.get("week"),
                "day_weather": item.get("dayweather"),
                "night_weather": item.get("nightweather"),
                "day_temp": item.get("daytemp"),
                "night_temp": item.get("nighttemp"),
                "day_wind": item.get("daywind"),
                "night_wind": item.get("nightwind"),
                "day_power": item.get("daypower"),
                "night_power": item.get("nightpower"),
            }
        )

    return {
        "city": city,
        "adcode": adcode,
        "weather": weather_list,
    }


def weather_tool(city: str) -> Dict[str, Any]:
    return _clone_data(_weather_tool_cached((city or "").strip()))


@lru_cache(maxsize=256)
def _search_poi_cached(
    city: str,
    keywords: str,
    types: str,
    offset: int,
) -> List[Dict[str, Any]]:
    params = {
        "city": city,
        "keywords": keywords,
        "offset": offset,
        "page": 1,
        "citylimit": "true",
        "extensions": "base",
    }

    if types:
        params["types"] = types

    data = _request(AMAP_PLACE_URL, params)

    pois = []
    for poi in data.get("pois", []):
        pois.append(
            {
                "name": poi.get("name"),
                "type": poi.get("type"),
                "address": poi.get("address"),
                "location": poi.get("location"),
                "adname": poi.get("adname"),
                "pname": poi.get("pname"),
                "cityname": poi.get("cityname"),
            }
        )

    return pois


def search_poi(
    city: str,
    keywords: str,
    types: Optional[str] = None,
    offset: int = 10,
) -> List[Dict[str, Any]]:
    return _clone_data(
        _search_poi_cached(
            (city or "").strip(),
            (keywords or "").strip(),
            (types or "").strip(),
            int(offset),
        )
    )


def attraction_tool(city: str, preferences: Optional[List[str]] = None) -> Dict[str, Any]:
    preferences = preferences or []

    keywords = "景点"
    if "夜景" in preferences:
        keywords = "夜景 景点"
    elif "历史文化" in preferences:
        keywords = "历史文化 景点"
    elif "博物馆" in preferences:
        keywords = "博物馆"
    elif "购物" in preferences:
        keywords = "商圈"
    elif "自然风景" in preferences:
        keywords = "公园 景区"

    pois = search_poi(city=city, keywords=keywords, offset=15)

    return {
        "city": city,
        "keywords": keywords,
        "attractions": pois,
    }


def restaurant_tool(city: str, keyword: str = "美食") -> Dict[str, Any]:
    pois = search_poi(city=city, keywords=keyword, offset=10)
    return {
        "city": city,
        "keyword": keyword,
        "restaurants": pois,
    }


def hotel_tool(city: str, keyword: str = "酒店") -> Dict[str, Any]:
    pois = search_poi(city=city, keywords=keyword, offset=10)
    return {
        "city": city,
        "keyword": keyword,
        "hotels": pois,
    }


@lru_cache(maxsize=256)
def _route_distance_tool_cached(
    origin: str,
    destination: str,
    mode: str,
) -> Dict[str, Any]:
    url = AMAP_DRIVING_URL if mode == "driving" else AMAP_WALKING_URL

    data = _request(
        url,
        {
            "origin": origin,
            "destination": destination,
        },
    )

    route = data.get("route", {})
    paths = route.get("paths", [])

    if not paths:
        return {
            "origin": origin,
            "destination": destination,
            "mode": mode,
            "distance": None,
            "duration": None,
        }

    path = paths[0]
    return {
        "origin": origin,
        "destination": destination,
        "mode": mode,
        "distance_meters": path.get("distance"),
        "duration_seconds": path.get("duration"),
    }


def route_distance_tool(
    origin: str,
    destination: str,
    mode: str = "walking",
) -> Dict[str, Any]:
    return _clone_data(
        _route_distance_tool_cached(
            (origin or "").strip(),
            (destination or "").strip(),
            (mode or "walking").strip(),
        )
    )


def distance_tool(attractions: List[Dict[str, Any]]) -> Dict[str, Any]:
    results = []
    valid = [a for a in attractions if a.get("location")]
    max_pairs = min(max(len(valid) - 1, 0), MAX_DISTANCE_ROUTES)

    if max_pairs <= 0:
        return {"routes": results}

    route_pairs: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = []
    for index in range(max_pairs):
        route_pairs.append((index, valid[index], valid[index + 1]))

    ordered_results: Dict[int, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(MAX_ROUTE_WORKERS, len(route_pairs))) as executor:
        future_map = {
            executor.submit(
                route_distance_tool,
                origin_poi["location"],
                destination_poi["location"],
                "driving",
            ): (index, origin_poi, destination_poi)
            for index, origin_poi, destination_poi in route_pairs
        }

        for future in as_completed(future_map):
            index, origin_poi, destination_poi = future_map[future]
            try:
                route = future.result()
                route["from_name"] = origin_poi["name"]
                route["to_name"] = destination_poi["name"]
                ordered_results[index] = route
            except Exception as e:
                ordered_results[index] = {
                    "from_name": origin_poi["name"],
                    "to_name": destination_poi["name"],
                    "error": str(e),
                }

    for index in sorted(ordered_results.keys()):
        results.append(ordered_results[index])

    return {"routes": results}
