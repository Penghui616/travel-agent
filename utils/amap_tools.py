import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from functools import lru_cache
from threading import Lock
from time import perf_counter, sleep
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

from utils.config import get_required_setting, get_setting

load_dotenv()

AMAP_GEO_URL = "https://restapi.amap.com/v3/geocode/geo"
AMAP_WEATHER_URL = "https://restapi.amap.com/v3/weather/weatherInfo"
AMAP_PLACE_URL = "https://restapi.amap.com/v3/place/text"
AMAP_WALKING_URL = "https://restapi.amap.com/v3/direction/walking"
AMAP_DRIVING_URL = "https://restapi.amap.com/v3/direction/driving"

MAX_DISTANCE_ROUTES = max(0, int(get_setting("TRAVEL_AGENT_MAX_DISTANCE_ROUTES", "2")))
MAX_ROUTE_WORKERS = max(1, int(get_setting("TRAVEL_AGENT_MAX_ROUTE_WORKERS", "2")))
AMAP_PLACE_OFFSET = min(25, max(1, int(get_setting("TRAVEL_AGENT_AMAP_PLACE_OFFSET", "20") or "20")))
MAX_POI_SEARCH_WORKERS = max(1, int(get_setting("TRAVEL_AGENT_MAX_POI_SEARCH_WORKERS", "1") or "1"))
ATTRACTION_POI_LIMIT = max(20, int(get_setting("TRAVEL_AGENT_ATTRACTION_POI_LIMIT", "40") or "40"))
RESTAURANT_POI_LIMIT = max(8, int(get_setting("TRAVEL_AGENT_RESTAURANT_POI_LIMIT", "16") or "16"))
HOTEL_POI_LIMIT = max(4, int(get_setting("TRAVEL_AGENT_HOTEL_POI_LIMIT", "10") or "10"))
AMAP_MIN_REQUEST_INTERVAL = max(
    0.0,
    float(get_setting("TRAVEL_AGENT_AMAP_MIN_REQUEST_INTERVAL", "0.35") or "0.35"),
)
AMAP_MAX_RETRIES = max(1, int(get_setting("TRAVEL_AGENT_AMAP_MAX_RETRIES", "3") or "3"))

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "travel-agent/1.0"})
_REQUEST_LOCK = Lock()
_LAST_REQUEST_AT = 0.0


class AMapAPIError(RuntimeError):
    pass


class AMapRateLimitError(AMapAPIError):
    pass


def _is_rate_limit_error(data: Dict[str, Any]) -> bool:
    return data.get("infocode") == "10021" or data.get("info") == "CUQPS_HAS_EXCEEDED_THE_LIMIT"


def _amap_error_message(data: Dict[str, Any]) -> str:
    return f"高德 API 调用失败：{data}"


def _wait_for_rate_limit_slot() -> None:
    global _LAST_REQUEST_AT
    if AMAP_MIN_REQUEST_INTERVAL <= 0:
        return

    with _REQUEST_LOCK:
        elapsed = perf_counter() - _LAST_REQUEST_AT
        if elapsed < AMAP_MIN_REQUEST_INTERVAL:
            sleep(AMAP_MIN_REQUEST_INTERVAL - elapsed)
        _LAST_REQUEST_AT = perf_counter()


def _request(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    request_params = dict(params)
    request_params["key"] = get_required_setting("AMAP_KEY")
    request_params["output"] = "JSON"

    for attempt in range(AMAP_MAX_RETRIES):
        _wait_for_rate_limit_slot()
        resp = _SESSION.get(url, params=request_params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") == "1":
            return data

        if _is_rate_limit_error(data):
            if attempt < AMAP_MAX_RETRIES - 1:
                sleep(min(2.0, 0.6 * (attempt + 1)))
                continue
            raise AMapRateLimitError(_amap_error_message(data))

        raise AMapAPIError(_amap_error_message(data))

    raise AMapAPIError("高德 API 调用失败：超过最大重试次数")


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
    normalized_city = (city or "").strip()
    try:
        return _clone_data(_weather_tool_cached(normalized_city))
    except Exception as exc:
        return {"city": normalized_city, "weather": [], "warning": str(exc)}


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


def _dedupe_pois(pois: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    deduped = []
    seen_names = set()
    for poi in pois:
        name = (poi or {}).get("name", "").strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        deduped.append(poi)
        if len(deduped) >= limit:
            break
    return deduped


def search_poi_many(
    city: str,
    keywords_list: List[str],
    types: Optional[str] = None,
    offset: int = AMAP_PLACE_OFFSET,
    limit: int = 30,
) -> List[Dict[str, Any]]:
    normalized_keywords = [
        keyword.strip()
        for keyword in dict.fromkeys(keywords_list)
        if isinstance(keyword, str) and keyword.strip()
    ]
    if not normalized_keywords:
        return []

    results: List[Dict[str, Any]] = []
    first_error: Optional[Exception] = None
    if MAX_POI_SEARCH_WORKERS <= 1:
        for keyword in normalized_keywords:
            try:
                results.extend(search_poi(city, keyword, types, offset))
            except AMapRateLimitError as exc:
                if first_error is None:
                    first_error = exc
                if results:
                    break
            except Exception as exc:
                if first_error is None:
                    first_error = exc
    else:
        max_workers = min(MAX_POI_SEARCH_WORKERS, len(normalized_keywords))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(search_poi, city, keyword, types, offset): keyword
                for keyword in normalized_keywords
            }
            for future in as_completed(future_map):
                try:
                    results.extend(future.result())
                except Exception as exc:
                    if first_error is None:
                        first_error = exc

    if not results and first_error is not None:
        raise first_error
    return _dedupe_pois(results, limit)


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

    keyword_candidates = [
        keywords,
        "景点",
        "夜景 景点",
        "历史文化 景点",
        "博物馆",
        "商圈",
        "公园 景区",
        "古镇 老街",
    ]
    warning = ""
    try:
        pois = search_poi_many(
            city=city,
            keywords_list=keyword_candidates,
            offset=AMAP_PLACE_OFFSET,
            limit=ATTRACTION_POI_LIMIT,
        )
    except Exception as exc:
        pois = []
        warning = str(exc)

    return {
        "city": city,
        "keywords": "、".join(dict.fromkeys(keyword_candidates)),
        "attractions": pois,
        "warning": warning,
    }


def restaurant_tool(city: str, keyword: str = "美食") -> Dict[str, Any]:
    keyword_candidates = [keyword, "小吃", "火锅", "特色餐厅"]
    warning = ""
    try:
        pois = search_poi_many(
            city=city,
            keywords_list=keyword_candidates,
            offset=AMAP_PLACE_OFFSET,
            limit=RESTAURANT_POI_LIMIT,
        )
    except Exception as exc:
        pois = []
        warning = str(exc)
    return {
        "city": city,
        "keyword": "、".join(dict.fromkeys(keyword_candidates)),
        "restaurants": pois,
        "warning": warning,
    }


def hotel_tool(city: str, keyword: str = "酒店") -> Dict[str, Any]:
    warning = ""
    try:
        pois = search_poi(city=city, keywords=keyword, offset=HOTEL_POI_LIMIT)
    except Exception as exc:
        pois = []
        warning = str(exc)
    return {
        "city": city,
        "keyword": keyword,
        "hotels": pois,
        "warning": warning,
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
