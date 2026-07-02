import requests

from utils.config import get_required_setting, get_setting


def generate_weather_advice(forecast: list):
    advice = []

    if not forecast:
        return ["暂时没有获取到天气数据。"]

    has_rain = any("雨" in str(day.get("condition_day", "")) or "雨" in str(day.get("condition_night", "")) for day in forecast)
    high_temp = any(int(day.get("temp_day", 0)) >= 30 for day in forecast if str(day.get("temp_day", "")).isdigit())
    low_temp = any(int(day.get("temp_night", 99)) <= 10 for day in forecast if str(day.get("temp_night", "")).isdigit())

    if has_rain:
        advice.append("行程里建议带伞，并预留一些室内景点。")
    else:
        advice.append("整体较适合户外活动和 citywalk。")

    if high_temp:
        advice.append("白天偏热，注意防晒和补水。")

    if low_temp:
        advice.append("早晚偏凉，建议带薄外套。")

    advice.append("夜景类行程适合安排在傍晚到晚上。")
    return advice


def get_weather(city: str, start_date: str = "", days: int = 1):
    """
    使用高德天气 API 查询天气
    city: 城市名或 adcode，比如 '重庆' 或 '500000'
    days: 计划展示几天，实际高德返回格式以接口为准
    """
    url = "https://restapi.amap.com/v3/weather/weatherInfo"
    api_key = get_setting("WEATHER_API_KEY") or get_required_setting("AMAP_KEY")

    params = {
        "key": api_key,
        "city": city,
        "extensions": "all",   # 预报天气；如果想查实时天气改成 base
        "output": "JSON",
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    # 高德返回 status=1 表示成功
    if data.get("status") != "1":
        raise ValueError(f"高德天气 API 调用失败: {data}")

    forecasts = data.get("forecasts", [])
    if not forecasts:
        raise ValueError(f"没有查到 {city} 的天气信息: {data}")

    cast_list = forecasts[0].get("casts", [])

    # 按用户需要截取天数
    cast_list = cast_list[:days] if days and days > 0 else cast_list

    forecast = []
    for day in cast_list:
        forecast.append({
            "date": day.get("date"),
            "week": day.get("week"),
            "dayweather": day.get("dayweather"),
            "nightweather": day.get("nightweather"),
            "daytemp": day.get("daytemp"),
            "nighttemp": day.get("nighttemp"),
            "daywind": day.get("daywind"),
            "nightwind": day.get("nightwind"),
        })

    # 为了和你前面规划模块更好兼容，再整理一份统一字段
    normalized_forecast = []
    for item in forecast:
        normalized_forecast.append({
            "date": item["date"],
            "condition_day": item["dayweather"],
            "condition_night": item["nightweather"],
            "temp_day": item["daytemp"],
            "temp_night": item["nighttemp"],
            "wind_day": item["daywind"],
            "wind_night": item["nightwind"],
        })

    summary = f"{city}未来{len(normalized_forecast)}天天气已获取。"

    weather_result = {
        "city": city,
        "start_date": start_date or None,
        "forecast": normalized_forecast,
        "summary": summary,
        "advice": generate_weather_advice(normalized_forecast),
        "raw": data,
    }

    return weather_result
