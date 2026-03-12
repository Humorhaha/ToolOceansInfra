import math
from typing import Any

import httpx

from tooloceans.impl.registry import InMemoryToolRegistry, tool


_WEATHER_CODE_LABELS = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snow fall",
    73: "moderate snow fall",
    75: "heavy snow fall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


def build_registry() -> InMemoryToolRegistry:
    import sys
    registry = InMemoryToolRegistry()
    registry.register_module(sys.modules[__name__])
    return registry


@tool(
    name="resolve_location",
    description="Find the most relevant place match for a location query.",
    input_schema={"query": "string"},
    output_schema={"matches": "array"},
)
async def _resolve_location(args: dict, ctx) -> dict:
    query = str(args.get("query", "")).strip()
    if not query:
        return {"error": "query is required"}

    timeout = ctx.timeout_seconds or 15
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": query, "count": 5, "language": "en", "format": "json"},
            )
            resp.raise_for_status()
            payload = resp.json()
    except httpx.HTTPError as exc:
        return {"query": query, "error": f"location lookup failed: {exc}"}

    results = payload.get("results") or []
    matches = [
        {
            "name": row["name"],
            "country": row.get("country"),
            "admin1": row.get("admin1"),
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "timezone": row.get("timezone"),
        }
        for row in results
    ]
    return {"query": query, "matches": matches}


@tool(
    name="get_current_weather",
    description="Fetch the current weather conditions for given coordinates.",
    input_schema={"latitude": "number", "longitude": "number"},
    output_schema={"current": "object"},
)
async def _get_current_weather(args: dict, ctx) -> dict:
    coordinates = _parse_coordinates(args)
    if "error" in coordinates:
        return coordinates

    timeout = ctx.timeout_seconds or 15
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": coordinates["latitude"],
                    "longitude": coordinates["longitude"],
                    "current": (
                        "temperature_2m,apparent_temperature,relative_humidity_2m,"
                        "precipitation,weather_code,wind_speed_10m,wind_gusts_10m"
                    ),
                    "timezone": "auto",
                    "forecast_days": 1,
                },
            )
            resp.raise_for_status()
            payload = resp.json()
    except httpx.HTTPError as exc:
        return {
            "latitude": coordinates["latitude"],
            "longitude": coordinates["longitude"],
            "error": f"current weather lookup failed: {exc}",
        }

    current = payload.get("current") or {}
    weather_code = current.get("weather_code")
    return {
        "latitude": coordinates["latitude"],
        "longitude": coordinates["longitude"],
        "timezone": payload.get("timezone"),
        "current": {
            "temperature_c": current.get("temperature_2m"),
            "feels_like_c": current.get("apparent_temperature"),
            "humidity_pct": current.get("relative_humidity_2m"),
            "precipitation_mm": current.get("precipitation"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
            "wind_gust_kmh": current.get("wind_gusts_10m"),
            "weather_code": weather_code,
            "condition": _WEATHER_CODE_LABELS.get(weather_code, "unknown"),
            "observed_at": current.get("time"),
        },
    }


@tool(
    name="get_weather_forecast",
    description="Fetch the daily weather forecast for 1-7 days for given coordinates.",
    input_schema={"latitude": "number", "longitude": "number", "days": "integer"},
    output_schema={"forecast_days": "array"},
)
async def _get_weather_forecast(args: dict, ctx) -> dict:
    coordinates = _parse_coordinates(args)
    if "error" in coordinates:
        return coordinates

    days = int(args.get("days", 3))
    days = max(1, min(days, 7))

    timeout = ctx.timeout_seconds or 15
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": coordinates["latitude"],
                    "longitude": coordinates["longitude"],
                    "daily": (
                        "weather_code,temperature_2m_max,temperature_2m_min,"
                        "precipitation_probability_max,precipitation_sum,"
                        "wind_speed_10m_max"
                    ),
                    "timezone": "auto",
                    "forecast_days": days,
                },
            )
            resp.raise_for_status()
            payload = resp.json()
    except httpx.HTTPError as exc:
        return {
            "latitude": coordinates["latitude"],
            "longitude": coordinates["longitude"],
            "error": f"forecast lookup failed: {exc}",
        }

    daily = payload.get("daily") or {}
    forecast_days = []
    for idx, date in enumerate(daily.get("time", [])):
        weather_code = _safe_index(daily.get("weather_code"), idx)
        forecast_days.append(
            {
                "date": date,
                "condition": _WEATHER_CODE_LABELS.get(weather_code, "unknown"),
                "weather_code": weather_code,
                "temp_max_c": _safe_index(daily.get("temperature_2m_max"), idx),
                "temp_min_c": _safe_index(daily.get("temperature_2m_min"), idx),
                "precip_probability_max_pct": _safe_index(daily.get("precipitation_probability_max"), idx),
                "precipitation_sum_mm": _safe_index(daily.get("precipitation_sum"), idx),
                "wind_speed_max_kmh": _safe_index(daily.get("wind_speed_10m_max"), idx),
            }
        )

    return {
        "latitude": coordinates["latitude"],
        "longitude": coordinates["longitude"],
        "timezone": payload.get("timezone"),
        "forecast_days": forecast_days,
    }


@tool(
    name="assess_weather_risk",
    description="Summarize weather risk levels (rain, wind, heat, cold) from a forecast object.",
    input_schema={"forecast": "object"},
    output_schema={"risk_level": "string", "summary": "string"},
)
async def _assess_weather_risk(args: dict, ctx) -> dict:
    forecast = args.get("forecast")
    if not isinstance(forecast, dict):
        return {"error": "forecast object is required"}

    forecast_days = forecast.get("forecast_days")
    if not isinstance(forecast_days, list) or not forecast_days:
        return {"error": "forecast.forecast_days must be a non-empty list"}

    rain_days = []
    wind_days = []
    heat_days = []
    cold_days = []
    summary_labels: list[str] = []

    for day in forecast_days:
        date = day.get("date")
        precip_prob = _as_float(day.get("precip_probability_max_pct"))
        precip_sum = _as_float(day.get("precipitation_sum_mm"))
        wind = _as_float(day.get("wind_speed_max_kmh"))
        temp_max = _as_float(day.get("temp_max_c"))
        temp_min = _as_float(day.get("temp_min_c"))

        if precip_prob >= 60 or precip_sum >= 8:
            rain_days.append(date)
        if wind >= 35:
            wind_days.append(date)
        if temp_max >= 32:
            heat_days.append(date)
        if temp_min <= 0:
            cold_days.append(date)

    if rain_days:
        summary_labels.append("rain risk")
    if wind_days:
        summary_labels.append("wind risk")
    if heat_days:
        summary_labels.append("heat risk")
    if cold_days:
        summary_labels.append("cold risk")
    if not summary_labels:
        summary_labels.append("no major weather risk detected")

    return {
        "risk_level": _classify_risk(
            rain_days=rain_days,
            wind_days=wind_days,
            heat_days=heat_days,
            cold_days=cold_days,
        ),
        "summary": ", ".join(summary_labels),
        "rain_days": rain_days,
        "wind_days": wind_days,
        "heat_days": heat_days,
        "cold_days": cold_days,
    }


def _parse_coordinates(args: dict[str, Any]) -> dict[str, Any]:
    try:
        latitude = float(args["latitude"])
        longitude = float(args["longitude"])
    except KeyError:
        return {"error": "latitude and longitude are required"}
    except (TypeError, ValueError):
        return {"error": "latitude and longitude must be numbers"}

    if not math.isfinite(latitude) or not math.isfinite(longitude):
        return {"error": "latitude and longitude must be finite numbers"}
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        return {"error": "latitude or longitude out of range"}
    return {"latitude": latitude, "longitude": longitude}


def _safe_index(values: list[Any] | None, index: int) -> Any:
    if values is None or index >= len(values):
        return None
    return values[index]


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _classify_risk(
    *,
    rain_days: list[str],
    wind_days: list[str],
    heat_days: list[str],
    cold_days: list[str],
) -> str:
    risk_count = sum(bool(days) for days in (rain_days, wind_days, heat_days, cold_days))
    if risk_count >= 3:
        return "high"
    if risk_count >= 1:
        return "medium"
    return "low"
