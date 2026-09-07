"""Open-Meteo weather tool with integrated geocoding, WMO code interpretation, and caching."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Dict, Optional, Tuple
import requests

from cache.ttl_cache import cache
from config.logging_config import logger


WMO_WEATHER_CODES: Dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


class WeatherService:
    """Service to fetch live meteorological data from Open-Meteo."""

    GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    def extract_location(self, query: str) -> Optional[str]:
        """Extract likely location name from a natural language question."""
        patterns = [
            r"weather\s+(?:in|for|at|around)\s+([a-zA-Z\s]+?)(?:\s+right\s+now|\s+today|\s+currently|\?|$)",
            r"temperature\s+(?:in|for|at)\s+([a-zA-Z\s]+?)(?:\s+right\s+now|\s+today|\s+currently|\?|$)",
            r"forecast\s+(?:in|for)\s+([a-zA-Z\s]+?)(?:\s+today|\?|$)",
            r"(?:in|for)\s+([a-zA-Z\s]+?)\s+weather",
        ]
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                # Exclude common question stop words
                if candidate.lower() not in {"what", "how", "the", "now", "today", "currently"}:
                    return candidate

        # Fallback: remove stop words and look for noun phrase
        cleaned = re.sub(r"[?!.,]", "", query)
        words = cleaned.split()
        for i, w in enumerate(words):
            if w.lower() in ("in", "for", "at") and i + 1 < len(words):
                return " ".join(words[i + 1 :])

        return None

    def geocode(self, location_name: str) -> Optional[Tuple[float, float, str, str]]:
        """Resolve location name into (latitude, longitude, resolved_name, country)."""
        cache_key = f"geocode:{location_name.lower().strip()}"
        cached = cache.get(cache_key)
        if cached:
            return tuple(cached)  # type: ignore

        try:
            resp = requests.get(
                self.GEOCODING_URL,
                params={"name": location_name, "count": 1, "language": "en", "format": "json"},
                timeout=6.0,
                headers={"User-Agent": "GroundLens/1.0"},
            )
            if resp.status_code != 200:
                logger.warning("Geocoding API failed (%s): %s", resp.status_code, resp.text[:150])
                return None

            data = resp.json()
            results = data.get("results")
            if not results:
                logger.info("No geocoding results found for '%s'", location_name)
                return None

            top = results[0]
            lat = float(top["latitude"])
            lon = float(top["longitude"])
            name = str(top.get("name", location_name))
            country = str(top.get("country", ""))

            res = (lat, lon, name, country)
            cache.set(cache_key, list(res), ttl_seconds=86400)  # Geocoding cached for 24h
            return res

        except Exception as e:
            logger.error("Geocoding error for '%s': %s", location_name, e)
            return None

    def get_weather(self, location_query: str) -> Optional[Dict[str, Any]]:
        """Retrieve live weather for a location query.

        Returns normalized dictionary with unique source_id openmeteo_01.
        Never invents missing weather values.
        """
        loc_name = self.extract_location(location_query) or location_query.strip()
        geo = self.geocode(loc_name)
        if not geo:
            logger.warning("Unable to geocode location: '%s'", loc_name)
            return None

        lat, lon, resolved_city, country = geo
        cache_key = f"weather:{lat:.2f}:{lon:.2f}"
        cached = cache.get(cache_key)
        if cached:
            return cached

        try:
            resp = requests.get(
                self.FORECAST_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                    "timezone": "auto",
                },
                timeout=6.0,
                headers={"User-Agent": "GroundLens/1.0"},
            )
            if resp.status_code != 200:
                logger.warning("Open-Meteo forecast API error %s: %s", resp.status_code, resp.text[:150])
                return None

            data = resp.json()
            current = data.get("current", {})
            temp = current.get("temperature_2m")
            humidity = current.get("relative_humidity_2m")
            wind_speed = current.get("wind_speed_10m")
            weather_code = current.get("weather_code")

            if temp is None:
                logger.warning("Missing temperature value in Open-Meteo response.")
                return None

            condition = WMO_WEATHER_CODES.get(weather_code, "Unknown weather condition")
            loc_label = f"{resolved_city}, {country}" if country else resolved_city
            now_iso = datetime.now(timezone.utc).isoformat()

            content = (
                f"Location: {loc_label} (Lat: {lat:.2f}, Lon: {lon:.2f}). "
                f"Current temperature: {temp}°C. "
                f"Relative humidity: {humidity}%. "
                f"Wind speed: {wind_speed} km/h. "
                f"Current condition: {condition}."
            )

            result: Dict[str, Any] = {
                "source_id": "openmeteo_01",
                "source_type": "open_meteo",
                "title": f"Live Weather for {loc_label}",
                "content": content,
                "url": f"https://open-meteo.com/en/docs#latitude={lat:.2f}&longitude={lon:.2f}",
                "location": loc_label,
                "temperature": float(temp),
                "humidity": int(humidity) if humidity is not None else None,
                "wind_speed": float(wind_speed) if wind_speed is not None else None,
                "weather_code": weather_code,
                "condition": condition,
                "retrieved_at": now_iso,
                "metadata": {
                    "latitude": lat,
                    "longitude": lon,
                    "elevation": data.get("elevation"),
                    "timezone": data.get("timezone"),
                },
            }

            # Cache weather for 10 minutes
            cache.set(cache_key, result, ttl_seconds=600)
            return result

        except Exception as e:
            logger.error("Open-Meteo request error for '%s': %s", loc_name, e)
            return None


# Module singleton
weather_service = WeatherService()
