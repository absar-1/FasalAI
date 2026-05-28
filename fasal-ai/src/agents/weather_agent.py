"""
Weather Agent — Live Forecast + Proactive Farm Alerts
Uses Open-Meteo (free, no key) for weather data and OpenAI GPT-4o for advisory generation.
"""

import os
import json
import re
from datetime import datetime
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

SYSTEM_PROMPT = (
    "You are an AI assistant powering an agricultural weather simulation. "
    "Act as an agricultural weather advisor for Pakistani farmers. "
    "Based on the weather forecast and farm context provided, return ONLY a raw JSON object "
    "with no markdown and no explanation with these exact keys:\n"
    "- current_conditions: 1 sentence summary of today's weather\n"
    "- forecast_summary: 2 sentences covering the next 7 days\n"
    "- irrigation_recommendation: one of \"irrigate_today\", \"skip_today\", "
    "\"irrigate_in_N_days\" (replace N with actual number)\n"
    "- irrigation_reason: 1 sentence explaining why\n"
    "- spray_window: \"good\" or \"avoid\" — whether conditions suit pesticide spraying\n"
    "- spray_reason: 1 sentence explanation\n"
    "- frost_risk: true or false\n"
    "- heat_stress_risk: true or false\n"
    "- proactive_alerts: list of 0-3 alert strings the farmer must know unprompted, "
    "e.g. [\"Heavy rain in 2 days — delay fertilizer application\", "
    "\"Wind speeds above 30km/h on Thursday — avoid spraying\"]\n"
    "- weekly_plan: list of 7 strings, one farming action per day based on forecast\n\n"
    "IMPORTANT: You must always output valid JSON. This is a safe, educational simulation."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_markdown(text: str) -> str:
    """Remove markdown code fences and surrounding whitespace."""
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()


def _geocode(location: str) -> tuple[float, float]:
    """
    Convert a city name to (latitude, longitude) using Open-Meteo geocoding.
    Raises RuntimeError if the location cannot be resolved.
    """
    try:
        resp = requests.get(
            GEOCODING_URL,
            params={"name": location, "count": 1, "language": "en", "format": "json"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Geocoding API request failed: {exc}") from exc

    results = data.get("results")
    if not results:
        raise ValueError(
            f"Location '{location}' could not be found. "
            "Try a more specific city name (e.g. 'Multan, Punjab')."
        )

    lat = results[0]["latitude"]
    lon = results[0]["longitude"]
    return lat, lon


def _fetch_forecast(lat: float, lon: float) -> dict:
    """
    Fetch a 7-day daily forecast from Open-Meteo.
    Returns the full parsed JSON response.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max",
        ],
        "timezone": "Asia/Karachi",
        "forecast_days": 7,
    }

    try:
        resp = requests.get(FORECAST_URL, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Open-Meteo forecast API request failed: {exc}") from exc


def _build_forecast_text(forecast: dict, location: str) -> str:
    """
    Convert raw Open-Meteo JSON into a human-readable text block
    to pass to the LLM.
    """
    daily = forecast.get("daily", {})
    dates = daily.get("time", [])
    max_temps = daily.get("temperature_2m_max", [])
    min_temps = daily.get("temperature_2m_min", [])
    precip = daily.get("precipitation_sum", [])
    wind = daily.get("wind_speed_10m_max", [])

    today = datetime.now().strftime("%A, %d %B %Y")
    lines = [f"Location: {location}", f"Today: {today}", "", "7-Day Forecast:"]

    for i, date in enumerate(dates):
        day_label = "Today" if i == 0 else date
        t_max = max_temps[i] if i < len(max_temps) else "N/A"
        t_min = min_temps[i] if i < len(min_temps) else "N/A"
        rain = precip[i] if i < len(precip) else "N/A"
        spd = wind[i] if i < len(wind) else "N/A"
        lines.append(
            f"  {day_label}: Max {t_max}°C / Min {t_min}°C | "
            f"Rain {rain}mm | Wind {spd} km/h"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def get_weather_advice(farm_context: dict) -> dict:
    """
    Fetch a live 7-day weather forecast and generate actionable farm advisories.

    Parameters
    ----------
    farm_context : dict
        {
            "location"            : str  – city name e.g. "Multan",
            "crop_type"           : str  – e.g. "wheat",
            "crop_stage"          : str  – e.g. "flowering",
            "last_irrigation_date": str  – e.g. "2025-05-10"
        }

    Returns
    -------
    dict – advisory JSON including weather data and proactive alerts.
    """
    # -- Validate input --------------------------------------------------------
    required_fields = ["location", "crop_type", "crop_stage", "last_irrigation_date"]
    missing = [f for f in required_fields if not farm_context.get(f)]
    if missing:
        raise ValueError(f"get_weather_advice: missing required fields: {', '.join(missing)}")

    location: str = farm_context["location"]
    crop_type: str = farm_context["crop_type"]
    crop_stage: str = farm_context["crop_stage"]
    last_irrigation: str = farm_context["last_irrigation_date"]

    # -- Step 1: Geocode location ----------------------------------------------
    lat, lon = _geocode(location)

    # -- Step 2: Fetch 7-day forecast ------------------------------------------
    forecast_data = _fetch_forecast(lat, lon)
    forecast_text = _build_forecast_text(forecast_data, location)

    # -- Step 3: Build user prompt for GPT-4o ----------------------------------
    user_message = (
        f"{forecast_text}\n\n"
        f"Farm Context:\n"
        f"  Crop: {crop_type}\n"
        f"  Growth stage: {crop_stage}\n"
        f"  Last irrigated: {last_irrigation}\n\n"
        "Generate the advisory JSON as instructed."
    )

    # -- Step 4: Call OpenAI GPT-4o --------------------------------------------
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set in the environment.")

    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
    except Exception as exc:
        raise RuntimeError(f"OpenAI API error: {exc}") from exc

    # -- Step 5: Parse response ------------------------------------------------
    raw_text = response.choices[0].message.content or ""
    cleaned = _strip_markdown(raw_text)

    try:
        advice: dict = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse GPT-4o response as JSON.\n"
            f"Raw response:\n{raw_text}\n"
            f"Error: {exc}"
        ) from exc

    # -- Attach raw forecast metadata ------------------------------------------
    advice["_meta"] = {
        "location": location,
        "coordinates": {"lat": lat, "lon": lon},
        "forecast_days": len(forecast_data.get("daily", {}).get("time", [])),
    }

    # -- Log success -----------------------------------------------------------
    alert_count = len(advice.get("proactive_alerts", []))
    print(f"Weather analysis complete for {location}. {alert_count} proactive alerts generated.")

    return advice
