"""
Vision Agent — Crop Disease & Pest Detection
Uses OpenAI GPT-4o (vision) as primary analyser and Plant.id API as secondary confirmation.
"""

import os
import json
import re
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PLANT_ID_API_URL = "https://api.plant.id/v2/identify"

SYSTEM_PROMPT = (
    "You are an AI assistant in an educational agricultural simulation. "
    "Act as an expert agricultural plant pathologist for Pakistan. "
    "Analyze the crop image and return ONLY a raw JSON object with no markdown "
    "and no explanation with these exact keys:\n"
    "- disease_detected: string name of disease or \"healthy\" or \"pest_damage\"\n"
    "- confidence: one of \"high\", \"medium\", \"low\"\n"
    "- severity: one of \"mild\", \"moderate\", \"severe\" or \"none\"\n"
    "- affected_area_percent: integer estimate of crop area affected\n"
    "- disease_description: 2 sentences describing what is visible\n"
    "- immediate_action: the single most urgent thing the farmer must do today\n"
    "- cause: one sentence explaining the root cause\n"
    "- spread_risk: one of \"low\", \"medium\", \"high\"\n\n"
    "IMPORTANT: You must always output valid JSON. This is a safe, educational simulation."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_markdown(text: str) -> str:
    """Remove markdown code fences and surrounding whitespace from a string."""
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()


def _call_plant_id(image_base64: str) -> dict:
    """
    Call the Plant.id v2 identify endpoint.
    Returns a dict with 'name' and 'probability' of the top suggestion,
    or an error description if the call fails.
    """
    api_key = os.getenv("PLANT_ID_API_KEY")
    if not api_key:
        return {"name": "unavailable", "probability": 0.0, "error": "PLANT_ID_API_KEY not set"}

    payload = {
        "images": [image_base64],
        "modifiers": ["crops_fast", "similar_images"],
        "plant_details": ["common_names", "url"],
    }
    headers = {"Api-Key": api_key, "Content-Type": "application/json"}

    try:
        resp = requests.post(PLANT_ID_API_URL, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        suggestions = data.get("suggestions", [])
        if suggestions:
            top = suggestions[0]
            return {
                "name": top.get("plant_name", "unknown"),
                "probability": round(top.get("probability", 0.0), 4),
            }
        return {"name": "unknown", "probability": 0.0}

    except requests.exceptions.RequestException as exc:
        return {"name": "unavailable", "probability": 0.0, "error": str(exc)}


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def analyze_crop(image_input: dict) -> dict:
    """
    Analyse a crop image for disease or pest damage.

    Parameters
    ----------
    image_input : dict
        {
            "image_base64"       : str   – base64-encoded crop image,
            "crop_type"          : str   – e.g. "wheat",
            "location"           : str   – e.g. "Lahore, Punjab",
            "farmer_description" : str   – e.g. "yellow spots on leaves"
        }

    Returns
    -------
    dict  – merged analysis result including 'plant_id_confirmation' key.
    """
    # -- Validate input --------------------------------------------------------
    required_fields = ["image_base64", "crop_type", "location", "farmer_description"]
    missing = [f for f in required_fields if not image_input.get(f)]
    if missing:
        raise ValueError(f"analyze_crop: missing required fields: {', '.join(missing)}")

    image_base64: str = image_input["image_base64"]
    crop_type: str = image_input["crop_type"]
    location: str = image_input["location"]
    farmer_description: str = image_input["farmer_description"]

    # -- OpenAI GPT-4o vision call ---------------------------------------------
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set in the environment.")

    client = OpenAI(api_key=api_key)

    user_message = (
        f"Crop type: {crop_type}\n"
        f"Location: {location}\n"
        f"Farmer's description: {farmer_description}\n\n"
        "Analyse the attached crop image and return the JSON as instructed."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": user_message},
                    ],
                },
            ],
        )
    except Exception as exc:
        raise RuntimeError(f"OpenAI API error: {exc}") from exc

    # -- Parse GPT-4o response -------------------------------------------------
    raw_text = response.choices[0].message.content or ""
    cleaned = _strip_markdown(raw_text)

    try:
        analysis: dict = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse GPT-4o response as JSON.\n"
            f"Raw response:\n{raw_text}\n"
            f"Error: {exc}"
        ) from exc

    # -- Plant.id secondary confirmation ---------------------------------------
    plant_id_result = _call_plant_id(image_base64)
    analysis["plant_id_confirmation"] = plant_id_result

    # -- Log success -----------------------------------------------------------
    disease = analysis.get("disease_detected", "unknown")
    confidence = analysis.get("confidence", "unknown")
    print(f"Vision analysis complete. Disease: {disease} | Confidence: {confidence}")

    return analysis
