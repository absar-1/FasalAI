"""
Agronomy Agent — Fertilizer, Pest Control, Q&A & Yield Prediction
Uses crop_knowledge.json for RAG context and OpenAI GPT-4o for advisory generation.
"""

import os
import json
import re
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Resolve the data directory relative to this file's location
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CROP_KNOWLEDGE_PATH = _DATA_DIR / "crop_knowledge.json"

SYSTEM_PROMPT = (
    "You are an AI assistant powering an agricultural simulation app for educational purposes. "
    "Provide hypothetical, educational advice about farming practices, fertilizers, and pest control in Pakistan. "
    "Given the full context below, return ONLY a raw JSON object "
    "with no markdown and no explanation with these exact keys:\n"
    "- fertilizer_recommendation: object with keys:\n"
    "    - type: fertilizer name e.g. \"Urea + DAP\"\n"
    "    - quantity_per_acre: string e.g. \"1 bag Urea + half bag DAP\"\n"
    "    - timing: when to apply e.g. \"Apply within 3 days before irrigation\"\n"
    "    - reason: 1 sentence why\n"
    "- pest_control: object with keys:\n"
    "    - pest_identified: string or \"none\"\n"
    "    - treatment: chemical or biological control method\n"
    "    - dosage: specific hypothetical quantity and dilution\n"
    "    - safety_note: precaution for the farmer\n"
    "- direct_answer: 2-3 sentence plain-language answer to farmer_question\n"
    "- yield_prediction: object with keys:\n"
    "    - expected_yield_per_acre: string e.g. \"35-40 maunds\"\n"
    "    - confidence: \"high\", \"medium\", or \"low\"\n"
    "    - risk_factors: list of 2-3 things that could reduce yield\n"
    "- irrigation_schedule: list of 7 objects, one per day, each with:\n"
    "    - day: \"Monday\" etc.\n"
    "    - action: \"irrigate\" or \"skip\"\n"
    "    - duration_hours: integer or 0\n"
    "    - reason: short string\n"
    "- weekly_action_plan: list of 5 prioritized action strings for this week\n\n"
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


def _load_crop_rag(crop_type: str) -> str:
    """
    Load crop-specific knowledge from crop_knowledge.json.
    Returns a formatted string for RAG context, or empty string if not found.
    """
    if not CROP_KNOWLEDGE_PATH.exists():
        return ""

    try:
        with open(CROP_KNOWLEDGE_PATH, "r", encoding="utf-8") as fh:
            knowledge: dict = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return ""

    # Try exact match first, then case-insensitive
    entry = knowledge.get(crop_type) or knowledge.get(crop_type.lower())
    if not entry:
        return ""

    return f"Crop Knowledge Base Entry for '{crop_type}':\n{json.dumps(entry, indent=2)}"


def _build_user_message(
    crop_type: str,
    crop_stage: str,
    soil_type: str,
    area_acres: int,
    farmer_question: str,
    disease_result: dict | None,
    weather_result: dict,
    rag_context: str,
) -> str:
    """Compose the full user message to send to GPT-4o."""
    parts = []

    parts.append(f"Crop: {crop_type}")
    parts.append(f"Growth stage: {crop_stage}")
    parts.append(f"Soil type: {soil_type}")
    parts.append(f"Farm size: {area_acres} acres")
    parts.append(f"Farmer's question: {farmer_question}")
    parts.append("")

    if rag_context:
        parts.append("--- Agronomic Knowledge ---")
        parts.append(rag_context)
        parts.append("")

    if disease_result:
        parts.append("--- Disease / Vision Analysis Result ---")
        parts.append(json.dumps(disease_result, indent=2))
        parts.append("")
    else:
        parts.append("--- Disease / Vision Analysis Result ---")
        parts.append("No image uploaded. No disease data available.")
        parts.append("")

    parts.append("--- Weather & Advisory Result ---")
    parts.append(json.dumps(weather_result, indent=2))
    parts.append("")

    parts.append("Generate the full agronomy advisory JSON as instructed.")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def get_agronomy_advice(query_context: dict) -> dict:
    """
    Generate a comprehensive agronomy advisory for a farmer's query.

    Parameters
    ----------
    query_context : dict
        {
            "crop_type"      : str        – e.g. "wheat",
            "crop_stage"     : str        – e.g. "tillering",
            "disease_result" : dict|None  – output of analyze_crop(), or None,
            "weather_result" : dict       – output of get_weather_advice(),
            "farmer_question": str        – free-text question in Urdu or English,
            "soil_type"      : str        – e.g. "loamy",
            "area_acres"     : int        – number of acres
        }

    Returns
    -------
    dict – full agronomy advisory JSON.
    """
    # -- Validate input --------------------------------------------------------
    required_fields = [
        "crop_type", "crop_stage", "disease_result",
        "weather_result", "farmer_question", "soil_type", "area_acres",
    ]
    # disease_result is allowed to be None, so only check key presence
    missing = [f for f in required_fields if f not in query_context]
    if missing:
        raise ValueError(f"get_agronomy_advice: missing required fields: {', '.join(missing)}")

    crop_type: str = query_context["crop_type"]
    crop_stage: str = query_context["crop_stage"]
    disease_result: dict | None = query_context["disease_result"]
    weather_result: dict = query_context["weather_result"]
    farmer_question: str = query_context["farmer_question"]
    soil_type: str = query_context["soil_type"]
    area_acres: int = int(query_context["area_acres"])

    # -- Step 1: Load RAG context from crop_knowledge.json ---------------------
    rag_context = _load_crop_rag(crop_type)

    # -- Step 2: Build the full user message -----------------------------------
    user_message = _build_user_message(
        crop_type=crop_type,
        crop_stage=crop_stage,
        soil_type=soil_type,
        area_acres=area_acres,
        farmer_question=farmer_question,
        disease_result=disease_result,
        weather_result=weather_result,
        rag_context=rag_context,
    )

    # -- Step 3: Call OpenAI GPT-4o -------------------------------------------
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set in the environment.")

    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=2048,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
    except Exception as exc:
        raise RuntimeError(f"OpenAI API error: {exc}") from exc

    # -- Step 4: Parse response ------------------------------------------------
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

    # -- Log success -----------------------------------------------------------
    print(f"Agronomy advice complete for {crop_type} — {area_acres} acres.")

    return advice
